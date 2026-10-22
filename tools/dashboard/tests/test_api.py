import json
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

from dashboard.app import create_app


@pytest.fixture
def client(world):
    return TestClient(create_app(world))


def test_page_and_info(client, world):
    r = client.get("/")
    assert r.status_code == 200 and "Kunstwerk-dashboard" in r.text and "three@0.160.0" in r.text
    assert client.get("/static/app.js").status_code == 200
    info = client.get("/api/info").json()
    assert info["name"] == "kw" and info["mesh_present"] and info["sv_places"] == ["kw"]
    assert info["origin_rd"] == [91597.0, 435902.0]


def test_objects(client):
    d = client.get("/api/objects").json()
    L = d["layers"]
    assert {o["id"] for o in L["ours"]} == {"traffic_light:primary@10.0,5.0", "sign:C2@-3.0,7.0",
                                             "sign:D2@21.0,22.0"}
    assert [o["id"] for o in L["direction"]] == ["finger_post:@1.0,2.0"]
    assert L["heads"][0]["z"] == 5.0 and L["approaches"][0]["heading"] == pytest.approx(270)
    osm = L["osm_signals"][0]
    assert osm["id"] == "osm:42" and osm["at"] == pytest.approx([4.0, 6.0], abs=1e-3)
    assert L["ndw"] == [] and L["furniture"] == []            # (files absent: empty layers)
    tl = next(o for o in L["ours"] if o["kind"] == "traffic_light")
    assert tl["heading"] == pytest.approx(90) and tl["editable"]


def test_mesh_bin(client):
    b = client.get("/api/mesh.bin").content
    n = struct.unpack("<I", b[:4])[0]
    head = json.loads(b[4:4 + n])
    assert head["nverts"] == 4 and head["ntris"] == 2
    # grouped by material: bne_grass's triangle (index 1) after road's (index 0)
    assert [(g["material"], g["start"], g["count"]) for g in head["groups"]] == [
        ("road_asphalt_2lane", 0, 3), ("bne_grass", 3, 3)]
    pos = np.frombuffer(b, np.float32, 12, 4 + n).reshape(4, 3)
    idx = np.frombuffer(b, np.uint32, 6, 4 + n + 48).reshape(2, 3)
    assert pos[3].tolist() == [1, 1, 2] and idx.tolist() == [[1, 3, 2], [0, 1, 2]]


def test_ground_absent(client):
    assert client.get("/api/ground").json() == {"present": False}
    assert client.get("/api/ground/photo.jpg").status_code == 404


def test_sv(client):
    d = client.get("/api/sv").json()
    assert d["present"] and [o["id"] for o in d["objects"]] == ["sv:sv-1", "sv:sv-2"]
    o1, o2 = d["objects"]
    assert o1["match_id"] == "traffic_light:primary@10.0,5.0" and o1["verdict"] == "ok"
    assert o1["at"] == pytest.approx([10.5, 5.0], abs=1e-3) and o1["heading"] == pytest.approx(90)
    assert o2["match_id"] == "osm:42" and o2["verdict"] == "missing_in_ours"
    s = d["shots"][0]
    assert s["id"] == "S1" and s["png"] and not s["annotated"] and s["heading"] == 45
    assert client.get("/api/sv/shot/kw/S1.png").content.startswith(b"\x89PNG")
    assert client.get("/api/sv/shot/kw/S1.png?annotated=1").status_code == 404
    assert client.get("/api/sv/shot/kw/..%5Cref.png").status_code in (400, 404)
    assert client.get("/api/sv?place=..").status_code == 400
    assert client.get("/api/sv?place=elsewhere").json() == {"place": "elsewhere", "present": False,
                                                            "objects": [], "shots": []}


def test_overrides_api(client, world):
    r = client.post("/api/overrides", json={"id": "sign:nope@0.0,0.0", "heading": 3})
    assert r.status_code == 400
    r = client.post("/api/overrides", json={"id": "sign:C2@-3.0,7.0", "at": [0, 0], "code": "C3"})
    assert r.status_code == 200
    with open(world.overrides_path, encoding="utf-8") as f:
        assert json.load(f) == [{"id": "sign:C2@-3.0,7.0", "at": [0.0, 0.0], "code": "C3"}]
    o = next(o for o in client.get("/api/objects").json()["layers"]["ours"] if o["id"] == "sign:C2@-3.0,7.0")
    assert o["at"] == [0.0, 0.0] and o["label"] == "C3"
    assert client.get("/api/overrides").json()["items"][0]["code"] == "C3"
    assert client.post("/api/overrides/delete", json={"id": "sign:C2@-3.0,7.0"}).json() == {"items": []}
    assert client.post("/api/overrides/delete", json={"id": "sign:C2@-3.0,7.0"}).status_code == 404


def test_rebuild_streams(world):
    world.steps = [("een", [sys.executable, "-c", "print('hallo'); print('wereld')"]),
                   ("twee", [sys.executable, "-c", "import sys; print('stuk'); sys.exit(3)"]),
                   ("drie", [sys.executable, "-c", "print('nooit')"])]
    c = TestClient(create_app(world))
    job = c.post("/api/rebuild", json={}).json()
    assert job["steps"] == ["een", "twee", "drie"]
    with c.stream("GET", f"/api/rebuild/{job['job']}/events") as r:
        body = "".join(r.iter_text())
    lines = [json.loads(x[6:])["line"] for x in body.split("\n") if x.startswith("data: {\"line")]
    assert "hallo" in lines and "wereld" in lines and "stuk" in lines and "nooit" not in lines
    assert any(ln.startswith("== twee: exit 3") for ln in lines)
    assert "event: done" in body and '"ok": false' in body
    assert c.get("/api/rebuild/status").json()["ok"] is False
    assert c.get("/api/rebuild/nope/events").status_code == 404


def test_rebuild_one_at_a_time(world):
    world.steps = [("slaap", [sys.executable, "-c", "import time; time.sleep(1.5)"])]
    c = TestClient(create_app(world))
    assert c.post("/api/rebuild", json={}).status_code == 200
    assert c.post("/api/rebuild", json={}).status_code == 409


def test_default_steps(world):
    from dashboard.rebuild import default_steps
    names = [s[0] for s in default_steps(world, placement=True)]
    assert names == ["plaatsing", "model", "export"]
    export = default_steps(world)[-1][1]
    assert export[2].endswith("kunstwerk_export.py") and export[-1] == world.export_dir
    assert [s[0] for s in default_steps(world)] == ["model", "export"]


class _FakeMcp(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeMcp.calls.append(body)
        code = body["params"]["arguments"]["code"]
        text = "earth" if "~=" in code else "ok: road tiles requested again"
        out = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": text}]}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_game_push(world):
    srv = HTTPServer(("127.0.0.1", 0), _FakeMcp)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        world.mcp_url = f"http://127.0.0.1:{srv.server_port}/mcp"
        c = TestClient(create_app(world))
        assert c.get("/api/game/status").json()["earth"] is True
        r = c.post("/api/game/push").json()
        assert r == {"ok": True, "result": "ok: road tiles requested again"}
        sent = _FakeMcp.calls[-1]
        assert sent["method"] == "tools/call" and sent["params"]["name"] == "run_lua"
        assert "setRoadsEnabled(false)" in sent["params"]["arguments"]["code"]
        assert "setRoadsEnabled(true)" in sent["params"]["arguments"]["code"]
    finally:
        srv.shutdown()


def test_game_unreachable(world):
    world.mcp_url = "http://127.0.0.1:9/mcp"
    c = TestClient(create_app(world))
    r = c.post("/api/game/push")
    assert r.status_code == 502 and r.json()["ok"] is False
    assert c.get("/api/game/status").json()["reachable"] is False
