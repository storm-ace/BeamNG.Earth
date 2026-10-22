import json
import os
import sys

import pytest

TOOLS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REPO = os.path.dirname(TOOLS)
for p in (TOOLS, os.path.join(REPO, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

ORIGIN = [91597.0, 435902.0]


def _w(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture
def world(tmp_path):
    """A small kunstwerk 'kw': its ref files, an export, and a Street View scan (objects, shots, one picture)."""
    import ahn
    design = tmp_path / "design"
    ref = design / "kunstwerken" / "kw" / "ref"
    _w(str(ref / "ref.json"), {"origin_rd": ORIGIN, "box": [-160, -160, 160, 160], "features": []})
    _w(str(ref / "placement.json"), {"items": [
        {"module": "traffic_light", "role": "primary", "at": [10.0, 5.0], "facing": [1.0, 0.0], "bike": False},
        {"module": "sign", "code": "C2", "role": "ndw", "at": [-3.04, 7.0], "facing": [0.0, -1.0]},
        {"module": "sign", "code": "D2", "role": "island", "at": [20.0, 20.0], "facing": [0.0, 1.0]},
    ]})
    _w(str(ref / "placement_fix.json"), {"fixes": [
        {"action": "move", "near": [20.5, 20.0], "module": "sign", "at": [21.0, 22.0], "why": "test"}]})
    _w(str(ref / "direction_signs.json"), {"signs": [
        {"module": "finger_post", "args": [["a"]], "at": [1.0, 2.0], "facing": [0.0, 1.0]}]})
    _w(str(ref / "approaches.json"), {"approaches": [{"name": "east", "left": [30.0, 0.0], "right": [30.0, 10.0]}]})
    lat, lon = ahn.rd_to_wgs(ORIGIN[0] + 4.0, ORIGIN[1] + 6.0)
    _w(str(ref / "osm_signals.json"), {"elements": [
        {"type": "node", "id": 42, "lat": float(lat), "lon": float(lon), "tags": {"highway": "traffic_signals"}}]})
    export = tmp_path / "export"
    _w(str(export / "kw.kunstwerk.json"), {
        "name": "kw", "origin_rd": ORIGIN, "origin_wgs": [51.9, 4.46],
        "positions": [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 2]], "tris": [[0, 1, 2], [1, 3, 2]],
        "materials": ["road_asphalt_2lane", "bne_grass"], "tri_material": [1, 0],
        "signals": [{"at": [1.0, 2.0, 5.0], "facing": [0.0, 1.0]}], "wires": []})
    sv = design / "reference" / "sv" / "kw"
    slat, slon = ahn.rd_to_wgs(ORIGIN[0] + 10.5, ORIGIN[1] + 5.0)
    _w(str(sv / "objects.geojson"), {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(slon), float(slat)]},
         "properties": {"id": "sv-1", "cls": "light", "sign_code": None, "n_views": 3, "residual_m": 0.2,
                        "height_m": 3.0, "facing": 90.0, "shots": ["S1"], "verdict": "ok",
                        "match": {"source": "placement", "ref": "items[0]", "dist_m": 0.5}, "note": ""}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(slon), float(slat)]},
         "properties": {"id": "sv-2", "cls": "light", "verdict": "missing_in_ours",
                        "match": {"source": "osm", "ref": "node/42", "dist_m": 3.0}}}]})
    _w(str(sv / "shots.json"), [{"id": "S1", "lat": float(slat), "lon": float(slon), "heading": 45.0, "pitch": 0,
                                 "fov": 90, "w": 640, "h": 400}])
    os.makedirs(sv / "shots", exist_ok=True)
    (sv / "shots" / "S1.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    from dashboard.data import Config
    cfg = Config(repo=REPO, design=str(design), name="kw", export_dir=str(export))
    return cfg
