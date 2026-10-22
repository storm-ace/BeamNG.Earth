"""tools/svscan.py's geometry, URL reading, triangulation and matching on a synthetic place (no images, no YOLO).

    python -m pytest tools/tests/test_svscan.py
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import svscan as sv  # noqa: E402

LAT0, LON0 = 51.908, 4.4658
W, H, FOV = 1600, 900, 100.0


def test_pixel_ray_centre_and_edge():
    b, e = sv.pixel_ray(W / 2, H / 2, W, H, 269.0, 0.0, FOV)
    assert b == pytest.approx(269.0) and e == pytest.approx(0.0, abs=1e-9)
    b, e = sv.pixel_ray(W, H / 2, W, H, 10.0, 0.0, FOV)
    assert b == pytest.approx(60.0)                     # the right edge: heading + fov / 2
    b, e = sv.pixel_ray(W / 2, H / 2, W, H, 0.0, 5.0, FOV)
    assert e == pytest.approx(5.0)


@pytest.mark.parametrize("bearing,elev,heading,pitch", [(300.0, 4.0, 269.0, 2.0), (12.0, -3.0, 355.0, -1.5),
                                                       (95.0, 10.0, 120.0, 6.0)])
def test_pixel_round_trip(bearing, elev, heading, pitch):
    u, v = sv.world_to_pixel(bearing, elev, W, H, heading, pitch, FOV)
    b, e = sv.pixel_ray(u, v, W, H, heading, pitch, FOV)
    assert sv.wrap(b - bearing) == pytest.approx(0.0, abs=1e-6)
    assert e == pytest.approx(elev, abs=1e-6)


def test_parse_url_and_record():
    url = "https://www.google.com/maps/@51.908,4.466,3a,75y,269h,92t/data=!3m6!1e1!3m4!1sAbCdEfGhIjKlMnOp!2e0"
    u = sv.parse_url(url)
    assert (u["lat"], u["lon"], u["fov_y"], u["heading"], u["tilt"]) == (51.908, 4.466, 75.0, 269.0, 92.0)
    assert u["pano"] == "AbCdEfGhIjKlMnOp"
    r = sv.shot_record(url, "A269", 1600, 900, fov_axis="h")
    assert (r["fov"], r["heading"], r["pitch"]) == (75.0, 269.0, 2.0)
    r = sv.shot_record(url, "A269", 1600, 900, fov_axis="v")
    assert r["fov"] == pytest.approx(2 * math.degrees(math.atan(math.tan(math.radians(37.5)) * 16 / 9)), abs=1e-3)
    assert sv.parse_url("@51.9,4.4,3a,75y,90h")["tilt"] == 90.0


def test_intersect_and_solve_three_rays():
    target = np.array([10.0, 20.0])
    cams = [np.array([0.0, 0.0]), np.array([25.0, 5.0]), np.array([-5.0, 30.0])]
    bs = [sv.bearing_of(target - c) for c in cams]
    x, t1, t2 = sv.intersect(cams[0], bs[0], cams[1], bs[1])
    assert np.allclose(x, target) and t1 > 0 and t2 > 0
    x, res = sv.solve_point(cams, [b + d for b, d in zip(bs, (0.3, -0.3, 0.2))])      # a little noise
    assert np.hypot(*(x - target)) < 0.3 and res < 0.3
    assert sv.intersect(cams[0], 45.0, cams[1], 45.5) is None                          # near parallel


def synthetic(points, cams, heading_jitter=0.0):
    """Shots at cams (local e, n) looking at each point in turn's general direction, and a detection per point in
    view (bbox round its pixel)."""
    to_local, to_wgs = sv.frame(LAT0, LON0)
    shots, dets = [], []
    for ci, c in enumerate(cams):
        c = np.asarray(c, float)
        mid = np.mean([p[:2] for p in points], 0)
        heading = sv.bearing_of(mid - c)
        lat, lon = to_wgs(*c)
        sid = f"S{ci}"
        shots.append({"id": sid, "lat": lat, "lon": lon, "heading": heading + heading_jitter, "pitch": 0.0,
                      "fov": FOV, "w": W, "h": H})
        for cls, (e, n, z) in [(p[3], p[:3]) for p in points]:
            d = np.array([e, n]) - c
            r = float(np.hypot(*d))
            uv = sv.world_to_pixel(sv.bearing_of(d), math.degrees(math.atan2(z - sv.CAM_H, r)), W, H, heading, 0.0,
                                   FOV)
            if uv is None or not (0 < uv[0] < W and 0 < uv[1] < H):
                continue
            u, v = uv
            b, el = sv.pixel_ray(u, v, W, H, heading + heading_jitter, 0.0, FOV)
            dets.append({"shot": sid, "cls": cls, "label": cls, "conf": 0.9, "bbox": [u - 5, v - 10, u + 5, v + 10],
                         "bearing": b, "elev": el, "face": True})
    return shots, dets


def test_triangulate_synthetic_place():
    points = [(10.0, 20.0, 4.0, "light"), (14.0, 22.0, 2.2, "sign"), (-6.0, 25.0, 3.5, "light")]
    cams = [(0.0, 0.0), (25.0, 4.0), (-12.0, 8.0), (5.0, -8.0)]
    shots, dets = synthetic(points, cams)
    objs, (to_local, to_wgs, rays) = sv.objects_of(shots, dets)
    assert len(objs) == 3
    for e, n, z, cls in points:
        o = min(objs, key=lambda o: np.hypot(*(o["p"] - to_local(*_wgs_of(e, n)))))
        assert o["cls"] == cls
        assert np.hypot(*(o["p"] - to_local(*_wgs_of(e, n)))) < 0.2
        assert o["n_views"] >= 3
        assert o["height_m"] == pytest.approx(z, abs=0.2)
        assert o["facing"] is not None


def _wgs_of(e, n):
    return sv.frame(LAT0, LON0)[1](e, n)


def test_match_verdicts():
    points = [(10.0, 20.0, 4.0, "light"), (14.0, 22.0, 2.2, "sign"), (-6.0, 25.0, 3.5, "light")]
    cams = [(0.0, 0.0), (25.0, 4.0), (-12.0, 8.0), (5.0, -8.0)]
    shots, dets = synthetic(points, cams)
    objs, (to_local, to_wgs, rays) = sv.objects_of(shots, dets)
    L = lambda e, n: to_local(*_wgs_of(e, n))      # noqa: E731
    mine = [
        # the light at (10, 20): ours 1 m off, facing the cameras (south-west): ok
        {"source": "placement", "ref": "items[0]", "cls": "light", "p": L(10.5, 20.8), "facing": 200.0,
         "codes": [], "name": "light A"},
        # the sign: ours facing away from every camera (north-east): wrong_facing
        {"source": "placement", "ref": "items[1]", "cls": "sign", "p": L(14.3, 22.0), "facing": 30.0,
         "codes": ["B6"], "name": "sign B6"},
        # ours where nothing stands, in view: extra_in_ours
        {"source": "placement", "ref": "items[2]", "cls": "light", "p": L(2.0, 14.0), "facing": 180.0,
         "codes": [], "name": "light C"},
        # the light at (-6, 25) is only in OSM: missing_in_ours, its match the OSM node
        {"source": "osm", "ref": "node/42", "cls": "light", "p": L(-5.0, 25.5), "facing": None, "codes": [],
         "name": "OSM traffic_signals"},
    ]
    apps = [{"name": "from the south", "mid": L(10.0, 15.0), "travel": 0.0}]
    out = sv.match(objs, mine, apps, shots, rays, to_local)
    by_ref = {o["match"]["ref"]: o for o in out if o.get("match") and o["match"]["ref"]}
    assert by_ref["items[0]"]["verdict"] == "ok" and "approach: from the south" in by_ref["items[0]"]["note"]
    assert by_ref["items[1]"]["verdict"] == "wrong_facing"
    assert by_ref["items[2]"]["verdict"] == "extra_in_ours" and by_ref["items[2]"]["n_views"] == 0
    assert by_ref["node/42"]["verdict"] == "missing_in_ours" and by_ref["node/42"]["match"]["source"] == "osm"
    feats = [sv.feature(o, to_wgs) for o in out]
    for f in feats:
        assert set(f["properties"]) >= {"id", "cls", "n_views", "residual_m", "shots", "match", "verdict", "note"}


def test_moved_and_wrong_code():
    points = [(14.0, 22.0, 2.2, "sign")]
    cams = [(0.0, 0.0), (25.0, 4.0), (-12.0, 8.0)]
    shots, dets = synthetic(points, cams)
    for d in dets:
        d["sign_code"], d["sign_score"] = "C2", 0.8
    objs, (to_local, to_wgs, rays) = sv.objects_of(shots, dets)
    assert objs[0]["sign_code"] == "C2"
    L = lambda e, n: to_local(*_wgs_of(e, n))      # noqa: E731
    mine = [{"source": "placement", "ref": "items[5]", "cls": "sign", "p": L(14.0, 23.0), "facing": 200.0,
             "codes": ["B6"], "name": "sign B6"}]
    assert sv.match(objs, mine, [], shots, rays, to_local)[0]["verdict"] == "wrong_code"
    mine[0]["p"] = L(20.0, 22.0)
    objs, _ = sv.objects_of(shots, dets)
    assert sv.match(objs, mine, [], shots, rays, to_local)[0]["verdict"] == "moved"


def test_jittered_heading_still_one_object():
    shots, dets = synthetic([(10.0, 20.0, 4.0, "light")], [(0.0, 0.0), (25.0, 4.0), (-12.0, 8.0)],
                            heading_jitter=1.0)
    objs, _ = sv.objects_of(shots, dets)
    assert len(objs) == 1 and objs[0]["residual_m"] < sv.MAX_RESIDUAL


def test_sign_atlas_reads_its_own_cells():
    cv2 = pytest.importorskip("cv2")
    if not os.path.exists(sv.ATLAS):
        pytest.skip("no sign atlas")
    import sign_atlas as sa
    atlas = sv.SignAtlas()
    img = cv2.imread(sv.ATLAS, cv2.IMREAD_COLOR)
    for code in ("B6", "C2", "D2", "A1"):
        if code not in sa.CELLS:
            continue
        x, y = sa.cell(None, sa.CELLS[code])
        crop = cv2.resize(img[y:y + sa.C, x:x + sa.C], (60, 60), interpolation=cv2.INTER_AREA)
        got, score = atlas.classify(crop)
        assert got == code, (code, got, score)
    grey = np.full((50, 30, 3), 128, np.uint8)
    assert atlas.classify(grey)[0] == "back"
