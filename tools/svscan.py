#!/usr/bin/env python3
"""Street furniture read off Street View screenshots, set against what we ship: a diagnosis, never a source
(fork-only tool; never upstream). Claude takes the screenshots by hand in a browser (Google Maps' Street View,
looking round a place such as the Droogleever Fortuynplein); this tool only reads the PNG files already on disk:
no scraping, no Google API. YOLO finds the traffic lights, signs, lamp posts, poles and bollards in each shot,
each detection becomes a ray from the known camera, the rays of one object from several panoramas are
triangulated (in plan, and in height: every view must put it at the same height), and each object found is matched
to OUR objects (what the kunstwerk's mesh ships: its signs, heads and poles, each tied back to its placement item or
NDW sign) and to the open data (NDW's sign register, OSM's signal nodes, the BGT's lamp posts). The output says
"this head belongs to that approach / this sign faces the wrong way / this sign is missing / ours is extra"; the
positions read off Google never go into the mod (project decision B1): every correction is a reference to an
open-data object (an NDW id, an OSM node, a BGT id, a placement index) plus a suggested facing or approach.

    python3 tools/svscan.py shot-meta --place droogleever --id A269 --png shot.png --url "<Street View URL>"
    python3 tools/svscan.py detect --place droogleever [--model world|coco|both] [--conf 0.12] [--no-mask]
    python3 tools/svscan.py triangulate --place droogleever [--kunstwerk droogleever]
    python3 tools/svscan.py match --place droogleever --kunstwerk droogleever
    python3 tools/svscan.py annotate --place droogleever
    python3 tools/svscan.py all --place droogleever --kunstwerk droogleever
    (--root <sv dir> and --design <design dir> when not run from E:\\BeamNG\\src; or $BNE_SV_ROOT, $BNE_DESIGN)

Screenshot protocol (Google Maps, Street View):
- Take the shot of the Street View viewport only (no side panel, no browser chrome), at browser zoom 100 %, and save
  it as <root>/<place>/shots/<id>.png. The picture's centre must be the view's centre: crop symmetrically if at all.
  The overlays Maps draws on the picture (the place card and search box top left, the buttons top right and bottom
  right, the minimap bottom left, the logo bottom centre, the copyright line) are masked before detection (UI_MASK,
  in pixels from the edges: set for an 800 x 500 viewport, the boxes keep their size in larger ones).
- Copy the page's URL at that moment. Street View's URL reads
  .../@51.9080,4.4660,3a,60y,269h,92t/data=!3m6!1e1!3m4!1s<pano id>!2e0...
  51.9080,4.4660 the camera (the panorama's place), 3a the Street View mode, 60y the VERTICAL field of view (deg),
  269h the heading (compass deg of the picture's centre), 92t the tilt (90 = the horizon; pitch = tilt - 90),
  !1s<pano> the panorama's id. Google Earth's URLs (@lat,lon,<alt>a,<dist>d,<fov>y,<heading>h,<tilt>t) read alike.
- `shot-meta` reads all of it from the URL and appends (or replaces) the shot's record in shots.json, with the
  PNG's size; shots.json's fov is the horizontal one, 2 atan(tan(y/2) w/h). Measured (2026-10-22): one signal head
  at x 755 looking 226 deg and at x 228 looking 288 deg in 800 x 500 shots at 60y: a focal length of ~425 px, so
  60y is vertical (86 deg across); --fov-axis h takes y as horizontal instead.
- Use several panoramas round the place (each object wants two panoramas or more, 10 m or more apart, ideally at
  30 deg or more to each other) and a few headings per panorama; look at the signs' faces where possible.
  Shots inside a trench or under a deck triangulate with the street's: each camera stands 2.5 m over the
  kunstwerk's floor at its spot (its mesh's raster, else the AHN's ground), heights are NAP.

Data (shared with the dashboard), in <root>/<place>/ (root E:\\BeamNG\\design\\reference\\sv):
- shots/<id>.png; shots.json [{"id","lat","lon","heading","pitch","fov","w","h","pano"?,"taken"?}]
- detections.json [{"shot","cls","label","conf","bbox":[x0,y0,x1,y1],"bearing","elev","sign_code"?,"sign_score"?,
  "face"?}] (bearing, elev: deg of the box's centre; cls light, sign, lamp, pole or bollard, after the shape checks;
  label the prompt; sign_code an RVV code, "back" or "unknown"; face true where a sign's face or a head's lit lens
  is seen, false for a sign's back)
- objects.geojson: Points (WGS84) {"id","cls","sign_code"?,"n_views","residual_m","height_m"?,"facing"?,"shots",
  "match":{"source","ref","dist_m"},"verdict","note"}; n_views counts panoramas; residual_m is the rays' rms miss in
  plan (3 views or more) or the two views' disagreement in height (2 views); source ndw | osm | bgt | placement | null
  (placement refs: items[i] of placement.json, placement_fix[i], direction_signs[i], or kunstwerk:<what>@x,y for a
  shipped object no item explains); verdict ok | moved | missing_in_ours | extra_in_ours | wrong_facing |
  wrong_code (null after triangulate alone); our shipped objects no detection supports but some shot looked at are
  features too (extra_in_ours).
- approaches.json: per signalled approach (ref/approaches.json) the heads we ship for it, the heads seen for it,
  and whether their facings agree. report.md: a short Dutch summary. shots_annotated/: the shots with their boxes.

Accuracy, honestly: Street View's camera positions are GPS (often 1-3 m off) and the heading of a Maps URL is
rounded to a degree, so the triangulated points are good to 1-3 m at 20 m; OK_M and MOVED_M are set for that.
Two views always meet in plan: a 2-view object is kept only where the rays cross at MIN_CROSS2_DEG or more and both
put it at the same height (HEIGHT_TOL). Lights and lamps are told apart by height (LAMP_MIN_H) after triangulating
them together. The sign classifier (colour classes + grey correlation against tools/sign_atlas.py's cells) is
reliable only on crops of 40 px or more and on the common round/triangular signs; small or oblique crops come back
"unknown".

Weights: ultralytics' YOLO-World (yolov8s-worldv2.pt, open vocabulary; needs the clip package:
pip install git+https://github.com/ultralytics/CLIP.git) and COCO yolo11m.pt as the fallback, fetched from
ultralytics' GitHub releases into ~/.cache/beamng-earth/svscan/ (outside the repo).
"""
import argparse
import datetime
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SV_ROOT = os.environ.get("BNE_SV_ROOT") or os.path.join(ROOT, "..", "design", "reference", "sv")
DESIGN = os.environ.get("BNE_DESIGN") or os.path.join(ROOT, "..", "design")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth", "svscan")
ATLAS = os.path.join(ROOT, "mod", "beamng_earth", "levels", "beamng_earth", "art", "shapes", "beamngEarth",
                     "bne_signs3.png")

M_PER_DEG = 111320.0
CAM_H = 2.5                 # m: the Street View car's camera over the ground
MAX_RANGE = 70.0            # m: a detection further than this is too small to place
MIN_RANGE = 1.5
MIN_CROSS_DEG = 4.0         # two rays nearer parallel than this do not fix a point
MIN_CROSS2_DEG = 10.0       # ... and an object seen from two panoramas only wants this much
TOL_M = 1.0                 # m: a ray supports a point passing this near it ...
TOL_DEG = 2.0               # ... or within this angle of it, whichever is the more
TAN_TOL = math.tan(math.radians(TOL_DEG))
MAX_RESIDUAL = 2.5          # m: rms miss of an object's rays
HEIGHT_TOL = 1.0            # m: the views of one object agree on its height to this
MERGE_M = 2.0               # m: objects of one class this near are one
MAX_FROM_ORIGIN = 120.0     # m: objects further from the kunstwerk's origin are left out
OK_M = 3.0                  # m: ours this near the object found is where it is
MOVED_M = 10.0              # m: ours this near is the same object, moved
LOOK_M = {"light": 30.0, "sign": 20.0, "pole": 20.0}   # m: a shot looked at our object nearer than this, in view
LOOK_MIN = 2                # panoramas that looked at our object before its missing detection counts (extra_in_ours)
MIN_H, MAX_H = -0.5, 14.0   # m: an object's height over its ground outside this is no object
FACE_DOT = 0.2              # a camera this far (cos) in front of / behind our facing sees its face / back
LAMP_MIN_H = 7.0            # m: a light/lamp detection's object this high (over its ground) is a lamp's luminaire
TIE_M = 3.0                 # m: a shipped object this near a placement item comes from it
APPROACH_M = 35.0           # m: a head serves the approach whose stop line is this near, facing its drivers

# YOLO-World's prompts and the class each one counts as
PROMPTS = {"traffic light": "light", "traffic signal head": "light", "street light luminaire": "lamp",
           "lamp post": "lamp", "traffic sign": "sign", "black and white striped pole": "pole", "bollard": "bollard"}
COCO = {"traffic light": "light", "stop sign": "sign"}
# classes triangulated together (told apart afterwards by height)
GROUP = {"light": "overhead", "lamp": "overhead", "sign": "sign", "pole": "pole", "bollard": "bollard"}
WORLD_W = "yolov8s-worldv2.pt"
COCO_W = "yolo11m.pt"
CODE_RE = re.compile(r"^[A-L]\d+[a-z]?$")
SIGN_MIN_SCORE = 0.55       # a sign classification under this is "unknown"
# Google Maps' overlays on an 800 x 500 Street View (px): the place card and the search box top left, the minimap
# bottom left, the buttons top right and bottom right, the logo bottom centre, the copyright line; in a larger
# viewport each keeps its size and its corner (ui_boxes)
UI_MASK = [(0, 0, 230, 35), (0, 0, 200, 110), (0, 430, 125, 500), (700, 0, 800, 35), (720, 400, 800, 500),
           (350, 472, 450, 500), (510, 486, 800, 500)]
UI_SIZE = (800, 500)


# ---------------------------------------------------------------- geometry

def frame(lat0, lon0):
    """(to_local, to_wgs): an east/north metre frame round (lat0, lon0) (equirectangular: cm over 300 m)."""
    k = M_PER_DEG * math.cos(math.radians(lat0))

    def to_local(lat, lon):
        return np.array([(lon - lon0) * k, (lat - lat0) * M_PER_DEG])

    def to_wgs(e, n):
        return lat0 + n / M_PER_DEG, lon0 + e / k
    return to_local, to_wgs


def wrap(a):
    """An angle (deg) into [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def unit_of(bearing):
    b = math.radians(bearing)
    return np.array([math.sin(b), math.cos(b)])


def bearing_of(v):
    return math.degrees(math.atan2(v[0], v[1])) % 360.0


def _basis(heading, pitch):
    hd, p = math.radians(heading), math.radians(pitch)
    F = np.array([math.sin(hd) * math.cos(p), math.cos(hd) * math.cos(p), math.sin(p)])    # east, north, up
    R = np.array([math.cos(hd), -math.sin(hd), 0.0])
    return F, R, np.cross(R, F)


def pixel_ray(u, v, w, h, heading, pitch, fov):
    """(bearing, elev) in deg of pixel (u, v) of a w x h rectilinear picture looking along heading (compass deg),
    pitch (deg, 0 = level), with a horizontal field of view fov (deg)."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    F, R, U = _basis(heading, pitch)
    d = (u - w / 2.0) * R - (v - h / 2.0) * U + f * F
    return math.degrees(math.atan2(d[0], d[1])) % 360.0, math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def world_to_pixel(bearing, elev, w, h, heading, pitch, fov):
    """pixel_ray inverted: (u, v), or None behind the camera."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    b, e = math.radians(bearing), math.radians(elev)
    d = np.array([math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e)])
    F, R, U = _basis(heading, pitch)
    z = d @ F
    if z <= 1e-6:
        return None
    return w / 2.0 + f * (d @ R) / z, h / 2.0 - f * (d @ U) / z


def intersect(p1, b1, p2, b2):
    """The point where the rays from p1 along compass bearing b1 and from p2 along b2 cross, with the ranges along
    each: (point, t1, t2), or None for rays near parallel."""
    u1, u2 = unit_of(b1), unit_of(b2)
    cr = u1[0] * u2[1] - u1[1] * u2[0]
    if abs(cr) < math.sin(math.radians(MIN_CROSS_DEG)):
        return None
    d = np.asarray(p2, float) - np.asarray(p1, float)
    t1 = (d[0] * u2[1] - d[1] * u2[0]) / cr
    t2 = (d[0] * u1[1] - d[1] * u1[0]) / cr
    return np.asarray(p1, float) + t1 * u1, t1, t2


def solve_point(ps, bs):
    """The point nearest (least squares, perpendicular) to the rays from ps along bearings bs, and the rays' rms
    miss (m)."""
    A = np.zeros((2, 2))
    rhs = np.zeros(2)
    for p, b in zip(ps, bs):
        u = unit_of(b)
        P = np.eye(2) - np.outer(u, u)
        A += P
        rhs += P @ np.asarray(p, float)
    x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    miss = [perp(x, p, b) for p, b in zip(ps, bs)]
    return x, float(math.sqrt(sum(m * m for m in miss) / len(miss)))


def perp(x, p, b):
    """The distance from x to the ray from p along bearing b (to p itself behind it)."""
    u = unit_of(b)
    d = np.asarray(x, float) - np.asarray(p, float)
    t = float(d @ u)
    if t < 0:
        return float(np.hypot(*d))
    return abs(float(d[0] * u[1] - d[1] * u[0]))


def tol(r):
    return max(TOL_M, r * TAN_TOL)


def ray_z(ray, x):
    """The height (the camera's datum) ray puts a point at x (plan) at: its camera's plus range x tan(elev)."""
    return ray.get("cz", CAM_H) + float(np.hypot(*(np.asarray(x) - ray["p"]))) * math.tan(math.radians(ray["elev"]))


def crossing(rs):
    """The widest angle (deg, up to 90) between the rays rs."""
    best = 0.0
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            a = abs(wrap(rs[i]["bearing"] - rs[j]["bearing"]))
            best = max(best, min(a, 180.0 - a))
    return best


def accept(rs, x):
    """(ok, residual) of rays rs (one per panorama) meeting at x: 3 views or more within MAX_RESIDUAL in plan; two
    crossing at MIN_CROSS2_DEG or more and agreeing on the height to HEIGHT_TOL (residual: that disagreement)."""
    if len(rs) < 2:
        return False, None
    if len(rs) == 2:
        dz = abs(ray_z(rs[0], x) - ray_z(rs[1], x))
        return crossing(rs) >= MIN_CROSS2_DEG and dz <= HEIGHT_TOL, dz
    _, res = solve_point([r["p"] for r in rs], [r["bearing"] for r in rs])
    return res <= MAX_RESIDUAL, res


def triangulate(rays):
    """Objects from rays [{"p": (e, n), "bearing", "elev", "cz", "shot", "station", "cls", ...}] (station: the
    panorama; its shots count once): per class group, greedy, the point most panoramas agree on first, in plan
    and in height. Returns [{"p", "group", "rays": [indices], "residual", "z"}] (z: the median height, the
    cameras' datum)."""
    out = []
    by_grp = {}
    for i, r in enumerate(rays):
        by_grp.setdefault(GROUP.get(r["cls"], r["cls"]), []).append(i)
    for grp, idx in by_grp.items():
        R = [rays[i] for i in idx]
        P = np.array([r["p"] for r in R], float).reshape(-1, 2)
        B = np.radians([r["bearing"] for r in R])
        U = np.stack([np.sin(B), np.cos(B)], 1) if len(R) else np.zeros((0, 2))
        st = [r.get("station", r["shot"]) for r in R]
        cands = []
        for a in range(len(R)):
            for b in range(a + 1, len(R)):
                if st[a] == st[b] or np.hypot(*(P[a] - P[b])) < 1.0:
                    continue
                hit = intersect(P[a], R[a]["bearing"], P[b], R[b]["bearing"])
                if hit is None:
                    continue
                x, t1, t2 = hit
                if MIN_RANGE < t1 < MAX_RANGE and MIN_RANGE < t2 < MAX_RANGE \
                        and abs(ray_z(R[a], x) - ray_z(R[b], x)) <= 2 * HEIGHT_TOL:
                    cands.append((x, a, b))
        free = np.ones(len(R), bool)

        def support(x):
            """Per panorama the free ray nearest x that supports it, those agreeing on its height: [index]."""
            d = x[None, :] - P
            t = (d * U).sum(1)
            r = np.hypot(d[:, 0], d[:, 1])
            m = np.abs(d[:, 0] * U[:, 1] - d[:, 1] * U[:, 0])
            ok = free & (t > MIN_RANGE) & (r < MAX_RANGE) & (m <= np.maximum(TOL_M, r * TAN_TOL))
            best = {}
            for k in np.flatnonzero(ok):
                if st[k] not in best or m[k] < m[best[st[k]]]:
                    best[st[k]] = k
            ks = list(best.values())
            if len(ks) >= 2:
                zs = np.array([ray_z(R[k], x) for k in ks])
                med = float(np.median(zs))
                ks = [k for k, z in zip(ks, zs) if abs(z - med) <= HEIGHT_TOL]
            return ks
        while cands:
            best = None
            for x, a, b in cands:
                if not (free[a] and free[b]):
                    continue
                ks = support(x)
                if len(ks) < 2:
                    continue
                y, res = solve_point([P[k] for k in ks], [R[k]["bearing"] for k in ks])
                key = (len(ks), -res)
                if best is None or key > best[0]:
                    best = (key, ks, a, b)
            if best is None:
                break
            _, ks, a0, b0 = best
            for _ in range(3):                      # refine: re-solve, re-gather
                y, _r = solve_point([P[k] for k in ks], [R[k]["bearing"] for k in ks])
                ks2 = support(y)
                if len(ks2) < 2 or set(ks2) == set(ks):
                    break
                ks = ks2
            y, _r = solve_point([P[k] for k in ks], [R[k]["bearing"] for k in ks])
            ok, res = accept([R[k] for k in ks], y)
            if not ok:                              # (this seed pair leads nowhere: drop it, keep its rays)
                cands = [c for c in cands if not (c[1] == a0 and c[2] == b0)]
                continue
            for k in ks:
                free[k] = False
            out.append({"p": y, "group": grp, "rays": [idx[k] for k in ks], "residual": res,
                        "z": float(np.median([ray_z(R[k], y) for k in ks]))})
            cands = [c for c in cands if free[c[1]] and free[c[2]]]
    return merge(out, rays)


def merge(objs, rays):
    """Objects of one group within MERGE_M as one: their rays (one per panorama, the nearest) re-solved."""
    objs = list(objs)
    changed = True
    while changed:
        changed = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                if a["group"] != b["group"] or np.hypot(*(a["p"] - b["p"])) > MERGE_M:
                    continue
                mid = (a["p"] + b["p"]) / 2
                per = {}
                for k in a["rays"] + b["rays"]:
                    s = rays[k].get("station", rays[k]["shot"])
                    if s not in per or perp(mid, rays[k]["p"], rays[k]["bearing"]) < \
                            perp(mid, rays[per[s]]["p"], rays[per[s]]["bearing"]):
                        per[s] = k
                ks = list(per.values())
                y, _ = solve_point([rays[k]["p"] for k in ks], [rays[k]["bearing"] for k in ks])
                _, res = accept([rays[k] for k in ks], y)
                objs[i] = {"p": y, "group": a["group"], "rays": ks, "residual": res if res is not None else 0.0,
                           "z": float(np.median([ray_z(rays[k], y) for k in ks]))}
                objs.pop(j)
                changed = True
                break
            if changed:
                break
    return objs


def facing_of(obj_p, views):
    """The compass facing of an object from its views [(camera p, face: True/False/None)]: toward the mean of the
    cameras that saw its face, else away from those that saw its back; None without either."""
    fronts = [np.asarray(p, float) - obj_p for p, f in views if f is True]
    backs = [np.asarray(p, float) - obj_p for p, f in views if f is False]
    v = np.zeros(2)
    for d in fronts:
        v += d / max(np.hypot(*d), 1e-6)
    for d in backs:
        v -= d / max(np.hypot(*d), 1e-6)
    if not fronts and not backs or np.hypot(*v) < 1e-6:
        return None
    return bearing_of(v)


# ---------------------------------------------------------------- shots, URLs

def parse_url(url):
    """{"lat", "lon", "fov_y", "heading", "tilt", "pano"} of a Google Maps / Earth Street View URL
    (.../@lat,lon,3a,75y,269h,92t/data=...!1s<pano>!...). Missing heading 0, tilt 90, fov 75."""
    m = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)((?:,[-\d.]+[a-z])*)", url)
    if not m:
        raise ValueError(f"no @lat,lon in {url!r}")
    out = {"lat": float(m.group(1)), "lon": float(m.group(2)), "fov_y": 75.0, "heading": 0.0, "tilt": 90.0}
    for tok in m.group(3).split(",")[1:]:
        v, k = float(tok[:-1]), tok[-1]
        if k == "y":
            out["fov_y"] = v
        elif k == "h":
            out["heading"] = v % 360.0
        elif k == "t":
            out["tilt"] = v
    p = re.search(r"!1s([A-Za-z0-9_\-]{10,})!", url)
    if p:
        out["pano"] = p.group(1)
    return out


def hfov(fov_y, w, h):
    """The horizontal field of view (deg) of a w x h picture whose vertical one is fov_y."""
    return 2 * math.degrees(math.atan(math.tan(math.radians(fov_y) / 2) * w / h))


def shot_record(url, sid, w, h, fov_axis="v"):
    """The shots.json record of a screenshot w x h taken at url."""
    u = parse_url(url)
    fov = hfov(u["fov_y"], w, h) if fov_axis == "v" else u["fov_y"]
    rec = {"id": sid, "lat": u["lat"], "lon": u["lon"], "heading": round(u["heading"], 2),
           "pitch": round(u["tilt"] - 90.0, 2), "fov": round(fov, 3), "w": int(w), "h": int(h)}
    if "pano" in u:
        rec["pano"] = u["pano"]
    rec["taken"] = datetime.datetime.now().isoformat(timespec="seconds")
    return rec


def station_of(s):
    """A shot's panorama: its pano id, else its spot to ~1 m."""
    return s.get("pano") or f"{s['lat']:.5f},{s['lon']:.5f}"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path, obj):
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def ui_boxes(w, h):
    """UI_MASK's boxes (x0, y0, x1, y1) in a w x h shot: each moved with its corner (the centre's with the centre)."""
    W, H = UI_SIZE
    out = []
    for x0, y0, x1, y1 in UI_MASK:
        dx = 0 if x1 <= W / 2 else (w - W if x0 >= W / 2 else (w - W) / 2)
        dy = 0 if y1 <= H / 2 else h - H
        out.append((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
    return out


def in_ui(bbox, boxes, frac=0.3):
    """The detection's box lies frac or more in the UI."""
    x0, y0, x1, y1 = bbox
    area = max((x1 - x0) * (y1 - y0), 1e-6)
    cov = sum(max(0.0, min(x1, b[2]) - max(x0, b[0])) * max(0.0, min(y1, b[3]) - max(y0, b[1])) for b in boxes)
    return cov / area >= frac


# ---------------------------------------------------------------- signs, shapes

class SignAtlas:
    """The RVV signs of tools/sign_atlas.py's atlas (bne_signs3.png, its CELLS), each trimmed to its plate, for
    matching crops: colour classes (red, blue, white, yellow, black) agree, and grey correlation."""
    SIZE = 40

    def __init__(self, path=ATLAS):
        import cv2
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import sign_atlas as sa
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        self.t = []
        for code, k in sa.CELLS.items():
            if not CODE_RE.match(code):
                continue
            x, y = sa.cell(None, k)
            c = img[y:y + sa.C, x:x + sa.C]
            bg = np.median(np.concatenate([c[:4].reshape(-1, 3), c[-4:].reshape(-1, 3)]), 0)
            mask = np.abs(c.astype(int) - bg).sum(2) > 40
            ys, xs = np.nonzero(mask)
            if len(xs) < 100:
                continue
            c = c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            hh, ww = c.shape[:2]
            s = self.SIZE / max(hh, ww)
            size = (max(8, round(ww * s)), max(8, round(hh * s)))
            c = cv2.resize(c, size, interpolation=cv2.INTER_AREA)
            m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            self.t.append((code, c, m, colour_classes(c)))

    def classify(self, crop):
        """(code, score) of a BGR crop: an RVV code, "back" (a sign's grey back) or "unknown"."""
        import cv2
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 10:
            return "unknown", 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        core = hsv[crop.shape[0] // 5: -crop.shape[0] // 5 or None, crop.shape[1] // 5: -crop.shape[1] // 5 or None]
        if core.size and np.median(core[..., 1]) < 35 and 50 < np.median(core[..., 2]) < 190 \
                and np.percentile(core[..., 1], 90) < 60:
            return "back", 0.6
        best = ("unknown", 0.0)
        H, W = crop.shape[:2]
        for code, t, m, tc in self.t:
            th, tw = t.shape[:2]
            for shrink in (0.0, 0.08):              # YOLO's boxes are often a little loose
                dy, dx = int(H * shrink), int(W * shrink)
                c = crop[dy:H - dy or None, dx:W - dx or None]
                c = cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)
                cc = colour_classes(c)
                agree = float((cc[m] == tc[m]).mean())
                g1 = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)[m].astype(float)
                g2 = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)[m].astype(float)
                g1, g2 = g1 - g1.mean(), g2 - g2.mean()
                ncc = float((g1 * g2).sum() / (math.sqrt((g1 * g1).sum() * (g2 * g2).sum()) + 1e-6))
                s = 0.6 * agree + 0.4 * max(ncc, 0.0)
                if s > best[1]:
                    best = (code, s)
        if best[1] < SIGN_MIN_SCORE:
            return "unknown", round(best[1], 3)
        return best[0], round(best[1], 3)


def colour_classes(bgr):
    """Per pixel 0 other/grey, 1 red, 2 blue, 3 white, 4 yellow, 5 black."""
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(h.shape, np.uint8)
    sat = (s > 90) & (v > 60)
    out[sat & ((h < 10) | (h > 165))] = 1
    out[sat & (h > 95) & (h < 130)] = 2
    out[sat & (h >= 15) & (h < 35)] = 4
    out[(s < 50) & (v > 170)] = 3
    out[v < 55] = 5
    return out


def lens_lit(bgr):
    """A signal head's lit lens in the crop (its face toward the camera): bright saturated red, amber or green."""
    import cv2
    if bgr is None or bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 110) & (v > 170) & ((h < 30) | (h > 165) | ((h > 45) & (h < 95)))
    return True if lit.mean() > 0.01 else None


def head_like(bgr):
    """A signal head's shape: a box 1.4 to 4.5 times as tall as wide, dark (its black body), or with a lit lens."""
    import cv2
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 4:
        return False
    hh, ww = bgr.shape[:2]
    asp = hh / max(ww, 1)
    if not 1.4 <= asp <= 4.5:
        return False
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    core = v[:, ww // 4: ww - ww // 4 or None]
    return bool(np.median(core) < 90 or lens_lit(bgr))


def striped(bgr):
    """A black-and-white ringed pole: tall, its middle column's rows alternating dark and light 4 times or more."""
    import cv2
    if bgr is None or bgr.size == 0:
        return False
    hh, ww = bgr.shape[:2]
    if hh < 3 * ww or hh < 20:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[:, ww // 3: ww - ww // 3 or None].mean(1)
    lo, hi = np.percentile(g, 15), np.percentile(g, 85)
    if hi - lo < 60:
        return False
    s = np.where(g < lo + (hi - lo) * 0.35, 0, np.where(g > hi - (hi - lo) * 0.35, 1, -1))
    s = s[s >= 0]
    return int((np.diff(s) != 0).sum()) >= 4


def reclassify(cls, crop):
    """The class after the shape checks: a lamp box shaped like a head is a light, a light or a sign's back that is
    a ringed pole is a pole, a light box much taller than a head is a pole."""
    if cls == "lamp" and head_like(crop):
        return "light"
    if cls in ("sign", "light", "lamp") and striped(crop):
        return "pole"
    if cls == "light" and crop is not None and crop.size and crop.shape[0] > 6 * crop.shape[1]:
        return "pole"
    return cls


# ---------------------------------------------------------------- detection

def model(kind):
    from ultralytics import YOLO
    os.makedirs(CACHE, exist_ok=True)
    if kind == "world":
        m = YOLO(os.path.join(CACHE, WORLD_W))
        m.set_classes(list(PROMPTS))
        return m, PROMPTS
    return YOLO(os.path.join(CACHE, COCO_W)), COCO


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def detect(place_dir, kind="world", conf=0.12, imgsz=1280, atlas=None, mask=True):
    """detections.json of the place's shots."""
    import cv2
    shots = load(os.path.join(place_dir, "shots.json"), [])
    kinds = ["world", "coco"] if kind == "both" else [kind]
    models = []
    for k in kinds:
        try:
            models.append(model(k))
        except Exception as e:                       # (no clip package: COCO alone)
            print(f"{k} model unavailable ({e}); COCO instead", file=sys.stderr)
            if "coco" not in kinds:
                models.append(model("coco"))
    atlas = atlas or SignAtlas()
    dets = []
    for s in shots:
        path = os.path.join(place_dir, "shots", s["id"] + ".png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"missing {path}", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        if (w, h) != (s["w"], s["h"]):
            print(f"{s['id']}: {w}x{h} on disk, {s['w']}x{s['h']} in shots.json (the fov is the record's)",
                  file=sys.stderr)
        boxes = ui_boxes(w, h) if mask else []
        seen = img.copy()
        for b in boxes:                              # (Maps' overlays: grey, not objects)
            seen[int(b[1]):int(math.ceil(b[3])), int(b[0]):int(math.ceil(b[2]))] = 128
        found = []
        for m, names in models:
            r = m.predict(seen, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                label = m.names[int(k)]
                if label not in names or in_ui(b, boxes):
                    continue
                if any(iou(b, f["bbox"]) > 0.6 for f in found):
                    continue
                found.append({"cls": names[label], "label": label, "conf": round(float(c), 3),
                              "bbox": [round(float(v), 1) for v in b]})
        for f in found:
            x0, y0, x1, y1 = f["bbox"]
            crop = img[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
            f["cls"] = reclassify(f["cls"], crop)
            sx, sy = w / s["w"], h / s["h"]          # (the record's pixel grid)
            bearing, elev = pixel_ray((x0 + x1) / 2 / sx, (y0 + y1) / 2 / sy, s["w"], s["h"], s["heading"],
                                      s["pitch"], s["fov"])
            d = {"shot": s["id"], **f, "bearing": round(bearing, 3), "elev": round(elev, 3)}
            if f["cls"] == "sign":
                code, score = atlas.classify(crop)
                d["sign_code"], d["sign_score"] = code, score
                d["face"] = {"back": False, "unknown": None}.get(code, True)
            elif f["cls"] == "light":
                d["face"] = lens_lit(crop)
            dets.append(d)
        print(f"{s['id']}: {len(found)} objects")
    save(os.path.join(place_dir, "detections.json"), dets)
    return dets


# ---------------------------------------------------------------- the kunstwerk: its frame, ground, what it ships

class Kunstwerk:
    """A kunstwerk's frame (its origin), its ground (the mesh's floor raster, else the AHN's ground; NAP) and its
    local RD <-> WGS84."""

    def __init__(self, design, name):
        sys.path.insert(0, os.path.join(ROOT, "server"))
        from ahn import rd_to_wgs, wgs_to_rd
        self._rd_to_wgs, self._wgs_to_rd = rd_to_wgs, wgs_to_rd
        self.name = name
        self.ref = os.path.join(design, "kunstwerken", name, "ref")
        self.kw = load(os.path.join(ROOT, "server", "data", "kunstwerken", f"{name}.kunstwerk.json"))
        refj = load(os.path.join(self.ref, "ref.json"))
        src = refj or self.kw
        self.origin_rd = src["origin_rd"]
        self.origin_wgs = src["origin_wgs"]
        self.features = (refj or {}).get("features", [])
        self.floor = None
        if self.kw and self.kw.get("floor"):
            fl = self.kw["floor"]
            self.floor = np.asarray(fl["z"], float).reshape(fl["ny"], fl["nx"])
            self.floor[self.floor < -900] = np.nan
            self.fx0, self.fy0 = fl["x0"], fl["y0"]
        a = os.path.join(self.ref, "ahn.npz")
        self.ahn = np.load(a) if os.path.exists(a) else None

    def rd(self, lat, lon):
        x, y = self._wgs_to_rd(lat, lon)
        return float(x) - self.origin_rd[0], float(y) - self.origin_rd[1]

    def wgs(self, x, y):
        lat, lon = self._rd_to_wgs(self.origin_rd[0] + x, self.origin_rd[1] + y)
        return float(lat), float(lon)

    def ground(self, x, y):
        """The ground's height (NAP) at local RD (x, y): the mesh's floor, else the AHN's ground, else 0."""
        if self.floor is not None:
            i, j = int(math.floor(x - self.fx0)), int(math.floor(y - self.fy0))
            if 0 <= j < self.floor.shape[0] and 0 <= i < self.floor.shape[1] and np.isfinite(self.floor[j, i]):
                return float(self.floor[j, i])
        if self.ahn is not None:
            res, x0, y1 = float(self.ahn["res"]), float(self.ahn["x0"]), float(self.ahn["y1"])
            c, r = int((x - x0) / res), int((y1 - y) / res)
            d = self.ahn["dtm"]
            if 0 <= r < d.shape[0] and 0 <= c < d.shape[1] and np.isfinite(d[r, c]):
                return float(d[r, c])
        return 0.0


def _clusters(pts, r, dz=None):
    """Greedy clusters of points (n, 3) within r in plan (and dz in height): [indices]."""
    out = []
    cent = []
    for i, p in enumerate(pts):
        for k, c in enumerate(cent):
            if math.hypot(p[0] - c[0], p[1] - c[1]) < r and (dz is None or abs(p[2] - c[2]) < dz):
                out[k].append(i)
                cent[k] = pts[out[k]].mean(0)
                break
        else:
            out.append([i])
            cent.append(p.copy())
    return out


def placement_items(ref):
    """placement.json's items with placement_fix.json applied as tools/kunstwerk_solid.py applies it, each with
    its ref (items[i] of placement.json; placement_fix[i] for the ones a fix adds)."""
    items = [dict(it, _ref=f"items[{i}]") for i, it in
             enumerate((load(os.path.join(ref, "placement.json"), {}) or {}).get("items", []))]
    for k, fx in enumerate((load(os.path.join(ref, "placement_fix.json"), {}) or {}).get("fixes", [])):
        if fx["action"] == "add":
            items.append(dict({a: v for a, v in fx.items() if a not in ("action", "why")}, _ref=f"placement_fix[{k}]"))
            continue
        near = np.asarray(fx["near"], float)
        cand = [i for i, it in enumerate(items) if it["module"] == fx["module"]
                and np.hypot(*(np.asarray(it["at"]) - near)) < 2.0]
        if not cand:
            continue
        i = min(cand, key=lambda i: np.hypot(*(np.asarray(items[i]["at"]) - near)))
        if fx["action"] == "remove":
            items.pop(i)
        else:
            items[i].update({a: v for a, v in fx.items() if a not in ("action", "why", "near")})
    return items


def mast_heads(it):
    """The heads (x, y, height over the ground) of a placement mast with its arm, as tools/kunstwerk_props.py's
    mast_arm() and tools/kunstwerk_solid.py build it, and its low head on the mast."""
    lanes = int(it["mast"])
    reach = float(it.get("reach", min(lanes * 3.25 * 0.85, 9.0)))
    n = max(1, min(3, lanes))
    f = np.asarray(it["facing"], float)
    f = f / max(np.hypot(*f), 1e-9)
    side = np.array([f[1], -f[0]]) * (-1 if it.get("arm") == "right" else 1)   # (the module's +x)
    q = np.asarray(it["at"], float)
    out = [(*(q + side * reach * (i + 0.6) / n), 6.4 - 0.45 + 0.35 * (i + 0.6) / n - 0.1) for i in range(n)]
    out.append((*q, 3.3 - 0.51 - 0.05))
    return out


def shipped(kw, to_local):
    """What the kunstwerk ships (its mesh, server/data/kunstwerken/<name>.kunstwerk.json) in the place's frame:
    [{"source", "ref", "cls", "p", "z", "facing", "codes", "name"}]: its sign plates (their atlas cells), its
    heads (the switchable ones in "signals", the masts' lenses in the mesh), its ringed poles; each tied to the
    placement item, direction sign or NDW sign it comes from (ref), else kunstwerk:<what>@x,y. Also the open data:
    NDW's signs, OSM's signals, the BGT's lamp posts (lichtmast)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import sign_atlas as sa
    items = placement_items(kw.ref)
    dsigns = (load(os.path.join(kw.ref, "direction_signs.json"), {}) or {}).get("signs", [])
    ndw = load(os.path.join(kw.ref, "ndw", "signs.json"), []) or []
    ndw_rd = [(f, kw.rd(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])) for f in ndw]

    def loc(x, y):
        return to_local(*kw.wgs(x, y))

    def face(x, y, f):
        if f is None or not np.hypot(*f):
            return None
        return bearing_of(loc(x + f[0], y + f[1]) - loc(x, y))     # (RD's grid north is not true north)

    def tie(cls, q, codes=()):
        """(source, ref, facing) of what built the shipped object at q (local RD)."""
        if cls == "sign":
            for c in codes:
                near = [(np.hypot(q[0] - p[0], q[1] - p[1]), f) for f, p in ndw_rd
                        if f["properties"].get("rvvCode") == c]
                near = [n for n in near if n[0] < 4.0]
                if near:
                    f = min(near, key=lambda n: n[0])[1]
                    return "ndw", str(f.get("id")), None
            for k, it in enumerate(dsigns):
                if np.hypot(*(np.asarray(it["at"]) - q[:2])) < 2.5:
                    return "placement", f"direction_signs[{k}]", face(*it["at"], it.get("facing"))
        best = None
        for it in items:
            if it["module"] not in (("sign",) if cls == "sign" else ("traffic_light",)):
                continue
            pts = [np.asarray(it["at"], float)]
            if it.get("mast") and cls == "light":
                pts += [np.asarray(h[:2]) for h in mast_heads(it)]
            d = min(np.hypot(*(p - q[:2])) for p in pts)
            if d < TIE_M and (best is None or d < best[0]):
                best = (d, it)
        if best:
            it = best[1]
            return "placement", it["_ref"], face(*it["at"], it.get("facing"))
        return "placement", f"kunstwerk:{cls}@{q[0]:.1f},{q[1]:.1f}", None
    out = []
    if kw.kw:
        P = np.asarray(kw.kw["positions"], float)
        T = np.asarray(kw.kw["tris"], np.int64)
        UV = np.asarray(kw.kw["uvs"], float)
        tm = np.asarray(kw.kw["tri_material"], np.int64)
        mats = kw.kw["materials"]
        if "bne_signs" in mats:
            T = T[tm == mats.index("bne_signs")]
            uv = UV[T].mean(1)
            col = np.clip(np.floor(uv[:, 0] * sa.GRID[0]), 0, sa.GRID[0] - 1).astype(int)
            row = sa.GRID[1] - 1 - np.clip(np.floor(uv[:, 1] * sa.GRID[1]), 0, sa.GRID[1] - 1).astype(int)
            cell = row * sa.GRID[0] + col
            cen = P[T].mean(1)

            def mesh_face(q, behind, r):
                """The facing of a plate or a head at q: from the middle of its back (the cell behind: a plate's
                grey back, a head's black body) within r to q."""
                k = np.flatnonzero((cell == sa.CELLS[behind]) & (np.linalg.norm(cen - q, axis=1) < r))
                if not len(k):
                    return None
                v = q[:2] - cen[k, :2].mean(0)
                return face(q[0], q[1], v) if np.hypot(*v) > 1e-3 else None
            inv = {k: c for c, k in sa.CELLS.items() if CODE_RE.match(c)}
            sel = np.flatnonzero(np.isin(cell, list(inv)))
            for cl in _clusters(cen[sel], 0.5, 2.0):
                k = sel[cl]
                q = cen[k].mean(0)
                codes = sorted({inv[c] for c in cell[k]})
                src, ref, f = tie("sign", q, codes)
                fac = mesh_face(q, "back", 0.6)
                out.append({"source": src, "ref": ref, "cls": "sign", "p": loc(*q[:2]), "z": float(q[2]),
                            "facing": fac, "codes": codes, "name": f"sign {'+'.join(codes)}"})
            lens = [sa.CELLS[c] for c in ("lens_red", "lens_amber", "lens_green") if c in sa.CELLS]
            sel = np.flatnonzero(np.isin(cell, lens))
            for cl in _clusters(cen[sel], 0.45, 0.8):
                k = sel[cl]
                q = cen[k].mean(0)
                src, ref, f = tie("light", q)
                fac = mesh_face(q, "black", 0.35)
                out.append({"source": src, "ref": ref, "cls": "light", "p": loc(*q[:2]), "z": float(q[2]),
                            "facing": fac, "codes": [], "name": "head (mesh)"})
            if "stripes" in sa.CELLS:
                sel = np.flatnonzero(cell == sa.CELLS["stripes"])
                for cl in _clusters(cen[sel], 0.6):
                    q = cen[sel[cl]].mean(0)
                    src, ref, _f = tie("pole", q)
                    out.append({"source": src, "ref": ref, "cls": "pole", "p": loc(*q[:2]), "z": float(q[2]),
                                "facing": None, "codes": [], "name": "ringed pole (mesh)"})
        for s in kw.kw.get("signals", []):
            q = np.asarray(s["at"], float)
            if any(o["cls"] == "light" and np.hypot(*(o["p"] - loc(*q[:2]))) < 0.5 for o in out):
                continue
            src, ref, _f = tie("light", q)
            out.append({"source": src, "ref": ref, "cls": "light", "p": loc(*q[:2]), "z": float(q[2]),
                        "facing": face(q[0], q[1], s["facing"]), "codes": [], "name": "switchable head"})
    else:                                           # (no mesh: the placement as planned)
        for it in items:
            cls = {"traffic_light": "light", "sign": "sign"}.get(it["module"])
            if cls:
                code = it.get("code")
                codes = code if isinstance(code, list) else ([code] if code else [])
                out.append({"source": "placement", "ref": it["_ref"], "cls": cls, "p": loc(*it["at"]), "z": None,
                            "facing": face(*it["at"], it.get("facing")), "codes": codes, "name": it["module"]})
    for f, (x, y) in ndw_rd:
        pr = f.get("properties", {})
        b = pr.get("bearing")
        out.append({"source": "ndw", "ref": str(f.get("id")), "cls": "sign", "p": loc(x, y), "z": None,
                    "facing": None if b is None else (float(b) + 180.0) % 360.0, "open": True,
                    "codes": [pr["rvvCode"]] if pr.get("rvvCode") else [], "name": f"NDW {pr.get('rvvCode')}"})
    seen = set()
    for e in (load(os.path.join(kw.ref, "osm_signals.json"), {}) or {}).get("elements", []):
        if e.get("type") == "node" and e.get("tags", {}).get("highway") == "traffic_signals":
            seen.add(e["id"])
            out.append({"source": "osm", "ref": f"node/{e['id']}", "cls": "light", "p": to_local(e["lat"], e["lon"]),
                        "z": None, "facing": None, "codes": [], "name": "OSM traffic_signals", "open": True})
    xml = os.path.join(kw.ref, "osm_map.xml")
    if os.path.exists(xml):
        for n in ET.parse(xml).getroot().iter("node"):
            tags = {t.get("k"): t.get("v") for t in n.iter("tag")}
            cls = ("light" if tags.get("highway") == "traffic_signals" else "lamp" if tags.get("highway") ==
                   "street_lamp" else "bollard" if tags.get("barrier") == "bollard" else None)
            nid = int(n.get("id"))
            if cls is None or nid in seen:
                continue
            out.append({"source": "osm", "ref": f"node/{nid}", "cls": cls, "z": None, "open": True,
                        "p": to_local(float(n.get("lat")), float(n.get("lon"))), "facing": None, "codes": [],
                        "name": f"OSM {cls}"})
    for f in kw.features:
        if f.get("layer") == "paal" and f.get("functie") in ("lichtmast",):
            q = np.asarray(f["rings"], float).reshape(-1, 2).mean(0)
            out.append({"source": "bgt", "ref": f["id"], "cls": "lamp", "p": loc(*q), "z": None, "facing": None,
                        "codes": [], "name": "BGT lichtmast", "open": True})
    return out


def approaches(kw, to_local):
    """ref/approaches.json's signalled approaches: [{"name", "mid" (local), "travel" (compass deg)}]."""
    out = []
    for a in (load(os.path.join(kw.ref, "approaches.json"), {}) or {}).get("approaches", []):
        L, R = to_local(*kw.wgs(*a["left"])), to_local(*kw.wgs(*a["right"]))
        d = R - L
        out.append({"name": a["name"], "mid": (L + R) / 2, "travel": bearing_of(np.array([-d[1], d[0]]))})
    return out


def approach_of(p, facing, apps):
    """The approach a head at p facing (compass deg; None: any) speaks to: the nearest stop line within APPROACH_M
    whose drivers it faces, and whose drivers are in front of it."""
    best = None
    for a in apps:
        back = (a["travel"] + 180.0) % 360.0
        if facing is not None and abs(wrap(facing - back)) > 60:
            continue
        d = float(np.hypot(*(a["mid"] - p)))
        if d < APPROACH_M and (best is None or d < best[0]):
            best = (d, a["name"])
    return best[1] if best else None


# ---------------------------------------------------------------- objects

def rays_of(shots, dets, to_local, kw=None):
    by_id = {s["id"]: s for s in shots}
    rays = []
    for i, d in enumerate(dets):
        s = by_id.get(d["shot"])
        if s is None:
            continue
        cz = CAM_H + (kw.ground(*kw.rd(s["lat"], s["lon"])) if kw else 0.0)
        rays.append({**d, "det": i, "p": to_local(s["lat"], s["lon"]), "cz": cz, "station": station_of(s)})
    return rays


def majority_code(rs):
    votes = {}
    for r in rs:
        c = r.get("sign_code")
        if c and CODE_RE.match(c):
            votes[c] = votes.get(c, 0.0) + float(r.get("sign_score") or 0)
    return max(votes, key=votes.get) if votes else None


def objects_of(shots, dets, kw=None):
    """The triangulated objects: [{"id", "cls", "p", "n_views", "residual_m", "height_m", "facing", "shots",
    "sign_code", "rays", "z"}] in the place's local frame (round the kunstwerk's origin when given; objects
    further than MAX_FROM_ORIGIN from it are left out), and (to_local, to_wgs, rays)."""
    if kw is not None:
        to_local, to_wgs = frame(*kw.origin_wgs)
    else:
        to_local, to_wgs = frame(float(np.mean([s["lat"] for s in shots])), float(np.mean([s["lon"] for s in shots])))
    rays = rays_of(shots, dets, to_local, kw)
    objs = []
    counts = {}
    for o in triangulate(rays):
        if kw is not None and np.hypot(*o["p"]) > MAX_FROM_ORIGIN:
            continue
        rs = [rays[i] for i in o["rays"]]
        g = kw.ground(*kw.rd(*to_wgs(*o["p"]))) if kw else 0.0
        h = o["z"] - g
        if not MIN_H <= h <= MAX_H:                 # (under the ground or in the sky: rays that met by chance)
            continue
        note = ""
        if o["group"] == "overhead":                # (heads and lamps: told apart by height)
            cls = "lamp" if h >= LAMP_MIN_H else "light"
            labels = {r["cls"] for r in rs}
            if cls not in labels:
                note = f"{'/'.join(sorted(labels))} by YOLO, {cls} by its height"
        else:
            cls = o["group"]
        counts[cls] = counts.get(cls, 0) + 1
        objs.append({"id": f"sv-{cls}-{counts[cls]}", "cls": cls, "p": o["p"], "z": o["z"],
                     "n_views": len({r["station"] for r in rs}), "residual_m": round(o["residual"], 2),
                     "height_m": round(h, 2), "facing": facing_of(o["p"], [(r["p"], r.get("face")) for r in rs]),
                     "shots": sorted({r["shot"] for r in rs}), "sign_code": majority_code(rs) if cls == "sign"
                     else None, "rays": o["rays"], "note": note})
    return objs, (to_local, to_wgs, rays)


def feature(o, to_wgs):
    lat, lon = to_wgs(*o["p"])
    props = {"id": o["id"], "cls": o["cls"], "n_views": o["n_views"], "residual_m": o["residual_m"],
             "shots": o["shots"], "match": o.get("match") or {"source": None, "ref": None, "dist_m": None},
             "verdict": o.get("verdict"), "note": o.get("note", "")}
    if o.get("sign_code"):
        props["sign_code"] = o["sign_code"]
    if o.get("height_m") is not None:
        props["height_m"] = o["height_m"]
    props["facing"] = None if o.get("facing") is None else round(o["facing"], 1)
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
            "properties": props}


def write_objects(place_dir, objs, to_wgs):
    save(os.path.join(place_dir, "objects.geojson"),
         {"type": "FeatureCollection", "features": [feature(o, to_wgs) for o in objs]})


# ---------------------------------------------------------------- matching

# which of our classes an object found may be: a ringed pole is a light's (or a mast's) pole
COMPAT = {"light": ("light",), "sign": ("sign",), "pole": ("pole", "light"), "lamp": ("lamp",),
          "bollard": ("bollard",)}


def looked_at(shots, to_local, p, reach=30.0):
    """The shots whose view holds p (horizontally) within reach (m)."""
    out = []
    for s in shots:
        d = p - to_local(s["lat"], s["lon"])
        r = float(np.hypot(*d))
        if 2.0 < r < reach and abs(wrap(bearing_of(d) - s["heading"])) < s["fov"] / 2 - 2:
            out.append(s["id"])
    return out


def match(objs, mine, apps, shots, rays, to_local):
    """Each object's match and verdict (against what we ship; where we ship nothing, the open data's nearest), and
    our shipped objects no detection supports (extra_in_ours) appended."""
    ship = [m for m in mine if not m.get("open")]
    opendata = [m for m in mine if m.get("open")]
    taken = set()
    pairs = []
    for oi, o in enumerate(objs):
        for mi, m in enumerate(ship):
            if m["cls"] in COMPAT.get(o["cls"], (o["cls"],)):
                d = float(np.hypot(*(m["p"] - o["p"])))
                if d < MOVED_M:
                    pairs.append((d, oi, mi))
    done = set()
    for d, oi, mi in sorted(pairs):
        if oi in done or mi in taken:
            continue
        done.add(oi)
        taken.add(mi)
        o, m = objs[oi], ship[mi]
        o["match"] = {"source": m["source"], "ref": m["ref"], "dist_m": round(d, 2)}
        notes, verdict = [n for n in (o.get("note"), m["name"]) if n], "ok" if d <= OK_M else "moved"
        if verdict == "moved":
            notes.append(f"ours {d:.1f} m from where the views put it")
        if m["facing"] is not None and o["cls"] in ("light", "sign"):
            rs = [rays[i] for i in o["rays"]]
            f = unit_of(m["facing"])
            fronts = [(r["p"] - o["p"]) @ f / max(np.hypot(*(r["p"] - o["p"])), 1e-6) for r in rs if r.get("face")]
            if fronts and max(fronts) < -FACE_DOT:
                if verdict == "ok":
                    verdict = "wrong_facing"
                sug = o["facing"]
                notes.append(f"its face is seen from behind ours (facing {m['facing']:.0f}°)"
                             + (f"; suggest facing {sug:.0f}°" if sug is not None else ""))
        if o["cls"] == "sign" and o.get("sign_code") and m["codes"] and o["sign_code"] not in m["codes"]:
            if verdict == "ok":
                verdict = "wrong_code"
            notes.append(f"seen {o['sign_code']}, ours {'/'.join(m['codes'])}")
        if o["cls"] == "light":
            a = approach_of(o["p"], o["facing"] if o["facing"] is not None else m["facing"], apps)
            if a:
                notes.append(f"approach: {a}")
                o["approach"] = a
        o["verdict"], o["note"] = verdict, "; ".join(notes)
    for oi, o in enumerate(objs):
        if oi in done:
            continue
        cands = [(float(np.hypot(*(m["p"] - o["p"]))), m) for m in opendata if m["cls"] in COMPAT.get(o["cls"], ())]
        cands = [c for c in cands if c[0] < MOVED_M]
        o["verdict"] = "missing_in_ours"
        pre = (o.get("note") + "; ") if o.get("note") else ""
        if cands:
            d, m = min(cands, key=lambda c: c[0])
            o["match"] = {"source": m["source"], "ref": m["ref"], "dist_m": round(d, 2)}
            if m["source"] == "bgt" and d <= OK_M:
                o["verdict"] = "ok"                 # (the BGT's lamp posts: the world builds them, not the kunstwerk)
                o["note"] = pre + f"{m['name']} {m['ref']} {d:.1f} m away"
            else:
                o["note"] = pre + f"not shipped; {m['name']} {m['ref']} {d:.1f} m away"
        else:
            o["note"] = pre + "not shipped, nor in the open data near it"
        if o["cls"] == "light":
            a = approach_of(o["p"], o["facing"], apps)
            if a:
                o["note"] += f"; approach: {a}"
                o["approach"] = a
    extra = []
    for mi, m in enumerate(ship):
        if mi in taken:
            continue
        seen_by = looked_at(shots, to_local, m["p"], LOOK_M.get(m["cls"], 20.0))
        by = {s["id"]: station_of(s) for s in shots}
        if len({by[i] for i in seen_by}) < LOOK_MIN:
            continue
        single = [r for r in rays if r["cls"] in [c for c, v in COMPAT.items() if m["cls"] in v]
                  and r["shot"] in seen_by
                  and perp(m["p"], r["p"], r["bearing"]) <= tol(float(np.hypot(*(m["p"] - r["p"]))))]
        e = {"id": f"ours-{m['ref']}" + (f"-{mi}" if m["ref"].startswith("items") else ""), "cls": m["cls"],
             "p": m["p"], "n_views": len({r['station'] for r in single}), "residual_m": None, "height_m": None,
             "facing": m["facing"], "shots": seen_by, "rays": [],
             "match": {"source": m["source"], "ref": m["ref"], "dist_m": None}}
        if single:
            e["verdict"] = "ok"
            e["note"] = f"{m['name']}: on the ray of one panorama only ({single[0]['shot']}); not placed"
        else:
            e["verdict"] = "extra_in_ours"
            e["note"] = f"{m['name']}: in view of {len(seen_by)} shots, no detection (or hidden)"
        if m["cls"] == "light":
            a = approach_of(m["p"], m["facing"], apps)
            if a:
                e["approach"] = a
        extra.append(e)
    return objs + extra


def approach_summary(objs, mine, apps):
    """Per approach: the heads we ship for it, the heads seen for it, and whether their facings agree."""
    out = []
    for a in apps:
        back = (a["travel"] + 180.0) % 360.0
        ours_ = [m for m in mine if not m.get("open") and m["cls"] == "light"
                 and approach_of(m["p"], m["facing"], apps) == a["name"]]
        seen = [o for o in objs if o["cls"] == "light" and o.get("approach") == a["name"] and o["rays"]]
        facing = []
        for o in seen:
            if o.get("facing") is not None:
                facing.append({"id": o["id"], "facing": round(o["facing"], 1),
                               "off_deg": round(abs(wrap(o["facing"] - back)), 1)})
        out.append({"approach": a["name"], "drivers_heading": round(a["travel"], 1),
                    "heads_ours": len(ours_), "ours": [m["ref"] for m in ours_],
                    "heads_seen": len(seen), "seen": [o["id"] for o in seen],
                    "seen_matched": [o["match"]["ref"] for o in seen if o.get("match") and o["match"]["ref"]],
                    "facing_seen": facing,
                    "facing_agrees": all(f["off_deg"] <= 45 for f in facing) if facing else None})
    return out


# ---------------------------------------------------------------- report, annotate

VERDICT_NL = {"ok": "klopt", "moved": "verschoven", "missing_in_ours": "ontbreekt bij ons",
              "extra_in_ours": "extra bij ons", "wrong_facing": "verkeerd gericht", "wrong_code": "verkeerde code"}


def report(place_dir, place, objs, n_shots, n_dets, apps_sum=()):
    counts = {}
    for o in objs:
        counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
    lines = [f"# Street View-scan: {place}", "",
             f"{n_shots} shots, {n_dets} detecties, {sum(1 for o in objs if o['rays'])} getrianguleerde objecten. "
             "Diagnose alleen: posities uit Street View gaan niet de mod in (B1); correcties verwijzen naar "
             "open data (NDW-id, OSM-node, BGT-id, placement-index).", "",
             "| oordeel | aantal |", "|---|---|"]
    lines += [f"| {VERDICT_NL.get(k, k)} | {v} |" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))]
    if apps_sum:
        lines += ["", "## Per aanrijrichting", "", "| aanrijrichting | koppen bij ons | gezien | richting klopt |",
                  "|---|---|---|---|"]
        for a in apps_sum:
            fa = {True: "ja", False: "nee", None: "-"}[a["facing_agrees"]]
            lines.append(f"| {a['approach']} | {a['heads_ours']} | {a['heads_seen']} "
                         f"({', '.join(a['seen']) or '-'}) | {fa} |")
    lines += ["", "| id | soort | code | views | rest (m) | hoogte (m) | match | afstand (m) | oordeel | toelichting |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["wrong_facing", "wrong_code", "missing_in_ours", "moved", "extra_in_ours", "ok"]
    for o in sorted(objs, key=lambda o: (order.index(o["verdict"]) if o["verdict"] in order else 9, o["id"])):
        m = o.get("match") or {}
        ref = f"{m.get('source')}:{m.get('ref')}" if m.get("source") else "-"
        lines.append(f"| {o['id']} | {o['cls']} | {o.get('sign_code') or '-'} | {o['n_views']} | "
                     f"{'-' if o['residual_m'] is None else o['residual_m']} | "
                     f"{'-' if o.get('height_m') is None else o['height_m']} | {ref} | "
                     f"{'-' if m.get('dist_m') is None else m['dist_m']} | {VERDICT_NL.get(o['verdict'], '-')} | "
                     f"{o.get('note', '')} |")
    with open(os.path.join(place_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


COLOURS = {"light": (0, 200, 255), "sign": (255, 80, 0), "lamp": (0, 255, 120), "pole": (200, 200, 200),
           "bollard": (255, 0, 255)}


def annotate(place_dir):
    import cv2
    dets = load(os.path.join(place_dir, "detections.json"), [])
    objs = (load(os.path.join(place_dir, "objects.geojson"), {}) or {}).get("features", [])
    shot_obj = {}
    for f in objs:
        if f["properties"].get("residual_m") is None:
            continue
        for s in f["properties"]["shots"]:
            shot_obj.setdefault(s, []).append(f["properties"]["id"])
    out_dir = os.path.join(place_dir, "shots_annotated")
    os.makedirs(out_dir, exist_ok=True)
    by_shot = {}
    for d in dets:
        by_shot.setdefault(d["shot"], []).append(d)
    shots = load(os.path.join(place_dir, "shots.json"), [])
    for s in shots:
        sid = s["id"]
        ds = by_shot.get(sid, [])
        img = cv2.imread(os.path.join(place_dir, "shots", sid + ".png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        shade = img.copy()
        for b in ui_boxes(w, h):
            cv2.rectangle(shade, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 60, 60), -1)
        img = cv2.addWeighted(shade, 0.5, img, 0.5, 0)
        for d in ds:
            x0, y0, x1, y1 = (int(round(v)) for v in d["bbox"])
            col = COLOURS.get(d["cls"], (255, 255, 255))
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            txt = f"{d['cls']} {d['conf']:.2f}"
            if d.get("sign_code"):
                txt += f" {d['sign_code']}"
            cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        txt = f"{sid}: {len(ds)} dets; objects {', '.join(shot_obj.get(sid, [])) or '-'}"
        cv2.putText(img, txt, (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (8, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, sid + ".png"), img)
    print(f"{out_dir}: {len(shots)} shots")


# ---------------------------------------------------------------- commands

def run_triangulate(place_dir, design=None, kunstwerk=None):
    shots = load(os.path.join(place_dir, "shots.json"), [])
    dets = load(os.path.join(place_dir, "detections.json"), [])
    kw = Kunstwerk(design, kunstwerk) if kunstwerk else None
    objs, (to_local, to_wgs, rays) = objects_of(shots, dets, kw)
    for o in objs:
        o["verdict"], o["note"] = None, (o.get("note") + "; " if o.get("note") else "") + "unmatched (run match)"
    write_objects(place_dir, objs, to_wgs)
    print(f"{len(objs)} objects from {len(dets)} detections in {len(shots)} shots")
    return objs


def run_match(place_dir, place, design, kunstwerk):
    shots = load(os.path.join(place_dir, "shots.json"), [])
    dets = load(os.path.join(place_dir, "detections.json"), [])
    kw = Kunstwerk(design, kunstwerk)
    objs, (to_local, to_wgs, rays) = objects_of(shots, dets, kw)
    mine = shipped(kw, to_local)
    apps = approaches(kw, to_local)
    objs = match(objs, mine, apps, shots, rays, to_local)
    summ = approach_summary(objs, mine, apps)
    write_objects(place_dir, objs, to_wgs)
    save(os.path.join(place_dir, "approaches.json"), summ)
    report(place_dir, place, objs, len(shots), len(dets), summ)
    counts = {}
    for o in objs:
        counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
    n_ship = sum(1 for m in mine if not m.get("open"))
    print(f"{len(objs)} objects: {counts} (ours shipped: {n_ship})")
    return objs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--place", required=True)
        p.add_argument("--root", default=SV_ROOT, help="the sv folder (design/reference/sv)")
        return p
    p = common(sub.add_parser("shot-meta", help="append a screenshot's record to shots.json from its URL"))
    p.add_argument("--url", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--png", help="the screenshot (copied to shots/<id>.png when elsewhere)")
    p.add_argument("--fov-axis", choices=("v", "h"), default="v", help="what Maps' y is (default vertical)")
    for name in ("detect", "all"):
        p = common(sub.add_parser(name))
        p.add_argument("--model", choices=("world", "coco", "both"), default="world")
        p.add_argument("--conf", type=float, default=0.12)
        p.add_argument("--imgsz", type=int, default=1280)
        p.add_argument("--no-mask", action="store_true", help="detect in Maps' overlays too")
        if name == "all":
            p.add_argument("--kunstwerk")
            p.add_argument("--design", default=DESIGN)
    p = common(sub.add_parser("triangulate"))
    p.add_argument("--kunstwerk")
    p.add_argument("--design", default=DESIGN)
    p = common(sub.add_parser("match"))
    p.add_argument("--kunstwerk", required=True)
    p.add_argument("--design", default=DESIGN)
    common(sub.add_parser("annotate"))
    a = ap.parse_args(argv)
    place_dir = os.path.join(a.root, a.place)

    if a.cmd == "shot-meta":
        import cv2
        os.makedirs(os.path.join(place_dir, "shots"), exist_ok=True)
        dst = os.path.join(place_dir, "shots", a.id + ".png")
        src = a.png or dst
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        img = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"cannot read {dst}")
        rec = shot_record(a.url, a.id, img.shape[1], img.shape[0], a.fov_axis)
        path = os.path.join(place_dir, "shots.json")
        shots = [s for s in load(path, []) if s["id"] != a.id] + [rec]
        save(path, shots)
        print(json.dumps(rec))
    elif a.cmd == "detect":
        detect(place_dir, a.model, a.conf, a.imgsz, mask=not a.no_mask)
    elif a.cmd == "triangulate":
        run_triangulate(place_dir, a.design, a.kunstwerk)
    elif a.cmd == "match":
        run_match(place_dir, a.place, a.design, a.kunstwerk)
    elif a.cmd == "annotate":
        annotate(place_dir)
    elif a.cmd == "all":
        detect(place_dir, a.model, a.conf, a.imgsz, mask=not a.no_mask)
        if a.kunstwerk:
            run_match(place_dir, a.place, a.design, a.kunstwerk)
        else:
            run_triangulate(place_dir)
        annotate(place_dir)


if __name__ == "__main__":
    main()
