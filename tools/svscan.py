#!/usr/bin/env python3
"""Street furniture read off Street View screenshots, set against our open data: a diagnosis, never a source
(fork-only tool; never upstream). Claude takes the screenshots by hand in a browser (Google Maps' Street View,
looking round a place such as the Droogleever Fortuynplein); this tool only reads the PNG files already on disk:
no scraping, no Google API. YOLO finds the traffic lights, signs, lamp posts, poles and bollards in each shot,
each detection becomes a compass ray from the known camera, the rays of one object from several shots are
triangulated, and each object found is matched to OUR objects (the kunstwerk's placement.json and
direction_signs.json) and to the open data they come from (NDW's sign register, OSM's signal nodes). The output
says "this head belongs to that approach / this sign faces the wrong way / this sign is missing / ours is extra";
the positions read off Google never go into the mod (project decision B1): every correction is a reference to an
open-data object (an NDW id, an OSM node, a placement index) plus a suggested facing or approach.

    python3 tools/svscan.py shot-meta --place droogleever --id A269 --png shot.png --url "<Street View URL>"
    python3 tools/svscan.py detect --place droogleever [--model world|coco|both] [--conf 0.12]
    python3 tools/svscan.py triangulate --place droogleever
    python3 tools/svscan.py match --place droogleever --kunstwerk droogleever
    python3 tools/svscan.py annotate --place droogleever
    python3 tools/svscan.py all --place droogleever --kunstwerk droogleever
    (--root <sv dir> and --design <design dir> when not run from E:\\BeamNG\\src; or $BNE_SV_ROOT, $BNE_DESIGN)

Screenshot protocol (Google Maps, Street View):
- Take the shot of the Street View viewport only (no side panel, no browser chrome; the small overlays Maps draws
  over the picture are fine), unzoomed by the page (browser zoom 100 %), and save it as
  <root>/<place>/shots/<id>.png. The picture's centre must be the view's centre: crop symmetrically if at all.
- Copy the page's URL at that moment. Street View's URL reads
  .../@51.9080,4.4660,3a,75y,269h,92t/data=!3m6!1e1!3m4!1s<pano id>!2e0...
  51.9080,4.4660 the camera (the panorama's place), 3a the Street View mode, 75y the field of view (deg),
  269h the heading (compass deg of the picture's centre), 92t the tilt (90 = the horizon; pitch = tilt - 90),
  !1s<pano> the panorama's id. Google Earth's URLs (@lat,lon,<alt>a,<dist>d,<fov>y,<heading>h,<tilt>t) read alike.
- `shot-meta` reads all of it from the URL and appends (or replaces) the shot's record in shots.json, with the
  PNG's size. Maps' "y" is taken as the VERTICAL field of view (the picture widens with the window, the y stays):
  shots.json's fov is the horizontal one, 2 atan(tan(y/2) w/h). --fov-axis h takes y as horizontal instead.
- Use several panoramas round the place (each object wants two views or more, from spots 10 m or more apart,
  ideally at 30 deg or more to each other) and a few headings per panorama; look at the signs' faces where
  possible (a sign's code is read from its face; a back only says which way it does NOT face).

Data (shared with the dashboard; keep exact), in <root>/<place>/ (root E:\\BeamNG\\design\\reference\\sv):
- shots/<id>.png; shots.json [{"id","lat","lon","heading","pitch","fov","w","h","pano"?,"taken"?}]
- detections.json [{"shot","cls","label","conf","bbox":[x0,y0,x1,y1],"bearing","elev","sign_code"?,"sign_score"?,
  "face"?}] (bearing, elev: deg of the box's centre; cls one of light, sign, lamp, pole, bollard; label the prompt;
  sign_code an RVV code, "back" or "unknown"; face true where a sign's face or a head's lit lens is seen)
- objects.geojson: Points (WGS84) {"id","cls","sign_code"?,"n_views","residual_m","height_m"?,"facing"?,"shots",
  "match":{"source","ref","dist_m"},"verdict","note"}; verdict ok | moved | missing_in_ours | extra_in_ours |
  wrong_facing | wrong_code (null after triangulate alone, before match); our objects no detection supports but
  some shot looked at are features too (extra_in_ours).
- report.md: a short Dutch summary. shots_annotated/: the shots with their boxes, for people.
The camera stands 2.5 m over the ground (the Street View car's mast); height_m is the box centre's height over the
camera's ground. Positions use lat/lon only (a local east/north frame round the shots).

Accuracy, honestly: Street View's camera positions are GPS (often 1-3 m off) and the heading of a Maps URL is
rounded to a degree, so the triangulated points are good to 1-3 m at 20 m; OK_M and MOVED_M are set for that.
The sign classifier (colour classes + grey correlation against tools/sign_atlas.py's cells) is reliable only on
crops of 40 px or more and on the common round/triangular signs; small or oblique crops come back "unknown".

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
TOL_M = 1.0                 # m: a ray supports a point passing this near it ...
TOL_DEG = 2.0               # ... or within this angle of it, whichever is the more
MAX_RESIDUAL = 2.5          # m: rms miss of an object's rays
HEIGHT_TOL = 1.5            # m: the rays of one object agree on its height to this
OK_M = 3.0                  # m: ours this near the object found is where it is
MOVED_M = 10.0              # m: ours this near is the same object, moved
LOOK_M = 35.0               # m: a shot looked at our object nearer than this and inside its view
FACE_DOT = 0.2              # a camera this far (cos) in front of / behind our facing sees its face / back

# YOLO-World's prompts and the class each one counts as
PROMPTS = {"traffic light": "light", "traffic signal head": "light", "traffic sign": "sign",
           "road sign on pole": "sign", "street lamp": "lamp", "pole": "pole", "bollard": "bollard"}
COCO = {"traffic light": "light", "stop sign": "sign"}
WORLD_W = "yolov8s-worldv2.pt"
COCO_W = "yolo11m.pt"
CODE_RE = re.compile(r"^[A-L]\d+[a-z]?$")
SIGN_MIN_SCORE = 0.55       # a sign classification under this is "unknown"


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


def pixel_ray(u, v, w, h, heading, pitch, fov):
    """(bearing, elev) in deg of pixel (u, v) of a w x h rectilinear picture looking along heading (compass deg),
    pitch (deg, 0 = level), with a horizontal field of view fov (deg)."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    x, y = u - w / 2.0, v - h / 2.0                 # right, down
    hd, p = math.radians(heading), math.radians(pitch)
    F = np.array([math.sin(hd) * math.cos(p), math.cos(hd) * math.cos(p), math.sin(p)])    # east, north, up
    R = np.array([math.cos(hd), -math.sin(hd), 0.0])
    U = np.cross(R, F)
    d = x * R - y * U + f * F
    return math.degrees(math.atan2(d[0], d[1])) % 360.0, math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def world_to_pixel(bearing, elev, w, h, heading, pitch, fov):
    """pixel_ray inverted: (u, v), or None behind the camera."""
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    b, e = math.radians(bearing), math.radians(elev)
    d = np.array([math.sin(b) * math.cos(e), math.cos(b) * math.cos(e), math.sin(e)])
    hd, p = math.radians(heading), math.radians(pitch)
    F = np.array([math.sin(hd) * math.cos(p), math.cos(hd) * math.cos(p), math.sin(p)])
    R = np.array([math.cos(hd), -math.sin(hd), 0.0])
    U = np.cross(R, F)
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


TAN_TOL = math.tan(math.radians(TOL_DEG))


def tol(r):
    return max(TOL_M, r * TAN_TOL)


def triangulate(rays):
    """Objects from rays [{"p": (e, n), "bearing", "elev", "shot", "cls", ...}]: greedy, the point most shots
    agree on first. Returns [{"p", "cls", "rays": [indices], "residual", "height"}]."""
    out = []
    by_cls = {}
    for i, r in enumerate(rays):
        by_cls.setdefault(r["cls"], []).append(i)
    for cls, idx in by_cls.items():
        P = np.array([rays[i]["p"] for i in idx], float).reshape(-1, 2)
        B = np.radians([rays[i]["bearing"] for i in idx])
        U = np.stack([np.sin(B), np.cos(B)], 1) if len(idx) else np.zeros((0, 2))
        shot = [rays[i]["shot"] for i in idx]
        cands = []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if shot[a] == shot[b] or np.hypot(*(P[a] - P[b])) < 1.0:
                    continue
                hit = intersect(P[a], rays[idx[a]]["bearing"], P[b], rays[idx[b]]["bearing"])
                if hit is None:
                    continue
                x, t1, t2 = hit
                if MIN_RANGE < t1 < MAX_RANGE and MIN_RANGE < t2 < MAX_RANGE:
                    cands.append((x, a, b))
        free = np.ones(len(idx), bool)

        def support(x):
            """Per shot the free ray nearest x that supports it: [local index]."""
            d = x[None, :] - P
            t = (d * U).sum(1)
            r = np.hypot(d[:, 0], d[:, 1])
            m = np.abs(d[:, 0] * U[:, 1] - d[:, 1] * U[:, 0])
            ok = free & (t > MIN_RANGE) & (r < MAX_RANGE) & (m <= np.maximum(TOL_M, r * TAN_TOL))
            best = {}
            for k in np.flatnonzero(ok):
                if shot[k] not in best or m[k] < m[best[shot[k]]]:
                    best[shot[k]] = k
            ks = list(best.values())
            if len(ks) >= 3:                         # the rays agree on its height
                hs = np.array([CAM_H + r[k] * math.tan(math.radians(rays[idx[k]]["elev"])) for k in ks])
                med = float(np.median(hs))
                ks = [k for k, hh in zip(ks, hs) if abs(hh - med) <= max(HEIGHT_TOL, 0.3 * abs(med - CAM_H))]
            return ks
        while cands:
            best = None
            for x, a, b in cands:
                if not (free[a] and free[b]):
                    continue
                ks = support(x)
                if len(ks) < 2:
                    continue
                y, res = solve_point([P[k] for k in ks], [rays[idx[k]]["bearing"] for k in ks])
                key = (len(ks), -res)
                if best is None or key > best[0]:
                    best = (key, ks)
            if best is None:
                break
            ks = best[1]
            for _ in range(3):                      # refine: re-solve, re-gather
                y, res = solve_point([P[k] for k in ks], [rays[idx[k]]["bearing"] for k in ks])
                ks2 = support(y)
                if len(ks2) < 2 or set(ks2) == set(ks):
                    break
                ks = ks2
            y, res = solve_point([P[k] for k in ks], [rays[idx[k]]["bearing"] for k in ks])
            for k in ks:
                free[k] = False
            if res > MAX_RESIDUAL:
                continue
            hs = [CAM_H + np.hypot(*(y - P[k])) * math.tan(math.radians(rays[idx[k]]["elev"])) for k in ks]
            out.append({"p": y, "cls": cls, "rays": [idx[k] for k in ks], "residual": res,
                        "height": float(np.median(hs))})
            cands = [c for c in cands if free[c[1]] and free[c[2]]]
    return out


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


def shot_record(url, sid, w, h, fov_axis="v"):
    """The shots.json record of a screenshot w x h taken at url."""
    u = parse_url(url)
    fy = u["fov_y"]
    fov = 2 * math.degrees(math.atan(math.tan(math.radians(fy) / 2) * w / h)) if fov_axis == "v" else fy
    rec = {"id": sid, "lat": u["lat"], "lon": u["lon"], "heading": round(u["heading"], 2),
           "pitch": round(u["tilt"] - 90.0, 2), "fov": round(fov, 3), "w": int(w), "h": int(h)}
    if "pano" in u:
        rec["pano"] = u["pano"]
    rec["taken"] = datetime.datetime.now().isoformat(timespec="seconds")
    return rec


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


# ---------------------------------------------------------------- signs

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


def detect(place_dir, kind="world", conf=0.12, imgsz=1280, atlas=None):
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
        found = []
        for m, names in models:
            r = m.predict(img, conf=conf, imgsz=imgsz, agnostic_nms=True, verbose=False)[0]
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                label = m.names[int(k)]
                if label not in names:
                    continue
                if any(iou(b, f["bbox"]) > 0.6 and names[label] == f["cls"] for f in found):
                    continue
                found.append({"cls": names[label], "label": label, "conf": round(float(c), 3),
                              "bbox": [round(float(v), 1) for v in b]})
        for f in found:
            x0, y0, x1, y1 = f["bbox"]
            sx, sy = w / s["w"], h / s["h"]          # (the record's pixel grid)
            bearing, elev = pixel_ray((x0 + x1) / 2 / sx, (y0 + y1) / 2 / sy, s["w"], s["h"], s["heading"],
                                      s["pitch"], s["fov"])
            d = {"shot": s["id"], **f, "bearing": round(bearing, 3), "elev": round(elev, 3)}
            crop = img[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))]
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


# ---------------------------------------------------------------- objects

def place_frame(shots):
    lat0 = float(np.mean([s["lat"] for s in shots]))
    lon0 = float(np.mean([s["lon"] for s in shots]))
    return frame(lat0, lon0)


def rays_of(shots, dets, to_local):
    by_id = {s["id"]: s for s in shots}
    rays = []
    for i, d in enumerate(dets):
        s = by_id.get(d["shot"])
        if s is None:
            continue
        rays.append({**d, "det": i, "p": to_local(s["lat"], s["lon"])})
    return rays


def majority_code(rs):
    votes = {}
    for r in rs:
        c = r.get("sign_code")
        if c and CODE_RE.match(c):
            votes[c] = votes.get(c, 0.0) + float(r.get("sign_score") or 0)
    return max(votes, key=votes.get) if votes else None


def objects_of(shots, dets):
    """The triangulated objects: [{"id", "cls", "p", "n_views", "residual_m", "height_m", "facing", "shots",
    "sign_code", "rays"}] in the place's local frame, and (to_local, to_wgs, rays)."""
    to_local, to_wgs = place_frame(shots)
    rays = rays_of(shots, dets, to_local)
    objs = []
    counts = {}
    for o in triangulate(rays):
        rs = [rays[i] for i in o["rays"]]
        counts[o["cls"]] = counts.get(o["cls"], 0) + 1
        objs.append({"id": f"sv-{o['cls']}-{counts[o['cls']]}", "cls": o["cls"], "p": o["p"],
                     "n_views": len(rs), "residual_m": round(o["residual"], 2), "height_m": round(o["height"], 2),
                     "facing": facing_of(o["p"], [(r["p"], r.get("face")) for r in rs]),
                     "shots": sorted({r["shot"] for r in rs}), "sign_code": majority_code(rs),
                     "rays": o["rays"]})
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


# ---------------------------------------------------------------- our objects, open data

def ours(design, kunstwerk, to_local):
    """Our objects and the open data round the kunstwerk, in the place's frame: [{"source", "ref", "cls", "p",
    "facing" (compass deg or None), "codes" (list), "name"}]. source placement (placement.json items[i],
    direction_signs.json signs[i]: what the mod builds), ndw (ref/ndw/signs.json), osm (signal nodes)."""
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from ahn import rd_to_wgs
    ref_dir = os.path.join(design, "kunstwerken", kunstwerk, "ref")
    refj = load(os.path.join(ref_dir, "ref.json")) or \
        load(os.path.join(ROOT, "server", "data", "kunstwerken", f"{kunstwerk}.kunstwerk.json"))
    ox, oy = refj["origin_rd"]

    def from_rd(x, y):
        lat, lon = rd_to_wgs(ox + x, oy + y)
        return to_local(float(lat), float(lon))

    def facing(at, f):
        if f is None:
            return None
        a = from_rd(*at)
        b = from_rd(at[0] + f[0], at[1] + f[1])     # (RD's grid north is not true north)
        return bearing_of(b - a)
    out = []
    pl = load(os.path.join(ref_dir, "placement.json"), {"items": []})
    for i, it in enumerate(pl.get("items", [])):
        cls = {"traffic_light": "light", "sign": "sign"}.get(it.get("module"))
        if cls is None:
            continue
        code = it.get("code")
        codes = code if isinstance(code, list) else ([code] if code else [])
        codes += [c for c in it.get("plates", []) if c not in codes]
        out.append({"source": "placement", "ref": f"items[{i}]", "cls": cls, "p": from_rd(*it["at"]),
                    "facing": facing(it["at"], it.get("facing")), "codes": codes,
                    "name": f"{it.get('module')} {it.get('role', '')} {' '.join(codes)}".strip()})
    ds = load(os.path.join(ref_dir, "direction_signs.json"), {"signs": []})
    for i, it in enumerate(ds.get("signs", [])):
        codes = [a for a in it.get("args", []) if isinstance(a, str) and CODE_RE.match(a)]
        out.append({"source": "placement", "ref": f"direction_signs[{i}]", "cls": "sign", "p": from_rd(*it["at"]),
                    "facing": facing(it["at"], it.get("facing")), "codes": codes, "name": it.get("module")})
    for f in load(os.path.join(ref_dir, "ndw", "signs.json"), []) or []:
        pr = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        b = pr.get("bearing")
        out.append({"source": "ndw", "ref": str(f.get("id") or pr.get("id") or pr.get("externalId")), "cls": "sign",
                    "p": to_local(lat, lon), "facing": None if b is None else (float(b) + 180.0) % 360.0,
                    "codes": [pr["rvvCode"]] if pr.get("rvvCode") else [], "name": f"NDW {pr.get('rvvCode')}"})
    seen = set()
    for e in (load(os.path.join(ref_dir, "osm_signals.json"), {}) or {}).get("elements", []):
        if e.get("type") == "node" and e.get("tags", {}).get("highway") == "traffic_signals":
            seen.add(e["id"])
            out.append({"source": "osm", "ref": f"node/{e['id']}", "cls": "light", "p": to_local(e["lat"], e["lon"]),
                        "facing": None, "codes": [], "name": "OSM traffic_signals"})
    xml = os.path.join(ref_dir, "osm_map.xml")
    if os.path.exists(xml):
        for n in ET.parse(xml).getroot().iter("node"):
            tags = {t.get("k"): t.get("v") for t in n.iter("tag")}
            cls = ("light" if tags.get("highway") == "traffic_signals" else "lamp" if tags.get("highway") ==
                   "street_lamp" else "bollard" if tags.get("barrier") == "bollard" else "sign"
                   if tags.get("traffic_sign") else None)
            nid = int(n.get("id"))
            if cls is None or nid in seen:
                continue
            codes = [c.split(":")[-1] for c in (tags.get("traffic_sign") or "").split(";") if c]
            out.append({"source": "osm", "ref": f"node/{nid}", "cls": cls,
                        "p": to_local(float(n.get("lat")), float(n.get("lon"))), "facing": None, "codes": codes,
                        "name": f"OSM {cls}"})
    apps = []
    for a in (load(os.path.join(ref_dir, "approaches.json"), {}) or {}).get("approaches", []):
        L, R = from_rd(*a["left"]), from_rd(*a["right"])
        d = R - L
        travel = np.array([-d[1], d[0]])             # (left to right as the arriving drivers see it)
        apps.append({"name": a["name"], "mid": (L + R) / 2, "travel": bearing_of(travel)})
    return out, apps


def approach_of(p, facing, apps):
    """The approach a head at p facing (compass deg) speaks to: the nearest stop line whose drivers it faces."""
    best = None
    for a in apps:
        if facing is not None and abs(wrap(facing - (a["travel"] + 180.0))) > 60:
            continue
        d = float(np.hypot(*(a["mid"] - p)))
        if d < 40 and (best is None or d < best[0]):
            best = (d, a["name"])
    return best[1] if best else None


def looked_at(shots, to_local, p):
    """The shots whose view holds p (horizontally) within LOOK_M."""
    out = []
    for s in shots:
        d = p - to_local(s["lat"], s["lon"])
        r = float(np.hypot(*d))
        if 2.0 < r < LOOK_M and abs(wrap(bearing_of(d) - s["heading"])) < s["fov"] / 2 - 2:
            out.append(s["id"])
    return out


def match(objs, mine, apps, shots, rays, to_local):
    """Each object's match and verdict, and our objects no detection supports (extra_in_ours) appended."""
    placed = [m for m in mine if m["source"] == "placement"]
    taken = set()
    pairs = []
    for oi, o in enumerate(objs):
        for mi, m in enumerate(placed):
            if m["cls"] == o["cls"] or {m["cls"], o["cls"]} <= {"sign", "pole"}:
                d = float(np.hypot(*(m["p"] - o["p"])))
                if d < MOVED_M:
                    pairs.append((d, oi, mi))
    done = set()
    for d, oi, mi in sorted(pairs):
        if oi in done or mi in taken:
            continue
        done.add(oi)
        taken.add(mi)
        o, m = objs[oi], placed[mi]
        o["match"] = {"source": m["source"], "ref": m["ref"], "dist_m": round(d, 2)}
        notes, verdict = [m["name"]], "ok" if d <= OK_M else "moved"
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
        o["verdict"], o["note"] = verdict, "; ".join(notes)
    for oi, o in enumerate(objs):
        if oi in done:
            continue
        cands = [(float(np.hypot(*(m["p"] - o["p"]))), m) for m in mine
                 if m["source"] != "placement" and m["cls"] == o["cls"]]
        cands = [c for c in cands if c[0] < MOVED_M]
        o["verdict"] = "missing_in_ours"
        if cands:
            d, m = min(cands, key=lambda c: c[0])
            o["match"] = {"source": m["source"], "ref": m["ref"], "dist_m": round(d, 2)}
            o["note"] = f"not in our placement; {m['name']} {m['ref']} {d:.1f} m away"
        else:
            o["note"] = "not in our placement nor in the open data near it"
        if o["cls"] == "light":
            a = approach_of(o["p"], o["facing"], apps)
            if a:
                o["note"] += f"; approach: {a}"
    extra = []
    for mi, m in enumerate(placed):
        if mi in taken:
            continue
        seen_by = looked_at(shots, to_local, m["p"])
        if not seen_by:
            continue
        single = [r for r in rays if r["cls"] == m["cls"] and r["shot"] in seen_by
                  and perp(m["p"], r["p"], r["bearing"]) <= tol(float(np.hypot(*(m["p"] - r["p"]))))]
        e = {"id": f"ours-{m['ref']}", "cls": m["cls"], "p": m["p"], "n_views": len({r['shot'] for r in single}),
             "residual_m": None, "height_m": None, "facing": m["facing"], "shots": seen_by,
             "match": {"source": m["source"], "ref": m["ref"], "dist_m": None}, "rays": []}
        if single:
            e["verdict"] = "ok"
            e["note"] = f"{m['name']}: on the ray of one view only ({single[0]['shot']}); not placed"
        else:
            e["verdict"] = "extra_in_ours"
            e["note"] = f"{m['name']}: in view of {len(seen_by)} shots, no detection (or hidden)"
        extra.append(e)
    return objs + extra


# ---------------------------------------------------------------- report, annotate

VERDICT_NL = {"ok": "klopt", "moved": "verschoven", "missing_in_ours": "ontbreekt bij ons",
              "extra_in_ours": "extra bij ons", "wrong_facing": "verkeerd gericht", "wrong_code": "verkeerde code"}


def report(place_dir, place, objs, n_shots, n_dets):
    counts = {}
    for o in objs:
        counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
    lines = [f"# Street View-scan: {place}", "",
             f"{n_shots} shots, {n_dets} detecties, {sum(1 for o in objs if o['rays'])} getrianguleerde objecten. "
             "Diagnose alleen: posities uit Street View gaan niet de mod in (B1); correcties verwijzen naar "
             "open data (NDW-id, OSM-node, placement-index).", "",
             "| oordeel | aantal |", "|---|---|"]
    lines += [f"| {VERDICT_NL.get(k, k)} | {v} |" for k, v in sorted(counts.items())]
    lines += ["", "| id | soort | code | views | rest (m) | match | afstand (m) | oordeel | toelichting |",
              "|---|---|---|---|---|---|---|---|---|"]
    order = ["wrong_facing", "wrong_code", "missing_in_ours", "moved", "extra_in_ours", "ok"]
    for o in sorted(objs, key=lambda o: (order.index(o["verdict"]) if o["verdict"] in order else 9, o["id"])):
        m = o.get("match") or {}
        ref = f"{m.get('source')}:{m.get('ref')}" if m.get("source") else "-"
        lines.append(f"| {o['id']} | {o['cls']} | {o.get('sign_code') or '-'} | {o['n_views']} | "
                     f"{'-' if o['residual_m'] is None else o['residual_m']} | {ref} | "
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
        for s in f["properties"]["shots"]:
            shot_obj.setdefault(s, []).append(f["properties"]["id"])
    out_dir = os.path.join(place_dir, "shots_annotated")
    os.makedirs(out_dir, exist_ok=True)
    by_shot = {}
    for d in dets:
        by_shot.setdefault(d["shot"], []).append(d)
    for sid, ds in by_shot.items():
        img = cv2.imread(os.path.join(place_dir, "shots", sid + ".png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for d in ds:
            x0, y0, x1, y1 = (int(round(v)) for v in d["bbox"])
            col = COLOURS.get(d["cls"], (255, 255, 255))
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            txt = f"{d['cls']} {d['conf']:.2f}"
            if d.get("sign_code"):
                txt += f" {d['sign_code']}"
            cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"{sid}: {len(ds)} dets; objects {', '.join(shot_obj.get(sid, [])) or '-'}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, sid + ".png"), img)
    print(f"{out_dir}: {len(by_shot)} shots")


# ---------------------------------------------------------------- commands

def run_triangulate(place_dir):
    shots = load(os.path.join(place_dir, "shots.json"), [])
    dets = load(os.path.join(place_dir, "detections.json"), [])
    objs, (to_local, to_wgs, rays) = objects_of(shots, dets)
    for o in objs:
        o["verdict"], o["note"] = None, "unmatched (run match)"
    write_objects(place_dir, objs, to_wgs)
    print(f"{len(objs)} objects from {len(dets)} detections in {len(shots)} shots")
    return objs


def run_match(place_dir, place, design, kunstwerk):
    shots = load(os.path.join(place_dir, "shots.json"), [])
    dets = load(os.path.join(place_dir, "detections.json"), [])
    objs, (to_local, to_wgs, rays) = objects_of(shots, dets)
    mine, apps = ours(design, kunstwerk, to_local)
    objs = match(objs, mine, apps, shots, rays, to_local)
    write_objects(place_dir, objs, to_wgs)
    report(place_dir, place, objs, len(shots), len(dets))
    counts = {}
    for o in objs:
        counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
    print(f"{len(objs)} objects: {counts}")
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
        if name == "all":
            p.add_argument("--kunstwerk")
            p.add_argument("--design", default=DESIGN)
    common(sub.add_parser("triangulate"))
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
        detect(place_dir, a.model, a.conf, a.imgsz)
    elif a.cmd == "triangulate":
        run_triangulate(place_dir)
    elif a.cmd == "match":
        run_match(place_dir, a.place, a.design, a.kunstwerk)
    elif a.cmd == "annotate":
        annotate(place_dir)
    elif a.cmd == "all":
        detect(place_dir, a.model, a.conf, a.imgsz)
        if a.kunstwerk:
            run_match(place_dir, a.place, a.design, a.kunstwerk)
        else:
            run_triangulate(place_dir)
        annotate(place_dir)


if __name__ == "__main__":
    main()
