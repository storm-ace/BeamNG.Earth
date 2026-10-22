"""The dashboard's data: where everything lives (Config) and each layer read into one shape.

An object (every point layer): {"id", "layer", "kind", "label", "at": [x, y], "z", "facing": [fx, fy] | None,
"heading": degrees | None, "editable", "props": {...}} in the kunstwerk's local frame (geo.py).
"""
import json
import os
import sys
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .geo import Frame, facing_from_bearing, heading_of

HERE = os.path.dirname(os.path.abspath(__file__))
OSM_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
             "living_street", "pedestrian", "cycleway", "footway", "path", "motorway_link", "trunk_link",
             "primary_link", "secondary_link", "tertiary_link", "busway"}


def find_design(start):
    """The design folder (with kunstwerken/) next to the repo or one of its parents (a worktree sits deeper)."""
    d = os.path.abspath(start)
    while True:
        cand = os.path.join(os.path.dirname(d), "design")
        if os.path.isdir(os.path.join(cand, "kunstwerken")):
            return cand
        if os.path.dirname(d) == d:
            return os.path.join(os.path.dirname(os.path.abspath(start)), "design")
        d = os.path.dirname(d)


@dataclass
class Config:
    repo: str = os.path.normpath(os.path.join(HERE, "..", ".."))     # its tools/ run, its server/data written
    design: str = ""
    name: str = "droogleever"
    sv_root: str = ""                       # Street View scans: <sv_root>/<place>/objects.geojson ...
    place: str = ""                         # (default: name)
    mcp_url: str = "http://127.0.0.1:29292/mcp"
    python: str = sys.executable
    steps: list = field(default=None)       # the rebuild's commands (None: rebuild.default_steps)
    export_dir: str = ""                    # where the export goes (default: <repo>/server/data/kunstwerken: earthd's)

    def __post_init__(self):
        self.repo = os.path.abspath(self.repo)
        self.export_dir = os.path.abspath(self.export_dir or os.path.join(self.repo, "server", "data", "kunstwerken"))
        self.design = os.path.abspath(self.design or find_design(self.repo))
        self.sv_root = os.path.abspath(self.sv_root or os.path.join(self.design, "reference", "sv"))
        self.place = self.place or self.name
        tools = os.path.join(self.repo, "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)

    @property
    def kw_dir(self):
        return os.path.join(self.design, "kunstwerken", self.name)

    @property
    def ref(self):
        return os.path.join(self.kw_dir, "ref")

    @property
    def mesh_path(self):
        return os.path.join(self.export_dir, f"{self.name}.kunstwerk.json")

    @property
    def overrides_path(self):
        return os.path.join(self.ref, "overrides.json")


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class _Cached:
    """A value read from a file, read again when its mtime changes."""

    def __init__(self, fn):
        self.fn, self.key, self.val, self.lock = fn, None, None, threading.Lock()

    def get(self, path):
        key = (path, os.stat(path).st_mtime_ns) if os.path.exists(path) else (path, None)
        with self.lock:
            if key != self.key:
                self.val, self.key = self.fn(path), key
            return self.val


class Ground:
    """The AHN's ground (ref/ahn.npz: dtm, res, x0, y1; row 0 north) sampled at local x, y (nearest cell)."""

    def __init__(self, path):
        self.ok = os.path.exists(path)
        if self.ok:
            z = np.load(path)
            self.dtm, self.res = z["dtm"].astype(np.float64), float(z["res"])
            self.x0, self.y1 = float(z["x0"]), float(z["y1"])
            good = np.isfinite(self.dtm)
            self.fill = float(np.median(self.dtm[good])) if good.any() else 0.0
            self.dtm[~good] = self.fill

    def at(self, x, y):
        if not self.ok:
            return 0.0
        i = int(np.clip((self.y1 - float(y)) / self.res, 0, self.dtm.shape[0] - 1))
        j = int(np.clip((float(x) - self.x0) / self.res, 0, self.dtm.shape[1] - 1))
        return float(self.dtm[i, j])

    def grid(self, box, step=2.0):
        """Heights over box (x0, y0, x1, y1) on a lattice about step apart, row 0 north (a three.js PlaneGeometry's
        vertex order): {"nx", "ny" (segments), "z": [(ny + 1) * (nx + 1)]}."""
        x0, y0, x1, y1 = box
        nx, ny = max(1, round((x1 - x0) / step)), max(1, round((y1 - y0) / step))
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y1, y0, ny + 1)
        z = [round(self.at(x, y), 2) for y in ys for x in xs]
        return {"nx": nx, "ny": ny, "z": z}


def _obj(id_, layer, kind, label, at, z, facing=None, editable=False, props=None):
    f = None if facing is None else [round(float(facing[0]), 4), round(float(facing[1]), 4)]
    return {"id": id_, "layer": layer, "kind": kind, "label": label,
            "at": [round(float(at[0]), 3), round(float(at[1]), 3)], "z": round(float(z), 3),
            "facing": f, "heading": None if f is None else round(heading_of(f), 1),
            "editable": editable, "props": props or {}}


def _label(it):
    c = it.get("code")
    if c:
        return c if isinstance(c, str) else "+".join(c)
    if it["module"] == "traffic_light":
        return ("VRI-mast" if it.get("mast") else "VRI") + (" herh." if it.get("role") == "repeater" else "") \
            + (" +fiets" if it.get("bike") else "")
    return it.get("module", "?")


class Data:
    """Every layer of one kunstwerk, read from its files on each call (cheap) or cached by mtime (the big ones)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._ground = _Cached(Ground)
        self._osm = _Cached(self._read_osm)
        self._ref = _Cached(_load)

    # -- frame, ground
    def refjson(self):
        return self._ref.get(os.path.join(self.cfg.ref, "ref.json")) or {}

    def frame(self):
        r = self.refjson()
        origin = r.get("origin_rd")
        if origin is None:
            mesh = _mesh_head(self.cfg.mesh_path)
            origin = mesh.get("origin_rd", [0, 0])
        return Frame(origin, self.cfg.repo)

    def ground(self):
        return self._ground.get(os.path.join(self.cfg.ref, "ahn.npz"))

    def approaches(self):
        return (_load(os.path.join(self.cfg.ref, "approaches.json"), {}) or {}).get("approaches", [])

    def overrides(self):
        return _load(self.cfg.overrides_path, []) or []

    # -- our furniture: placement.json + placement_fix.json + overrides.json, as tools/kunstwerk_solid.py builds it
    def _bases(self, log=None):
        """(placement items after the fixes, direction signs): what the overrides apply to."""
        import kunstwerk_placement as kp
        ref = self.cfg.ref
        items = (_load(os.path.join(ref, "placement.json"), {}) or {}).get("items", [])
        fixes = (_load(os.path.join(ref, "placement_fix.json"), {}) or {}).get("fixes", [])
        ds = (_load(os.path.join(ref, "direction_signs.json"), {}) or {}).get("signs", [])
        for i, it in enumerate(items):                  # (where it came from: the Street View scan's match refs)
            it["src_ref"] = f"items[{i}]"
        for i, it in enumerate(ds):
            it["src_ref"] = f"direction_signs[{i}]"
        return kp.apply_fixes(items, fixes, log=log or (lambda m: None)), ds

    def ours(self):
        """Our lights and signs (layer ours) and direction signs (layer direction) as tools/kunstwerk_solid.py places
        them: placement.json, placement_fix.json, then overrides.json; each with its place before the override
        (props base_at, base_facing) and its override; one an override removes kept, marked removed."""
        import kunstwerk_placement as kp
        msgs = []
        aps, ovs = self.approaches(), self.overrides()
        by_ov = {o["id"]: o for o in ovs}
        g = self.ground()
        out = []
        for layer, base in zip(("ours", "direction"), self._bases(msgs.append)):
            base_by = {it["id"]: it for it in kp.apply_overrides(base, [], aps)}
            now = {it["id"]: it for it in kp.apply_overrides(base, ovs, aps)}
            for id_, b in base_by.items():
                it = now.get(id_, b)
                props = {k: v for k, v in it.items() if k not in ("at", "facing")}
                props["base_at"], props["base_facing"] = b["at"], b["facing"]
                label = _label(it) if layer == "ours" else it["module"]
                if id_ in by_ov:
                    props["override"] = by_ov[id_]
                if id_ not in now:
                    props["removed"] = True
                    label += " (weg)"
                out.append(_obj(id_, layer, it["module"], label, it["at"], g.at(*it["at"]), it["facing"], True,
                                props))
        return out, msgs

    def known_ids(self):
        """The ids an override may name (before overrides)."""
        import kunstwerk_placement as kp
        base, ds = self._bases()
        return {it["id"] for it in kp.apply_overrides(base, []) + kp.apply_overrides(ds, [])}

    # -- what the model drew (read only)
    def furniture(self):
        g = self.ground()
        out = []
        for i, f in enumerate(_load(os.path.join(self.cfg.ref, "furniture.json"), []) or []):
            h = float(f.get("heading", 0.0))
            out.append(_obj(f"mesh:{i}", "furniture", "mesh", f"#{i}", f["at"], g.at(*f["at"]),
                            [np.cos(h), np.sin(h)], False, {"verts": f.get("verts")}))
        return out

    def heads(self):
        mesh = _mesh_head(self.cfg.mesh_path)
        out = []
        for i, s in enumerate(mesh.get("signals", [])):
            out.append(_obj(f"head:{i}", "heads", "signal_head", "kop", s["at"][:2], s["at"][2], s.get("facing"),
                            False, {"z_head": s["at"][2]}))
        return out

    # -- references
    def ndw(self):
        feats = _load(os.path.join(self.cfg.ref, "ndw", "signs.json"), []) or []
        if isinstance(feats, dict):
            feats = feats.get("features", [])
        fr, g, out = self.frame(), self.ground(), []
        for i, f in enumerate(feats):
            pr = f.get("properties", {})
            lon, lat = f["geometry"]["coordinates"][:2]
            x, y = fr.from_wgs(lat, lon)
            fid = f.get("id") or pr.get("id") or i
            out.append(_obj(f"ndw:{fid}", "ndw", "sign", pr.get("rvvCode") or "?", (x, y), g.at(x, y),
                            facing_from_bearing(pr.get("bearing")) if pr.get("bearing") is not None else None,
                            False, pr))
        return out

    def osm_signals(self):
        els = (_load(os.path.join(self.cfg.ref, "osm_signals.json"), {}) or {}).get("elements", [])
        fr, g, out = self.frame(), self.ground(), []
        for e in els:
            if e.get("type") != "node" or "lat" not in e:
                continue
            t = e.get("tags", {})
            x, y = fr.from_wgs(e["lat"], e["lon"])
            label = t.get("highway") or t.get("crossing") or t.get("railway") or "node"
            if t.get("crossing") and t.get("highway") == "crossing":
                label = "crossing:" + t["crossing"]
            out.append(_obj(f"osm:{e['id']}", "osm_signals", label, label, (x, y), g.at(x, y), None, False, t))
        return out

    def _read_osm(self, path):
        if not os.path.exists(path):
            return []
        root = ET.parse(path).getroot()
        nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.findall("node")}
        fr, out = self.frame(), []
        r = self.refjson().get("box") or [-200, -200, 200, 200]
        for w in root.findall("way"):
            t = {x.get("k"): x.get("v") for x in w.findall("tag")}
            if t.get("highway") not in OSM_ROADS and "railway" not in t:
                continue
            pts = [fr.from_wgs(*nodes[nd.get("ref")]) for nd in w.findall("nd") if nd.get("ref") in nodes]
            if len(pts) < 2 or not any(r[0] - 50 <= x <= r[2] + 50 and r[1] - 50 <= y <= r[3] + 50 for x, y in pts):
                continue
            out.append({"id": f"way:{w.get('id')}", "kind": t.get("highway") or t.get("railway"),
                        "pts": [[round(x, 2), round(y, 2)] for x, y in pts], "tags": t})
        return out

    def osm_ways(self):
        return self._osm.get(os.path.join(self.cfg.ref, "osm_map.xml"))

    def approach_objs(self):
        g, out = self.ground(), []
        for i, a in enumerate(self.approaches()):
            l, r = np.asarray(a["left"], float), np.asarray(a["right"], float)
            d = (r - l) / max(np.hypot(*(r - l)), 1e-9)
            mid = (l + r) / 2
            out.append(_obj(f"approach:{i}", "approaches", "approach", f"A{i}", mid, g.at(*mid),
                            [-d[1], d[0]], False,                 # (the way in: the line's right turned left)
                            {"name": a.get("name", ""), "left": a["left"], "right": a["right"], "index": i}))
        return out

    # -- Street View diagnosis (tools/streetview scanner's output; the folder may be absent)
    def sv_places(self):
        root = self.cfg.sv_root
        if not os.path.isdir(root):
            return []
        return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    def sv_dir(self, place):
        if not place or any(c in place for c in "/\\:") or place.startswith("."):
            raise ValueError("bad place")
        return os.path.join(self.cfg.sv_root, place)

    def sv(self, place):
        d = self.sv_dir(place)
        out = {"place": place, "present": os.path.isdir(d), "objects": [], "shots": []}
        if not out["present"]:
            return out
        fr, g = self.frame(), self.ground()
        ours = {o["props"].get("src_ref"): o for o in self.ours()[0] if o["props"].get("src_ref")}
        gj = _load(os.path.join(d, "objects.geojson"), {}) or {}
        for i, f in enumerate(gj.get("features", [])):
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            pr = dict(f.get("properties") or {})
            lon, lat = geom["coordinates"][:2]
            x, y = fr.from_wgs(lat, lon)
            fac = pr.get("facing")
            if isinstance(fac, (int, float)):
                facing = [np.sin(np.radians(fac)), np.cos(np.radians(fac))]
            elif isinstance(fac, (list, tuple)) and len(fac) >= 2:
                facing = fac[:2]
            else:
                facing = None
            z = g.at(x, y)
            o = _obj(f"sv:{pr.get('id', i)}", "sv", pr.get("cls") or "object",
                     pr.get("sign_code") or pr.get("cls") or "?", (x, y), z, facing, False, pr)
            o["verdict"] = pr.get("verdict") or "ok"
            m = pr.get("match") or {}
            o["match_id"] = match_id(m, ours)
            if m.get("lat") is not None and m.get("lon") is not None:
                o["match_at"] = [round(v, 3) for v in fr.from_wgs(m["lat"], m["lon"])]
            elif o["match_id"] is None and m.get("source") == "placement":    # (one a fix took away: where it was)
                raw = self._raw_at(str(m.get("ref")))
                if raw is not None:
                    o["match_at"] = raw
            out["objects"].append(o)
        shots = _load(os.path.join(d, "shots.json"), []) or []
        for s in shots:
            x, y = fr.from_wgs(s["lat"], s["lon"])
            sid = str(s["id"])
            out["shots"].append({"id": sid, "at": [round(x, 3), round(y, 3)], "z": round(g.at(x, y) + 2.5, 2),
                                 "heading": float(s.get("heading", 0)), "pitch": float(s.get("pitch", 0)),
                                 "fov": float(s.get("fov", 90)), "w": int(s.get("w", 640)), "h": int(s.get("h", 640)),
                                 "png": os.path.exists(os.path.join(d, "shots", sid + ".png")),
                                 "annotated": os.path.exists(os.path.join(d, "shots_annotated", sid + ".png"))})
        return out

    def _raw_at(self, ref):
        """The spot of placement.json's items[N] / direction_signs.json's signs[N] as the file has it, or None."""
        import re
        m = re.fullmatch(r"(items|direction_signs)\[(\d+)\]", ref or "")
        if not m:
            return None
        fn, key = ("placement.json", "items") if m.group(1) == "items" else ("direction_signs.json", "signs")
        lst = (_load(os.path.join(self.cfg.ref, fn), {}) or {}).get(key, [])
        i = int(m.group(2))
        return [float(v) for v in lst[i]["at"][:2]] if i < len(lst) else None

    def shot_path(self, place, sid, annotated=False):
        if not sid or any(c in sid for c in "/\\:") or sid.startswith("."):
            raise ValueError("bad shot id")
        return os.path.join(self.sv_dir(place), "shots_annotated" if annotated else "shots", sid + ".png")

    # -- everything for the page
    def objects(self):
        ours, msgs = self.ours()
        layers = {"ours": [], "direction": []}
        for o in ours:
            layers[o["layer"]].append(o)
        layers.update(furniture=self.furniture(), heads=self.heads(), ndw=self.ndw(),
                      osm_signals=self.osm_signals(), approaches=self.approach_objs())
        return {"layers": layers, "messages": msgs, "approaches": self.approaches(),
                "origin_rd": list(self.frame().origin)}


def match_id(m, ours):
    """The id of the object a Street View detection matched ({"source", "ref"}: placement items[N] /
    direction_signs[N] (indices into placement.json / direction_signs.json), osm node/N, ndw <guid>), or None.
    ours: our objects by their src_ref."""
    src, ref = m.get("source"), m.get("ref")
    if ref is None:
        return None
    ref = str(ref)
    if src == "placement" or ref in ours:
        o = ours.get(ref)
        return o["id"] if o else None
    if src == "osm":
        return "osm:" + ref.split("/")[-1]
    if src == "ndw":
        return "ndw:" + ref
    return ref


def _mesh_head(path):
    from . import mesh
    return mesh.head(path)
