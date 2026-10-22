#!/usr/bin/env python3
"""Where a kunstwerk's street furniture stands, as the rules for Dutch junctions put it (not on the map's points, which
sit on the road's middle): ref/placement.json, for tools/kunstwerk_props.py's modules.

- a traffic light: at each stop line (a solid bar in the paint, ref/markings.json, square to the lane dashes, near an
  OSM signal: a signalled one). Its traffic comes from its side away from the junction (the signals' middle round it).
  The primary stands at the line's right end, just off the carriageway (LIGHT_OFF_M) and PAST_LINE_M past the line,
  facing that traffic; where a median (at most ISLAND_MAX_M wide) bounds the line's left end, a repeater stands on it.
- an L2 (zebra): at OSM's zebra, on the right of each way's traffic, just off the carriageway, facing that traffic.
- a D2 (keep right): on each island at a junction, at its nose toward the traffic (its point farthest from the
  junction), facing that traffic.
- with NDW's sign register (ref/ndw/signs.json, tools/ndw_signs.py) instead of the last two: every sign where it
  stands (ndw_signs).

    python3 tools/kunstwerk_placement.py ../design/kunstwerken/droogleever/ref
    python3 tools/kunstwerk_placement.py ../design/kunstwerken/droogleever/ref --check \\
        [--kunstwerk server/data/kunstwerken/droogleever.kunstwerk.json [--graph]]

Writes ref/placement.json: {"items": [{"module": "traffic_light" | "sign", "code": ..., "role": ..., "at": [x, y],
"facing": [fx, fy]}]}; a light's "bike" when a cycle path runs beside it.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))

JUNCTION_M = 30.0       # m round a signal the signals whose middle is its junction's
AXIS_REACH_M = 14.0     # m round a signal whose lane dashes give the road's direction
LIGHT_OFF_M = 0.8       # m a light stands off its carriageway's edge
ISLAND_MAX_M = 6.0      # m: a median at most this wide carries a repeater
REACH_M = 25.0          # m a signal's carriageway edge is looked for
DEDUP_M = 2.5           # m: two lights nearer than this are one
STOP_W = (0.2, 0.6)     # m: a stop line's width (as the photo draws it)
STOP_MIN_M = 2.4        # m: and its length at least (one lane)
STOP_NEAR_SIGNAL_M = 25.0  # m: a stop line this near an OSM signal is a signalled one
HAND_REACH_M = 3.0      # m: a hand-read stop line's end is this near its kerb
PAST_LINE_M = 0.8       # m a primary light stands past its stop line
OSM_MATCH_M = 15.0      # m: a primary this near an OSM signal, facing its way's traffic, is that signal's
MEDIAN_MIN_M = 1.5      # m: a strip off the road this wide on the lanes' left (a median, an island, the tram's grass)
                        # carries their mast, its arm over them to the right (Street View A269, B20)
FOOT_R = 0.25           # m round a pole's foot that must all be ground it may stand on (not road, cycle path, tram)
SETTLE_M = 3.0          # m a pole is moved to find such ground, else it is left out
MAST_LANES = 3         # lanes in (OSM's lanes) from which the primary is a mast with an arm over them
OSM_CARS = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}


def unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.hypot(*v), 1e-9)


POST_M = 0.8            # m: NDW's signs this near each other (and looking the same way) are one post's plates
ON_LIGHT_M = 1.5        # m: a sign this near a light (looking its way) is a plate on the light's pole
# cos: a road's axis this near the register's quarter gives the sign's exact facing: within 45 degrees, the quarter's
# own reach (at 0.5, 60 degrees, two C4 and a B6 were turned square to a road beside the one they speak to: --check,
# 2026-10-22)
ALONG_ROAD = 0.7
# a post's plates from the top down: priority, then bans and orders, then the rest, the sub-signs (OB) last
PLATE_ORDER = "BCFADEGLJ"


def osm_ways(path, local):
    """OSM's roads and paths at grade (osm_map.xml) as segments (a (k, 2), b (k, 2)) in the kunstwerk's metres."""
    import xml.etree.ElementTree as ET
    if not os.path.exists(path):
        return np.zeros((0, 2)), np.zeros((0, 2))
    root = ET.parse(path).getroot()
    nodes = {n.get("id"): n for n in root.findall("node")}
    A, B = [], []
    for w in root.findall("way"):
        t = {x.get("k"): x.get("v") for x in w.findall("tag")}
        if "highway" not in t or t.get("layer", "0") != "0" or t.get("tunnel") not in (None, "no"):
            continue
        pts = [local({"lat": float(nodes[r].get("lat")), "lon": float(nodes[r].get("lon"))})
               for r in (nd.get("ref") for nd in w.findall("nd")) if r in nodes]
        A += pts[:-1]
        B += pts[1:]
    return np.array(A).reshape(-1, 2), np.array(B).reshape(-1, 2)


OSM_AXIS_M = 12.0       # m: an OSM way this near a sign gives its road's direction


def speed_code(pr):
    """A sign's code from NDW's record: its rvvCode, an A1 with its limit ("A1_50": the atlas's cell with that number,
    tools/sign_atlas.py SPEEDS) where the register gives one (blackCode, else a number in its textSigns); the plain A1
    (Commons' 50) else. Every A1 plate showed that 50 (the user 2026-10-22)."""
    import re
    import kunstwerk_props as kit
    code = pr.get("rvvCode") or ""
    if code != "A1":
        return code
    texts = [pr.get("blackCode")]
    for t in pr.get("textSigns") or ():
        texts += list(t.values()) if isinstance(t, dict) else [t]
    for t in texts:
        m = re.fullmatch(r"\s*(\d{1,3})\s*(km/?h)?\s*", str(t)) if t is not None else None
        if m and f"A1_{int(m.group(1))}" in kit.PLATES:
            return f"A1_{int(m.group(1))}"
    return code


def ndw_signs(feats, items, local, inside_box, road_axis, ways=(np.zeros((0, 2)), np.zeros((0, 2)))):
    """NDW's signs (GeoJSON features) as items: beside the road (placement L), with a code the kit draws; their facing
    against the register's bearing (the way the traffic they speak to drives: toward the junction for a give-way
    sign, 2026-10-22), set square to the road there where its lanes say which way it runs, else where OSM's nearest way
    does (the register's quarter alone turned every sign to a compass point). Signs at one spot one post,
    a sign by a light a plate on its pole. Returns (posts, plates on lights, left out)."""
    import kunstwerk_props as kit
    WA, WB = ways

    def osm_axis(p, face):
        """The direction (an axis) of the OSM way nearest p within OSM_AXIS_M among those running along face's
        quarter (a sign speaks to a road it stands square to), or None."""
        if not len(WA):
            return None
        d = WB - WA
        L2 = np.maximum((d * d).sum(1), 1e-9)
        t = np.clip(((p - WA) * d).sum(1) / L2, 0, 1)
        dist = np.hypot(*(WA + d * t[:, None] - p).T)
        u = d / np.sqrt(L2)[:, None]
        ok = (dist < OSM_AXIS_M) & (np.abs(u @ face) >= ALONG_ROAD)
        return u[np.flatnonzero(ok)[np.argmin(dist[ok])]] if ok.any() else None
    signs, skipped = [], 0
    for f in feats:
        pr = f["properties"]
        code = speed_code(pr)
        p = local({"lat": f["geometry"]["coordinates"][1], "lon": f["geometry"]["coordinates"][0]})
        if code not in kit.PLATES or pr.get("placement") != "L" or not inside_box(p):
            skipped += 1
            continue
        b = math.radians(float(pr.get("bearing") or 0.0))
        face = -np.array([math.sin(b), math.cos(b)])          # (RD: x east, y north)
        ax = road_axis(p)
        if ax is None or abs(float(ax @ face)) < ALONG_ROAD:
            ax = osm_axis(p, face)                            # (no lane dashes there: the nearest OSM way's)
        if ax is not None and abs(float(ax @ face)) >= ALONG_ROAD:
            face = ax if ax @ face > 0 else -ax
        signs.append((p, unit(face), code, float(pr.get("bearing") or 0.0)))
    posts = []
    for p, face, code, bearing in signs:
        for post in posts:
            if np.hypot(*(post["p"] - p)) < POST_M and post["face"] @ face > 0.7:
                if code not in post["codes"]:
                    post["codes"].append(code)
                break
        else:
            # a sign back to back with another (NDW lists each face: an E1 for each way of a street) stands on that
            # one's post, its plate on the other side: on two posts 0.2 m apart the plates stood one in the other
            # (--check, 2026-10-22)
            for post in posts:
                if np.hypot(*(post["p"] - p)) < POST_M and post["face"] @ face < -0.7:
                    p = post["p"]
                    break
            posts.append({"p": p, "face": face, "codes": [code], "bearing": bearing})
    n_posts = n_light = 0
    for post in posts:
        codes = sorted(post["codes"], key=lambda c: (PLATE_ORDER.find(c[0]), c))
        light = min((it for it in items if it["module"] == "traffic_light" and not it.get("mast")),
                    key=lambda it: np.hypot(*(np.asarray(it["at"]) - post["p"])), default=None)
        if light is not None and np.hypot(*(np.asarray(light["at"]) - post["p"])) < ON_LIGHT_M \
                and np.asarray(light["facing"]) @ post["face"] > 0.5:
            light.setdefault("plates", [])
            light["plates"] += [c for c in codes if c not in light["plates"]]
            n_light += len(codes)
            continue
        items.append({"module": "sign", "code": codes if len(codes) > 1 else codes[0], "role": "ndw",
                      "at": [round(float(v), 3) for v in post["p"]],
                      "facing": [round(float(v), 4) for v in post["face"]],
                      "bearing": post["bearing"]})        # (the register's: --check compares it with the road)
        n_posts += 1
    return n_posts, n_light, skipped


def load_items(ref_dir):
    """ref/placement.json's items with ref/placement_fix.json's corrections applied (read off Street View: each matched
    to the item of its module nearest its "near", within 2 m; "add" ones appended), as tools/kunstwerk_solid.py puts
    them. Returns (items, the fixes that matched nothing)."""
    pl_path = os.path.join(ref_dir, "placement.json")
    items = json.load(open(pl_path))["items"] if os.path.exists(pl_path) else []
    fx_path = os.path.join(ref_dir, "placement_fix.json")
    missed = []
    for fx in (json.load(open(fx_path))["fixes"] if os.path.exists(fx_path) else ()):
        if fx["action"] == "add":
            items.append({k: v for k, v in fx.items() if k not in ("action", "why")})
            continue
        near = np.asarray(fx["near"], np.float64)
        cand = [i for i, it in enumerate(items) if it["module"] == fx["module"]
                and np.hypot(*(np.asarray(it["at"]) - near)) < 2.0]
        if not cand:
            missed.append(fx)
            continue
        i = min(cand, key=lambda i: np.hypot(*(np.asarray(items[i]["at"]) - near)))
        if fx["action"] == "remove":
            items.pop(i)
        else:
            items[i].update({k: v for k, v in fx.items() if k not in ("action", "why", "near")})
    return items, missed


# --check: which way each placed sign and head looks, against the traffic of the road it stands at (the user,
# 2026-10-22: signs and lights placed for the wrong approach, mixed up, facing the wrong traffic)
CHECK_WAY_M = 15.0      # m: the OSM ways this near an item are the ones it may speak to
CHECK_AWAY = 60.0       # degrees: a face turned more than this from its traffic's oncoming is turned away
CHECK_NDW = 45.0        # degrees: NDW's bearing (a compass quarter) this far off the road's traffic disagrees
CHECK_POLE_M = 0.5      # m: heads this near each other are one pole's
CHECK_DOUBLE_M = 1.0    # m: one code twice this near (on two posts, looking one way) is one sign drawn twice
CHECK_DEG_M = 0.15      # m a degree of a way's traffic off an item's face counts as (its road: near, and square to it)
AGAINST = {"C2"}        # signs that face the traffic a one-way road forbids (no entry: at its far end)


def osm_traffic(path, local):
    """OSM's ways at grade (osm_map.xml) for --check: [(way id, tags, a (k, 2), b (k, 2), directions)] with each
    segment's travel directions as the way's oneway allows (1: a -> b, -1: b -> a), and its signal nodes
    [(node id, point, tags, the ways in: [(way id, travel unit into the node)])]."""
    import xml.etree.ElementTree as ET
    if not os.path.exists(path):
        return [], []
    root = ET.parse(path).getroot()
    nodes = {n.get("id"): n for n in root.findall("node")}
    xy = {}

    def at(nid):
        if nid not in xy:
            xy[nid] = local({"lat": float(nodes[nid].get("lat")), "lon": float(nodes[nid].get("lon"))})
        return xy[nid]
    ways, by_node = [], {}
    for w in root.findall("way"):
        t = {x.get("k"): x.get("v") for x in w.findall("tag")}
        hw = t.get("highway")
        if hw not in OSM_CARS and hw not in ("cycleway", "busway", "bus_guideway") \
                or t.get("layer", "0") != "0" or t.get("tunnel") not in (None, "no"):
            continue
        refs = [r.get("ref") for r in w.findall("nd") if r.get("ref") in nodes]
        if len(refs) < 2:
            continue
        P = np.array([at(r) for r in refs])
        ow = t.get("oneway")
        dirs = (1,) if ow in ("yes", "1", "true") or t.get("junction") == "roundabout" else (-1,) if ow == "-1" \
            else (1, -1)
        ways.append((w.get("id"), t, P[:-1], P[1:], dirs))
        for i, r in enumerate(refs):
            by_node.setdefault(r, []).append((w.get("id"), t, refs, i, dirs))
    sigs = []
    for nid, nd in nodes.items():
        nt = {x.get("k"): x.get("v") for x in nd.findall("tag")}
        if nt.get("highway") != "traffic_signals" and nt.get("crossing") != "traffic_signals":
            continue
        sd = nt.get("traffic_signals:direction")
        ins = []
        for wid, t, refs, i, dirs in by_node.get(nid, ()):
            p = at(nid)
            if i > 0 and 1 in dirs and sd in (None, "forward", "both"):
                ins.append((wid, unit(p - at(refs[i - 1]))))
            if i < len(refs) - 1 and -1 in dirs and sd in (None, "backward", "both"):
                ins.append((wid, unit(p - at(refs[i + 1]))))
        sigs.append((nid, at(nid), nt, ins))
    return ways, sigs


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(unit(u), unit(v))), -1.0, 1.0))))


def check_items(items, ways, sigs):
    """The --check table's rows: per item {"i", "module", "code", "at", "way", "travel", "face_err", "approach",
    "flags"}: the OSM way it stands at (the nearest within CHECK_WAY_M whose traffic it may face: a light's or a
    sign's face looks at its traffic), that traffic's travel direction, the angle between its face and the oncoming
    of that traffic, a light's approach (the OSM signal within OSM_MATCH_M whose way in it looks at); flags: "away"
    (its face turned more than CHECK_AWAY from every traffic near), "no signal" (a head whose approach has no signal),
    "2 heads" (two heads for one approach on one pole), "ndw" (NDW's bearing more than CHECK_NDW off the road's
    traffic), "double" (one code twice within CHECK_DOUBLE_M on two posts). Pure."""
    rows = []
    for i, it in enumerate(items):
        p = np.asarray(it["at"], np.float64)[:2]
        f = unit(np.asarray(it["facing"], np.float64)[:2])
        code = it.get("code", it.get("role", ""))
        code = code if isinstance(code, str) else "+".join(code)
        if it["module"] == "traffic_light" and it.get("plates"):
            code = "light+" + "+".join(it["plates"])
        elif it["module"] in ("traffic_light", "head"):
            code = it.get("code", "head") if it["module"] == "head" else "light"
        against = bool(set(code.split("+")) & AGAINST)
        best = None
        for wid, t, A, B, dirs in ways:
            d = B - A
            L2 = np.maximum((d * d).sum(1), 1e-9)
            s = np.clip(((p - A) * d).sum(1) / L2, 0, 1)
            dist = np.hypot(*(A + d * s[:, None] - p).T)
            for k in np.flatnonzero(dist < CHECK_WAY_M):
                for sg in (dirs if not against else (1, -1)):
                    tr = unit(d[k] * sg)
                    err = _angle(f, -tr)
                    key = (err > CHECK_AWAY, float(dist[k]) + CHECK_DEG_M * err, err)
                    if best is None or key < best[0]:
                        best = (key, wid, t, tr, err)
        row = {"i": i, "module": it["module"], "code": code, "at": [round(float(v), 1) for v in p], "facing": f,
               "way": None, "travel": None, "face_err": None, "approach": None, "flags": []}
        if best is None:
            row["flags"].append("no road")
        else:
            _, wid, t, tr, err = best
            row.update(way=f"{t.get('name', '?')} ({t.get('highway')} w{wid})", travel=[round(float(v), 2) for v in tr],
                       face_err=round(err))
            if err > CHECK_AWAY:
                row["flags"].append("away")
        if it["module"] in ("traffic_light", "head"):
            app = None
            for nid, q, nt, ins in sigs:
                dd = float(np.hypot(*(q - p)))
                if dd > OSM_MATCH_M:
                    continue
                for wid, h in ins:
                    if float(f @ -h) > 0.7 and (app is None or dd < app[0]):
                        app = (dd, f"n{nid}/w{wid}")
            if app is None:
                row["flags"].append("no signal")
            else:
                row["approach"] = app[1]
        if it.get("bearing") is not None and best is not None:
            b = np.radians(float(it["bearing"]))
            bt = np.array([np.sin(b), np.cos(b)])           # (the way its traffic drives: RD x east, y north)
            ang = min(_angle(bt, unit((B_ - A_)[k] * sg)) for wid_, t_, A_, B_, dirs_ in ways if wid_ == best[1]
                      for k in range(len(A_)) for sg in (dirs_ if not against else (1, -1)))
            row["ndw_err"] = round(ang)
            if ang > CHECK_NDW:
                row["flags"].append("ndw")
        rows.append(row)
    for a in rows:
        for b in rows:
            if b["i"] <= a["i"]:
                continue
            d = float(np.hypot(a["at"][0] - b["at"][0], a["at"][1] - b["at"][1]))
            if a["module"] in ("traffic_light", "head") and b["module"] == a["module"] and d < CHECK_POLE_M \
                    and a["approach"] and a["approach"] == b["approach"] and a["code"] == b["code"]:
                b["flags"].append("2 heads")
            if a["module"] == b["module"] == "sign" and d < CHECK_DOUBLE_M \
                    and set(a["code"].split("+")) & set(b["code"].split("+")) and a["facing"] @ b["facing"] > 0.7:
                b["flags"].append(f"double of {a['i']}")
    return rows


def print_check(rows, title):
    print(f"\n{title}: {len(rows)} items, {sum(1 for r in rows if r['flags'])} flagged")
    print(f"{'#':>3} {'item':<14} {'at':>13}  {'way':<46} {'travel':>13} {'face':>5} {'ndw':>4}  {'approach':<26} "
          f"{'ai':<5} flags")
    for r in rows:
        tr = "" if r["travel"] is None else f"({r['travel'][0]:+.2f},{r['travel'][1]:+.2f})"
        fe = "" if r["face_err"] is None else f"{r['face_err']}"
        print(f"{r['i']:>3} {r['code'][:14]:<14} {r['at'][0]:>6.1f},{r['at'][1]:>6.1f}  {str(r['way'])[:46]:<46} "
              f"{tr:>13} {fe:>5} {str(r.get('ndw_err', '')):>4}  {str(r['approach'] or '')[:26]:<26} "
              f"{r.get('ai', ''):<5} "
              f"{', '.join(r['flags'])}")


def check_main(ref_dir, kunstwerk=None, graph=False):
    """--check: the table for ref/placement.json (+ placement_fix.json) and, given the kunstwerk's server file, for
    its switchable heads as shipped (and, --graph, which of them the AI graph's approaches take: the map cache)."""
    import ahn as ahnmod
    ref = json.load(open(os.path.join(ref_dir, "ref.json"), encoding="utf-8"))

    def local(e):
        rx, ry = ahnmod.wgs_to_rd(e["lat"], e["lon"])
        return np.array([rx - ref["origin_rd"][0], ry - ref["origin_rd"][1]])
    ways, sigs = osm_traffic(os.path.join(ref_dir, "osm_map.xml"), local)
    items, missed = load_items(ref_dir)
    for fx in missed:
        print(f"placement fix matched nothing near {fx['near']}: {fx.get('why', '')}")
    rows = check_items(items, ways, sigs)
    print_check(rows, os.path.join(ref_dir, "placement.json"))
    out = {"placement": rows}
    if kunstwerk:
        kw = json.load(open(kunstwerk, encoding="utf-8"))
        heads = [{"module": "head", "at": s["at"][:2], "facing": s["facing"],
                  "code": "bike head" if s.get("bike") else "cycle head" if s.get("cycle") else "head"}
                 for s in kw.get("signals", ())]
        hrows = check_items(heads, ways, sigs)
        if graph:
            for r, got in zip(hrows, graph_heads(kunstwerk)):
                if not got:
                    r["flags"].append("no AI approach")
                else:
                    r["ai"] = got
        print_check(hrows, kunstwerk + " (its switchable heads)")
        out["heads"] = hrows
    return out


def graph_heads(kunstwerk_path):
    """Per head of the kunstwerk: how the AI graph round it (built offline from the map cache, the mapped controls as
    the server's; server kunstwerken.hide_heads) shows it: "app" (a graph's signalled approach's first head), "app+"
    (one more head of one: a repeater, a cyclists' head beside it), "made" (an approach made on its road), "" (none)."""
    import copy
    import geodesy
    import mapcells
    import mapdata
    import navgraph
    import netfetch
    import furniture
    import kunstwerken
    kw = kunstwerken.Kunstwerk(json.load(open(kunstwerk_path, encoding="utf-8")))
    saved, kunstwerken._all = kunstwerken._all, [kw]
    try:
        import ahn
        la0, lo0 = ahn.rd_to_wgs(*kw.origin)
        frame = geodesy.EnuFrame(float(la0), float(lo0))
        md = mapdata.MapData(mapdata.DEFAULT_CACHE, netfetch.Fetcher(offline=True))
        w = navgraph.WINDOW_M + 200.0
        la, lo, _ = frame.enu_to_geodetic(np.array([-w, w]), np.array([-w, w]), 0.0)
        keys = mapcells.cells_for_bbox(float(la.min()), float(lo.min()), float(la.max()), float(lo.max()))
        feats = [md.cell(k, layers=("roads",), kind="demand").layers.get("roads") for k in keys]
        mapped = furniture.Mapped.load(md, frame, (-w, -w, w, w))

        def elev(a, b):         # (the kunstwerk's decks and floor, the streets round it at 3 m: heads over their road)
            z = kw.deck_top(a, b)
            f = kw.floor_at(a, b)
            return np.where(np.isfinite(z), z, np.where(np.isfinite(f), f, 3.0)).reshape(np.shape(a))
        g = navgraph.build(frame, feats, elev, (0.0, 0.0), lambda a, b: np.ones(np.shape(a), bool), mapped=mapped)
        before = {id(a) for s in g["signals"] for a in s["app"]}
        sig = kunstwerken.hide_heads(frame, copy.copy(g["signals"]), g, mapped)
        got = []
        for s in sig:
            for a in s["app"]:
                kind = "app" if id(a) in before else "made"
                for n, hh in enumerate([a[7:12]] + (a[12] if len(a) > 12 else [])):
                    if hh[0] is not None:
                        got.append((hh[0], hh[1], kind if n == 0 or kind == "made" else "app+"))
        return [next((k for gx, gy, k in got if abs(gx - x) < 0.05 and abs(gy - y) < 0.05), "")
                for x, y, *_ in kw.heads(frame)]
    finally:
        kunstwerken._all = saved


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    ap.add_argument("--check", action="store_true", help="write nothing: a table of every placed sign and head, the "
                    "road it stands at, the angle of its face to that traffic's oncoming, its approach and what is off")
    ap.add_argument("--kunstwerk", help="with --check: the server's <name>.kunstwerk.json, its heads as shipped")
    ap.add_argument("--graph", action="store_true", help="with --kunstwerk: which approach of the AI graph (built "
                    "offline from the map cache) shows each head")
    a = ap.parse_args(argv)
    if a.check:
        rows = check_main(a.ref, a.kunstwerk, a.graph)
        return 1 if any(r["flags"] for rs in rows.values() for r in rs) else 0
    import ahn as ahnmod
    import kunstwerk_solid as ks
    from kunstwerk_design import carriageways
    ref = json.load(open(os.path.join(a.ref, "ref.json"), encoding="utf-8"))
    ks.REF_DIR[:] = [a.ref]
    frame, _ = carriageways(a.ref)
    c0, u, n = frame
    corners = np.array([c0 + u * t + n * w for t in ks.AREA_T for w in (-ks.AREA_W, ks.AREA_W)])
    box = (*np.floor(corners.min(0)), *np.ceil(corners.max(0)))
    plan = ks.Plan(ref, frame, box)
    names = [k[0] for k in ks.KINDS]
    ROAD, CYCLE, ISLAND = names.index("road"), names.index("cycle"), names.index("island")
    TRAM = names.index("tram")
    osm = json.load(open(os.path.join(a.ref, "osm_signals.json")))["elements"]

    def local(e):
        rx, ry = ahnmod.wgs_to_rd(e["lat"], e["lon"])
        return np.array([rx - ref["origin_rd"][0], ry - ref["origin_rd"][1]])

    def inside_box(p):
        return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]
    # the lane dashes: the marks longer than wide by far, their axes
    dashes, bars, solid = [], [], []
    for m in json.load(open(os.path.join(a.ref, "markings.json")))["marks"]:
        r = np.asarray(m["rings"][0], np.float64)
        if len(r) != 4:
            continue
        e1, e2 = r[1] - r[0], r[2] - r[1]
        L, W = max(np.hypot(*e1), np.hypot(*e2)), min(np.hypot(*e1), np.hypot(*e2))
        if W <= 0.2 and 1.5 <= L <= 6.0:
            dashes.append((r.mean(0), unit(e1 if np.hypot(*e1) > np.hypot(*e2) else e2)))
        elif 0.35 <= W <= 1.0 and 2.0 <= L <= 6.0:           # a zebra's bar: along the road too
            bars.append((r.mean(0), unit(e1 if np.hypot(*e1) > np.hypot(*e2) else e2)))
        if STOP_W[0] <= W <= STOP_W[1] and L >= STOP_MIN_M:  # maybe a stop line (if it runs across the road)
            ax_ = unit(e1 if np.hypot(*e1) > np.hypot(*e2) else e2)
            c_ = r.mean(0)
            solid.append((c_ - ax_ * L / 2, c_ + ax_ * L / 2))
    dpos = np.array([d[0] for d in dashes]) if dashes else np.zeros((0, 2))
    ddir = np.array([d[1] for d in dashes]) if dashes else np.zeros((0, 2))

    bpos = np.array([b[0] for b in bars]) if bars else np.zeros((0, 2))
    bdir = np.array([b[1] for b in bars]) if bars else np.zeros((0, 2))

    def road_axis(p, zebra=False):
        """The road's direction at p (an axis: sign arbitrary): its lane dashes' mean (at a zebra its bars'), or
        None."""
        pos, dirs, reach = (bpos, bdir, 6.0) if zebra else (dpos, ddir, AXIS_REACH_M)
        if not len(pos):
            return None
        near = np.hypot(*(pos - p).T) < reach
        if near.sum() < 2:
            return None if zebra else road_axis(p, zebra=True) if len(bpos) else None
        ang = np.arctan2(dirs[near, 1], dirs[near, 0])
        m = np.angle(np.mean(np.exp(2j * ang))) / 2      # (an axis: angles doubled, averaged, halved)
        return np.array([np.cos(m), np.sin(m)])

    def edge(p, d):
        """From p along d: the carriageway's edge (its last road point) and what lies beyond: (edge, kind beyond,
        width of that non-road strip (inf: no road again within ISLAND_MAX_M))."""
        s = np.arange(0.0, REACH_M, 0.2)
        ray = p + np.outer(s, d)
        k = plan.kind(*ray.T)
        off = np.flatnonzero(k != ROAD)
        if not len(off):
            return None, None, None
        i = off[0]
        back = np.flatnonzero(k[i:] == ROAD)
        width = s[i + back[0]] - s[i] if len(back) else np.inf
        return ray[max(i - 1, 0)], int(k[i]), width

    sigs = [local(e) for e in osm if e["tags"].get("highway") == "traffic_signals"
            or e["tags"].get("crossing") == "traffic_signals"]
    sigs = [p for p in sigs if inside_box(p)]
    # each signal's junction: the middle of the signals round it (chaining them ran round the whole ring and put the
    # middle in the oval, 2026-10-21)
    S = np.array(sigs) if sigs else np.zeros((0, 2))
    centre = {i: S[np.hypot(*(S - p).T) < JUNCTION_M].mean(0) for i, p in enumerate(sigs)}
    jid = list(range(len(sigs)))
    items, skipped = [], []

    def add(item):
        p = np.asarray(item["at"])
        if any(np.hypot(*(p - np.asarray(o["at"]))) < DEDUP_M and o["module"] == item["module"] for o in items):
            return
        item["at"] = [round(float(v), 3) for v in item["at"]]
        item["facing"] = [round(float(v), 4) for v in item["facing"]]
        items.append(item)
    # the ways in read off the photo by hand (ref/approaches.json: each stop line from its left end to its right end
    # as its drivers see it) where there are such; else found round each signalled junction (below)
    hand = os.path.join(a.ref, "approaches.json")
    if os.path.exists(hand):
        for ap_ in json.load(open(hand))["approaches"]:
            lft, rgt = np.asarray(ap_["left"], np.float64), np.asarray(ap_["right"], np.float64)
            right = unit(rgt - lft)
            h = np.array([-right[1], right[0]])            # the way in: the line's right turned a quarter left
            e, _, _ = edge(rgt - right * 0.8, right)
            if e is not None and np.hypot(*(e - rgt)) <= HAND_REACH_M:
                q = e + right * LIGHT_OFF_M + h * PAST_LINE_M
                near = plan.kind(*(q + 2.5 * np.c_[np.cos(np.linspace(0, 6.28, 12)),
                                                   np.sin(np.linspace(0, 6.28, 12))]).T)
                add({"module": "traffic_light", "role": "primary", "at": q, "facing": -h,
                     "bike": bool((near == CYCLE).any())})
            el, _, wl = edge(lft + right * 0.8, -right)
            if el is not None and np.hypot(*(el - lft)) <= HAND_REACH_M:     # (its kerb, not a road beyond)
                q2 = el - right * (min(LIGHT_OFF_M, wl / 2) if np.isfinite(wl) else LIGHT_OFF_M) + h * PAST_LINE_M
                if plan.kind(*q2[:, None])[0] != ROAD:
                    add({"module": "traffic_light", "role": "repeater", "at": q2, "facing": -h, "bike": False})
    # the approaches: round each signalled junction (its signals' middle) a circle at its signals' distance; every
    # carriageway crossing it is a mouth. Of two mouths with a median between them the one with the median on its
    # left is the way in (keeping right); a lone mouth's way in is its right half. A way in with an OSM signal on it
    # gets its lights: the primary on the kerb at its right, level with the signal (OSM puts it on the stop line),
    # facing the traffic; a repeater on the median at its left.
    juncs = []
    for c in ([] if os.path.exists(hand) else centre.values()):
        if all(np.hypot(*(c - d)) > 15 for d in juncs):
            juncs.append(c)
    ang = np.radians(np.arange(0, 360, 0.5))
    for C in juncs:
        mine = S[np.hypot(*(S - C).T) < JUNCTION_M]
        R = float(np.clip(np.median(np.hypot(*(mine - C).T)), 8.0, 25.0))
        ring = C + R * np.c_[np.cos(ang), np.sin(ang)]
        hit = plan.kind(*ring.T) == ROAD
        if hit.all() or not hit.any():
            continue
        k0 = int(np.argmin(hit))
        runs, cur = [], []
        for j in range(len(ang)):
            jj = (k0 + j) % len(ang)
            if hit[jj]:
                cur.append(jj)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        runs = [r for r in runs if np.hypot(*(ring[r[-1]] - ring[r[0]])) >= 2.5]
        nr = len(runs)
        for i, run in enumerate(runs):
            a0, a1 = ring[run[0]], ring[run[-1]]           # a1: the inbound drivers' right (counter-clockwise)
            w = float(np.hypot(*(a1 - a0)))
            nxt, prv = runs[(i + 1) % nr], runs[(i - 1) % nr]
            gap_prev = float(np.hypot(*(ring[prv[-1]] - a0))) if nr > 1 else np.inf
            gap_next = float(np.hypot(*(ring[nxt[0]] - a1))) if nr > 1 else np.inf
            if gap_next <= ISLAND_MAX_M and gap_prev > ISLAND_MAX_M:
                continue                                    # its median on its right: the way out
            if gap_prev <= ISLAND_MAX_M:
                lin, rin, median = a0, a1, True             # all of it in, the median on its left
            else:
                lin, rin, median = (a0 + a1) / 2, a1, False   # a two-way road: its right half in
            mid = (lin + rin) / 2
            h = unit(C - mid)                               # the way in
            right = np.array([h[1], -h[0]])
            # its signal: the OSM one nearest the way in's middle
            dsig = np.hypot(*(S - mid).T)
            if not len(S) or dsig.min() > 14.0:
                skipped.append(("no signal on a way in", mid))
                continue
            sig = S[np.argmin(dsig)]
            # the stop line: level with the signal along the way in
            along = float(np.dot(sig - mid, h))
            line_r = rin + h * along
            e, _, _ = edge(line_r - right * 1.0, right)
            if e is None:
                skipped.append(("no kerb", line_r))
                continue
            q = e + right * LIGHT_OFF_M + h * PAST_LINE_M
            if plan.kind(*q[:, None])[0] == ROAD:
                skipped.append(("kerb spot on the road", q))
                continue
            near = plan.kind(*(q + 2.5 * np.c_[np.cos(np.linspace(0, 6.28, 12)),
                                               np.sin(np.linspace(0, 6.28, 12))]).T)
            add({"module": "traffic_light", "role": "primary", "at": q, "facing": -h,
                 "bike": bool((near == CYCLE).any())})
            if median:
                el, _, wl = edge(lin + h * along + right * 1.0, -right)
                if el is not None and wl <= ISLAND_MAX_M:
                    q2 = el - right * min(LIGHT_OFF_M, wl / 2) + h * PAST_LINE_M
                    if plan.kind(*q2[:, None])[0] != ROAD:
                        add({"module": "traffic_light", "role": "repeater", "at": q2, "facing": -h, "bike": False})
    # OSM's ways through its signals (ref/osm_map.xml, the API's map call): each signal's way says which way its traffic
    # comes (traffic_signals:direction, else the way's oneway, else both) and how many lanes come in. A way in with no
    # primary yet facing its traffic gets one at its right kerb; a primary over three lanes or more becomes a mast with
    # an arm over them; a cycle path's signal gets its cyclists' light at the path's right
    osm_xml = os.path.join(a.ref, "osm_map.xml")
    n_osm = n_mast = 0
    if os.path.exists(osm_xml):
        import xml.etree.ElementTree as ET
        root = ET.parse(osm_xml).getroot()
        nodes = {n_.get("id"): n_ for n_ in root.findall("node")}
        ways = []
        for w_ in root.findall("way"):
            tags = {t.get("k"): t.get("v") for t in w_.findall("tag")}
            if tags.get("highway") in OSM_CARS or tags.get("highway") == "cycleway":
                ways.append(([nd.get("ref") for nd in w_.findall("nd")], tags))
        xy = {}

        def at(nid):
            if nid not in xy:
                xy[nid] = local({"lat": float(nodes[nid].get("lat")), "lon": float(nodes[nid].get("lon"))})
            return xy[nid]
        for nid, nd_ in nodes.items():
            ntags = {t.get("k"): t.get("v") for t in nd_.findall("tag")}
            if ntags.get("highway") != "traffic_signals" and ntags.get("crossing") != "traffic_signals":
                continue
            p = at(nid)
            if not inside_box(p):
                continue
            sd = ntags.get("traffic_signals:direction")
            for refs, wt in ways:
                if nid not in refs:
                    continue
                i = refs.index(nid)
                ow = wt.get("oneway")
                ins = []
                if i > 0 and (sd == "forward" or (sd is None and ow != "-1")):
                    ins.append(p - at(refs[i - 1]))
                if i < len(refs) - 1 and (sd == "backward" or (sd is None and ow not in ("yes", "1", "true"))):
                    ins.append(p - at(refs[i + 1]))
                try:
                    lanes = int(wt.get("lanes:forward") or (wt.get("lanes") if ow in ("yes", "1", "true")
                                                            else int(wt.get("lanes", "2")) // 2))
                except (TypeError, ValueError):
                    lanes = 1
                cyc = wt.get("highway") == "cycleway"
                for d_ in ins:
                    if np.hypot(*d_) < 0.5:
                        continue
                    h = unit(d_)                                # the way in
                    right = np.array([h[1], -h[0]])
                    mine = [o for o in items if o["module"] == "traffic_light" and o.get("role") == "primary"
                            and np.hypot(*(np.asarray(o["at"]) - p)) < OSM_MATCH_M
                            and np.dot(np.asarray(o["facing"]), -h) > 0.7]
                    if cyc:
                        if any(o.get("bike") and np.hypot(*(np.asarray(o["at"]) - p)) < 6.0 for o in items):
                            continue
                        s = np.arange(0.2, 4.0, 0.2)
                        off = np.flatnonzero(plan.kind(*(p + np.outer(s, right)).T) != CYCLE)
                        q = p + right * ((s[off[0]] if len(off) else 1.4) + 0.5) + h * PAST_LINE_M
                        if plan.kind(*q[:, None])[0] != ROAD:
                            add({"module": "traffic_light", "role": "primary", "at": q, "facing": -h, "bike": True,
                                 "cycle": True})     # (its cyclists' alone: no car stops at it)
                            n_osm += 1
                        continue
                    side = None
                    if lanes >= MAST_LANES:
                        # the mast on the side with the wider verge (the tram's grass, a median), its arm over the
                        # lanes to the other: A269 has it on the drivers' left, Street View 2026-10-21
                        er, kr_, wr = edge(p, right)
                        el, kl_, wl = edge(p, -right)
                        wr = 0.0 if er is None else (wr if np.isfinite(wr) else 8.0)
                        wl = 0.0 if el is None else (wl if np.isfinite(wl) else 8.0)
                        side = "left" if (el is not None and wl >= MEDIAN_MIN_M) else "right"
                    if mine:
                        if side:
                            for o in mine:
                                o["mast"] = lanes
                                o["arm"] = "right" if side == "left" else "left"
                                if side == "left":
                                    o["at"] = [round(float(v), 3) for v in el - right * LIGHT_OFF_M + h * PAST_LINE_M]
                        continue
                    e, _, _ = edge(p, right)
                    if e is None or np.hypot(*(e - p)) > 12.0:
                        continue
                    q = e + right * LIGHT_OFF_M + h * PAST_LINE_M
                    if plan.kind(*q[:, None])[0] == ROAD:
                        continue
                    near = plan.kind(*(q + 2.5 * np.c_[np.cos(np.linspace(0, 6.28, 12)),
                                                       np.sin(np.linspace(0, 6.28, 12))]).T)
                    item = {"module": "traffic_light", "role": "primary", "at": q, "facing": -h,
                            "bike": bool((near == CYCLE).any())}
                    if side:
                        item["mast"] = lanes
                        item["arm"] = "right" if side == "left" else "left"
                        if side == "left":
                            item["at"] = el - right * LIGHT_OFF_M + h * PAST_LINE_M
                    n0 = len(items)
                    add(item)
                    n_osm += len(items) - n0
        n_mast = sum(1 for o in items if o.get("mast"))
    print(f"OSM's ways: {n_osm} lights added, {n_mast} masts")
    n_l2 = 0
    # the signs where NDW's register has them (tools/ndw_signs.py: ref/ndw/signs.json): each where it stands, its code
    # and its plates; without the register the rules' zebra signs and keep-right noses below
    ndw_path = os.path.join(a.ref, "ndw", "signs.json")
    ndw = json.load(open(ndw_path, encoding="utf-8")) if os.path.exists(ndw_path) else None
    if ndw is not None:
        n_posts, n_on_lights, n_skipped = ndw_signs(ndw, items, local, inside_box, road_axis,
                                                    osm_ways(os.path.join(a.ref, "osm_map.xml"), local))
        print(f"NDW's signs: {n_posts} posts, {n_on_lights} plates on the lights' poles, {n_skipped} left out")
    for e_ in (osm if ndw is None else ()):
        if e_["tags"].get("crossing") != "zebra":
            continue
        p = local(e_)
        if not inside_box(p):
            continue
        ax = road_axis(p, zebra=True)
        if ax is None:
            ax = road_axis(p)
        if ax is None:
            continue
        for h in (ax, -ax):                                 # each way's traffic, its sign on its right
            right = np.array([h[1], -h[0]])
            e, _, _ = edge(p, right)
            q = None if e is None else e + right * LIGHT_OFF_M
            if q is not None and plan.kind(*q[:, None])[0] != ROAD and np.hypot(*(q - p)) < 15.0:
                add({"module": "sign", "code": "L2", "role": "zebra", "at": q, "facing": -h})
                n_l2 += 1
    # D2: each island near a junction, its nose toward the traffic
    from scipy import ndimage as ndi
    isl = np.argmax(plan.share, axis=0) == ISLAND
    lbl, nis = ndi.label(isl)
    n_d2 = 0
    cs = []
    for c in centre.values():                       # (the junctions, each once)
        if all(np.hypot(*(c - d)) > 15 for d in cs):
            cs.append(c)
    for i in range(1, nis + 1 if ndw is None else 1):
        rr, cc = np.nonzero(lbl == i)
        if len(rr) * ks.RES ** 2 < 4.0 or not cs:
            continue
        P = np.c_[plan.gx[cc], plan.gy[rr]]
        cen = min(cs, key=lambda c: np.hypot(*(c - P.mean(0))))
        if np.hypot(*(cen - P.mean(0))) > 60:
            continue
        nose = P[np.argmax(np.hypot(*(P - cen).T))]
        inward = unit(P.mean(0) - nose)
        q = nose + inward * 0.6
        add({"module": "sign", "code": "D2", "role": "island", "at": q, "facing": unit(nose - cen)})
        n_d2 += 1
    # every foot on ground a pole may stand on (in game a light stood in the lanes, poles on the cycle paths: the mast
    # moves above and the median's width took no account of what the strip was, 2026-10-22): moved to the nearest such
    # spot within SETTLE_M (across the way first, then along it), else left out
    FOOT = {names.index(k) for k in ("grass", "island", "foot", "platform")}
    ring_ = FOOT_R * np.c_[np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
                           np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False))]

    def stands(q, mast):
        k = plan.kind(*(np.r_[[q], q + ring_]).T)
        if all(int(v) in FOOT for v in k):
            return True
        # a mast on the tram's grass where it meets the carriageway (Rotterdam's: on the median's edge)
        return mast and all(int(v) in FOOT | {TRAM} for v in k)

    settled, moved, dropped = [], 0, []
    for it in items:
        q0 = np.asarray(it["at"], np.float64)
        mast = bool(it.get("mast"))
        if stands(q0, mast):
            settled.append(it)
            continue
        f_ = unit(np.asarray(it["facing"], np.float64))
        across = np.array([f_[1], -f_[0]])
        best = None
        for r in np.arange(0.25, SETTLE_M + 1e-9, 0.25):
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                d = np.array([np.cos(ang), np.sin(ang)])
                q = q0 + d * r
                if stands(q, mast):
                    cost = r * (1.0 + 0.5 * abs(float(np.dot(d, f_))))     # (across its traffic's way first)
                    if best is None or cost < best[0]:
                        best = (cost, q)
            if best is not None:
                break
        if best is None:
            dropped.append((it["module"], it.get("role"), [round(float(v), 1) for v in q0]))
            continue
        it["at"] = [round(float(v), 3) for v in best[1]]
        settled.append(it)
        moved += 1
    items[:] = settled
    print(f"feet settled: {moved} moved off the carriageway / cycle path / tram, {len(dropped)} left out {dropped}")
    out = os.path.join(a.ref, "placement.json")
    json.dump({"items": items, "junctions": [[round(float(v), 2) for v in c] for c in cs]}, open(out, "w"), indent=1)
    kinds = {}
    for it in items:
        key = it["module"] + ":" + it.get("role", "")
        kinds[key] = kinds.get(key, 0) + 1
    print(f"{out}: {kinds}; signals skipped: {len(skipped)} {[s[0] for s in skipped]}")


if __name__ == "__main__":
    sys.exit(main())
