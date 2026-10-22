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
- a signalled crossing's poles (osm_crossings, crossing_lights: where OSM's paths cross its car roads near a signal):
  one at each kerb, its pedestrians' head looking across the road, its cyclists' head at those arriving.

    python3 tools/kunstwerk_placement.py ../design/kunstwerken/droogleever/ref
    python3 tools/kunstwerk_placement.py ../design/kunstwerken/droogleever/ref --check \\
        [--kunstwerk server/data/kunstwerken/droogleever.kunstwerk.json [--graph]]

Writes ref/placement.json: {"items": [{"module": "traffic_light" | "crossing_light" | "sign", "code": ..., "role": ...,
"at": [x, y], "facing": [fx, fy]}]}; a light's "bike" when a cycle path runs beside it, a crossing pole's "heads".
--check adds the coverage: OSM's signal nodes with no head near, signalled crossings with heads on one side only.
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


# Signalled crossings (the user 2026-10-22, "we missen stoplichten": only the cars' heads stood). OSM maps the paths
# (footway, cycleway, path) and the car roads as ways; where a path crosses a car road at grade is a crossing. It is
# a signalled one when the crossing node says so (crossing=traffic_signals), when the path has its own signal near it
# (the cyclists' stop points round the Droogleever: highway=traffic_signals with traffic_signals:direction on the
# cycleway), or when a car signal stands within CROSS_JUNCTION_M: at a Dutch signalled junction every crossing of its
# arms is signalled (CROW; Street View P2_288 and P5_292: the crossings OSM tags "zebra" or "uncontrolled" there have
# their pedestrians' and cyclists' heads)
PATHS = {"footway", "cycleway", "path", "pedestrian", "bridleway"}
CROSS_JUNCTION_M = 25.0     # m: a crossing this near a car signal is that junction's, a signalled one (at 30 m the
                            # Baden Powelllaan's and the north side road's crossings, 26-29 m off, which Street View
                            # does not show)
CROSS_PATH_SIG_M = 15.0     # m: a path's own signal this near the crossing (along the path) makes it a signalled one
CROSS_MERGE_M = 6.0         # m: crossings of one road this near each other (a zebra beside its cycle crossing: the
                            # Westzeedijk's east arm 5.5 m, the 's-Gravendijkwal's south one 4.7 m) are one, its poles
                            # shared
CROSS_SAME_M = 1.5          # m: crossings of two roads this near (a road split at the crossing's node) are one
CROSS_NODE_M = 3.0          # m: a crossing node this near a path's crossing is that one's (else a crossing of its own)


def _xml_root(src):
    import xml.etree.ElementTree as ET
    return ET.fromstring(src) if src.lstrip().startswith("<") else ET.parse(src).getroot()


def osm_crossings(src, local):
    """The crossings of OSM's paths over its car roads at grade (osm_map.xml, or its text): [{"at": the crossing
    point, "road": the car road's axis (a unit), "path": the crossing users' axis (a unit), "kinds": ["bike"] and/or
    ["ped"] (a cycleway's cyclists, a footway's pedestrians; a crossing node alone: its pedestrians, and cyclists
    with bicycle=yes), "way": the car way's id, "name": its name, "node": the crossing node's id or None,
    "signalled": as above, "why": what says so}]."""
    root = _xml_root(src)
    nodes = {n.get("id"): n for n in root.findall("node")}
    xy, ntags = {}, {}

    def at(nid):
        if nid not in xy:
            xy[nid] = local({"lat": float(nodes[nid].get("lat")), "lon": float(nodes[nid].get("lon"))})
        return xy[nid]
    for nid, n in nodes.items():
        t = {x.get("k"): x.get("v") for x in n.findall("tag")}
        if t:
            ntags[nid] = t
    cars, paths = [], []
    for w in root.findall("way"):
        t = {x.get("k"): x.get("v") for x in w.findall("tag")}
        hw = t.get("highway")
        if t.get("layer", "0") != "0" or t.get("tunnel") not in (None, "no") or t.get("bridge") not in (None, "no"):
            continue
        refs = [r.get("ref") for r in w.findall("nd") if r.get("ref") in nodes]
        if len(refs) < 2:
            continue
        if hw in OSM_CARS:
            cars.append((w.get("id"), t, refs))
        elif hw in PATHS:
            paths.append((w.get("id"), t, refs))
    car_sigs = np.array([at(r) for _, _, refs in cars for r in refs
                         if ntags.get(r, {}).get("highway") == "traffic_signals"]).reshape(-1, 2)
    out = []
    for cid, ct, crefs in cars:
        C = np.array([at(r) for r in crefs])
        for pid, pt, prefs in paths:
            P = np.array([at(r) for r in prefs])
            kind = "bike" if pt.get("highway") == "cycleway" or pt.get("bicycle") == "designated" else "ped"
            for i in range(len(C) - 1):
                a, d = C[i], C[i + 1] - C[i]
                for j in range(len(P) - 1):
                    b, e = P[j], P[j + 1] - P[j]
                    den = d[0] * e[1] - d[1] * e[0]
                    if abs(den) < 1e-9:
                        continue
                    q = b - a
                    s = (q[0] * e[1] - q[1] * e[0]) / den
                    u = (q[0] * d[1] - q[1] * d[0]) / den
                    if not (-1e-6 <= s <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6):
                        continue
                    X = a + d * s
                    # the path's own signals near it, its crossing node
                    psig = [r for r in prefs if ntags.get(r, {}).get("highway") == "traffic_signals"
                            and np.hypot(*(at(r) - X)) <= CROSS_PATH_SIG_M]
                    cnode = next((r for r in crefs if ntags.get(r, {}).get("highway") == "crossing"
                                  and np.hypot(*(at(r) - X)) <= CROSS_NODE_M), None)
                    # (a crossing node: its pedestrians cross there too, a zebra's or a signalled one's)
                    kinds = [kind] + (["ped"] if cnode and ntags[cnode].get("crossing") not in (None, "no") else [])
                    out.append({"at": X, "road": unit(d), "path": unit(e), "kinds": kinds, "way": cid,
                                "name": ct.get("name", "?"), "node": cnode, "path_way": pid, "psig": psig})
    # crossing nodes on a car road with no path mapped over it (its sidewalks as the road's tags): across the road
    for cid, ct, crefs in cars:
        for i, r in enumerate(crefs):
            t = ntags.get(r, {})
            if t.get("highway") != "crossing" or t.get("crossing") == "no":
                continue
            if any(o["way"] == cid and np.hypot(*(o["at"] - at(r))) <= CROSS_NODE_M for o in out):
                continue
            d = unit(at(crefs[min(i + 1, len(crefs) - 1)]) - at(crefs[max(i - 1, 0)]))
            kinds = ["ped"] + (["bike"] if t.get("bicycle") in ("yes", "designated") else [])
            out.append({"at": at(r), "road": d, "path": np.array([-d[1], d[0]]), "kinds": kinds, "way": cid,
                        "name": ct.get("name", "?"), "node": r, "path_way": None, "psig": []})
    # one road's crossings this near each other: one crossing (a footway beside its cycle path)
    merged = []
    for c in out:
        for m in merged:
            dd = float(np.hypot(*(m["at"] - c["at"])))
            if (m["way"] == c["way"] and dd <= CROSS_MERGE_M) or dd <= CROSS_SAME_M:
                m["kinds"] = sorted(set(m["kinds"]) | set(c["kinds"]))
                m["psig"] = m["psig"] + c["psig"]
                m["node"] = m["node"] or c["node"]
                break
        else:
            merged.append(dict(c, kinds=sorted(c["kinds"])))
    for c in merged:
        t = ntags.get(c["node"], {}) if c["node"] else {}
        near = np.hypot(*(car_sigs - c["at"]).T) if len(car_sigs) else np.array([np.inf])
        if t.get("crossing") == "traffic_signals" or t.get("crossing:signals") == "yes":
            c["signalled"], c["why"] = True, f"crossing n{c['node']}: traffic_signals"
        elif c["psig"]:
            c["signalled"], c["why"] = True, "the path's signal n" + "/".join(sorted(set(c["psig"])))
        elif near.min() <= CROSS_JUNCTION_M:
            c["signalled"], c["why"] = True, f"a car signal {near.min():.0f} m off: the junction's"
        else:
            c["signalled"], c["why"] = False, ""
        del c["psig"]
    return merged


CROSS_REACH_M = 20.0        # m along the path from the crossing its carriageway's kerbs are looked for
CROSS_ON_ROAD_M = 4.0       # m: the carriageway must be this near the crossing point (OSM's line against the BGT's)
CROSS_POLE_OUT_M = 0.8      # m a crossing's pole stands back from its kerb
CROSS_POLE_SIDE_M = 0.6     # m and to the right of the users arriving there (the crossing's middle stays clear)
CROSS_HAS_BIKE_M = 6.0      # m: a cycle path's own light (its cyclists' head) this near an end serves its cyclists
                            # (the north arm's, read off Street View: 4.9 m from the crossing's kerb)


def crossing_lights(crossings, items, inside_box, edge_kind, ROAD, others=()):
    """A pole at each kerb of each signalled crossing (osm_crossings) inside the box: items "crossing_light", "facing"
    away from the road (toward those who arrive here: its buttons), its "heads": [{"kind": "ped" (the pedestrians'
    head, looking across the road at the far kerb's pedestrians: a Dutch pedestrians' head stands across the road from
    those it speaks to) or "bike" (the cyclists' head, looking at the cyclists arriving at this kerb: theirs stands at
    their stop line, nearside), "facing", "across": the axis of the road crossed (the game shows them the inverse of
    its traffic: server kunstwerken.hide_heads)}]. The kerbs: along the crossing users' axis the ends of the
    carriageway's run nearest the crossing point (edge_kind: the plan's kind of ground). A cycle path's own light near
    an end (OSM's cyclists' stop point: its cyclists' head is there; items, or others: placement_fix.json's) leaves out
    the cyclists' head there; two ends on one spot (a median) are one pole. Returns (poles, ends left out)."""
    s = np.arange(-CROSS_REACH_M, CROSS_REACH_M + 1e-9, 0.2)
    n0, skipped = len(items), []
    for c in crossings:
        if not c.get("signalled") or not inside_box(c["at"]):
            continue
        X, v = np.asarray(c["at"], np.float64), unit(c["path"])
        road = edge_kind(*(X + np.outer(s, v)).T) == ROAD
        runs = []
        k = 0
        while k < len(s):
            if road[k]:
                k1 = k
                while k1 + 1 < len(s) and road[k1 + 1]:
                    k1 += 1
                runs.append((s[k], s[k1]))
                k = k1 + 1
            else:
                k += 1
        runs = [r for r in runs if min(abs(r[0]), abs(r[1])) <= CROSS_ON_ROAD_M or r[0] <= 0 <= r[1]]
        if not runs:
            skipped.append((c["name"], [round(float(x), 1) for x in X], "no carriageway"))
            continue
        lo, hi = min(runs, key=lambda r: 0 if r[0] <= 0 <= r[1] else min(abs(r[0]), abs(r[1])))
        for t_end, sg in ((hi, 1.0), (lo, -1.0)):
            if abs(t_end) >= CROSS_REACH_M - 0.3:
                skipped.append((c["name"], [round(float(x), 1) for x in X], "no kerb"))
                continue
            o = v * sg                                          # away from the road: toward those who arrive here
            E = X + v * t_end
            right = np.array([-o[1], o[0]])                     # (the right of those walking -o, toward the road)
            q = E + o * CROSS_POLE_OUT_M + right * CROSS_POLE_SIDE_M
            if not inside_box(q):
                continue
            ped, bike = "ped" in c["kinds"], "bike" in c["kinds"]
            if bike and any(it["module"] == "traffic_light" and it.get("bike")
                            and np.hypot(*(np.asarray(it["at"]) - q)) < CROSS_HAS_BIKE_M
                            and float(np.dot(it["facing"], o)) > 0.5 for it in list(items) + list(others)):
                bike = False
            across = [round(float(x), 4) for x in unit(c["road"])]
            heads = ([{"kind": "ped", "facing": [round(float(x), 4) for x in -o], "across": across}] if ped else []) \
                + ([{"kind": "bike", "facing": [round(float(x), 4) for x in o], "across": across}] if bike else [])
            if not heads:
                continue
            # a pole near (the median between a road's two carriageways: each crossing's end on it): one pole, its
            # heads each way
            same = next((it for it in items[n0:] if np.hypot(*(np.asarray(it["at"]) - q)) < DEDUP_M), None)
            if same is not None:
                for h in heads:
                    if not any(g["kind"] == h["kind"] and np.dot(g["facing"], h["facing"]) > 0.7
                               for g in same["heads"]):
                        same["heads"].append(h)
                continue
            items.append({"module": "crossing_light", "role": "crossing", "at": [round(float(x), 3) for x in q],
                          "facing": [round(float(x), 4) for x in o], "heads": heads, "way": c["way"],
                          "node": c["node"], "why": c["why"]})
    return len(items) - n0, skipped


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
        if it["module"] == "crossing_light" or code.startswith("crossing"):
            # (a crossing's pole and heads look along its path, across the traffic: coverage() checks them)
            heads_ = it.get("heads") or ()
            rows.append({"i": i, "module": it["module"],
                         "code": code if heads_ == () else "crossing " + "+".join(h["kind"] for h in heads_),
                         "at": [round(float(v), 1) for v in p], "facing": f, "way": f"crossing w{it.get('way', '?')}",
                         "travel": None, "face_err": None, "approach": None, "flags": []})
            continue
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


COVER_M = 6.0           # m: an OSM signal node with no head this near is one we left out
CROSS_SIDE_W = 8.0      # m beside a crossing's line (along its road) its heads stand


def coverage(heads, sigs, crossings, inside, served=()):
    """--check's coverage (the user 2026-10-22: "we missen stoplichten"): the OSM signal nodes inside the kunstwerk
    (inside) with no head within COVER_M nor one whose approach it is (served: node ids, check_items' "approach";
    a car signal stands on the middle of its road, the head at its kerb: 7 to 12 m off on the Droogleever's four
    lanes) ([(node id, point, tags, distance to the nearest head)]), and its signalled
    crossings (osm_crossings) with heads on one side only or none ([(crossing, heads on its + side, on its - side)]:
    + is along its "path" axis; a head counts for a side within CROSS_SIDE_W of the crossing's line, 0.5 to
    CROSS_REACH_M out). heads: [(x, y)]. Pure."""
    H = np.asarray(heads, np.float64).reshape(-1, 2)
    lone = []
    for nid, q, nt, _ in sigs:
        if not inside(q):
            continue
        d = float(np.hypot(*(H - q).T).min()) if len(H) else np.inf
        if d > COVER_M and nid not in served:
            lone.append((nid, q, nt, d))
    sides = []
    for c in crossings:
        if not c.get("signalled") or not inside(c["at"]):
            continue
        r = H - np.asarray(c["at"], np.float64)
        out_, beside = r @ unit(c["path"]), np.abs(r @ unit(c["road"]))
        ok = beside <= CROSS_SIDE_W
        n_p = int((ok & (out_ >= 0.5) & (out_ <= CROSS_REACH_M)).sum())
        n_m = int((ok & (out_ <= -0.5) & (out_ >= -CROSS_REACH_M)).sum())
        if not (n_p and n_m):
            sides.append((c, n_p, n_m))
    return lone, sides


def print_coverage(lone, sides, title):
    print(f"\n{title}: coverage: {len(lone)} OSM signal nodes with no head within {COVER_M:.0f} m, {len(sides)} "
          f"signalled crossings with heads on one side only or none")
    for nid, q, nt, d in lone:
        what = nt.get("crossing") if nt.get("highway") == "crossing" else nt.get("traffic_signals:direction", "")
        print(f"  n{nid} at {q[0]:6.1f},{q[1]:6.1f}  {what or '':<10} nearest head {d:5.1f} m")
    for c, n_p, n_m in sides:
        print(f"  crossing of {c['name'][:24]} (w{c['way']}) at {c['at'][0]:6.1f},{c['at'][1]:6.1f} "
              f"{'+'.join(c['kinds']):<9} heads {n_p} / {n_m} ({c['why']})")


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
    # coverage: within the kunstwerk's area (kunstwerk_solid's: where its furniture stands)
    import kunstwerk_solid as ks
    from kunstwerk_design import carriageways
    ks.REF_DIR[:] = [ref_dir]
    (c0, u, n), _ = carriageways(ref_dir)

    def inside(q):
        d = np.asarray(q, np.float64) - c0
        return bool(ks.AREA_T[0] <= d @ u <= ks.AREA_T[1] and abs(d @ n) <= ks.AREA_W)
    crossings = osm_crossings(os.path.join(ref_dir, "osm_map.xml"), local)

    def cover(points, title, rows_):
        # (a signal is served by every head within OSM_MATCH_M that looks at one of its ways in, not only the head
        # whose nearest it is: the S junction's west approach's head is nearer the cycle path's signal beside it)
        served = set()
        for r in rows_:
            if r["module"] not in ("traffic_light", "head") or r["code"].startswith("crossing"):
                continue
            p, f = np.asarray(r["at"], np.float64), unit(r["facing"])
            served |= {nid for nid, q, _, ins in sigs if np.hypot(*(q - p)) <= OSM_MATCH_M
                       and any(float(f @ -h) > 0.7 for _, h in ins)}
        lone, sides = coverage(points, sigs, crossings, inside, served)
        print_coverage(lone, sides, title)
        return [{"flags": [f"n{nid}: no head within {COVER_M:.0f} m"]} for nid, *_ in lone] + \
            [{"flags": [f"crossing w{c['way']} at {c['at'][0]:.0f},{c['at'][1]:.0f}: heads {a_} / {b_}"]}
             for c, a_, b_ in sides]
    out["coverage"] = cover([it["at"][:2] for it in items if it["module"] in ("traffic_light", "crossing_light")],
                            os.path.join(ref_dir, "placement.json") + "'s poles", rows)
    if kunstwerk:
        kw = json.load(open(kunstwerk, encoding="utf-8"))
        heads = [{"module": "head", "at": s["at"][:2], "facing": s["facing"],
                  "code": "crossing " + s["crossing"] if s.get("crossing") else "bike head" if s.get("bike")
                  else "cycle head" if s.get("cycle") else "head"}
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
        out["coverage_heads"] = cover([s["at"][:2] for s in kw.get("signals", ())], kunstwerk + "'s heads", hrows)
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
                for n, hh in enumerate([a[7:12]] + (a[12] if len(a) > 12 and a[12] else [])):
                    if hh[0] is None:
                        continue
                    # ("inv": a crossing's head, the inverse of this approach's light)
                    got.append((hh[0], hh[1], "inv" if len(hh) > 5 and hh[5] == "inv"
                                else kind if n == 0 or kind == "made" else "app+"))
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
    fx_path = os.path.join(a.ref, "placement_fix.json")
    fix_adds = [fx for fx in (json.load(open(fx_path))["fixes"] if os.path.exists(fx_path) else ())
                if fx["action"] == "add"]
    n_x, x_skipped = crossing_lights(osm_crossings(osm_xml, local) if os.path.exists(osm_xml) else [], items,
                                     inside_box, plan.kind, ROAD, fix_adds)
    print(f"signalled crossings: {n_x} poles at their kerbs; ends left out: {x_skipped}")
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
