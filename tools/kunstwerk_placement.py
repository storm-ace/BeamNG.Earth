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
ALONG_ROAD = 0.5        # cos: a road's axis this near the register's quarter gives the sign's exact facing
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
        code = pr.get("rvvCode") or ""
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
        signs.append((p, unit(face), code))
    posts = []
    for p, face, code in signs:
        for post in posts:
            if np.hypot(*(post["p"] - p)) < POST_M and post["face"] @ face > 0.7:
                if code not in post["codes"]:
                    post["codes"].append(code)
                break
        else:
            posts.append({"p": p, "face": face, "codes": [code]})
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
                      "facing": [round(float(v), 4) for v in post["face"]]})
        n_posts += 1
    return n_posts, n_light, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    a = ap.parse_args(argv)
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
                            add({"module": "traffic_light", "role": "primary", "at": q, "facing": -h, "bike": True})
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
    main()
