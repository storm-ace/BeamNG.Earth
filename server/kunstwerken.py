"""Hand-built structures ("kunstwerken": a junction's decks over a sunken road, its retaining walls), modelled in Blender
on the AHN and the BGT (tools/blender_ref.py, tools/kunstwerk_walls.py; design/kunstwerken/<name>/) and placed in the
road tiles instead of what the generator makes of them (--nl-bgt on). The generator's surfaces stay; a kunstwerk adds
its meshes to the tile's colliding mesh, gives the streets over its decks their decks' planes (not the lidar's surface,
cars and lamp posts on it: holes and spikes, 2026-10-21) and keeps the BGT's own walls out of its box.

    for kw in near(frame, box):                    # the kunstwerken over a road tile's box
        mesh = kw.tile_mesh(frame, box, origin)    # (positions, tris, uvs, materials, tri material index) or None
    deck_top(lat, lon)                              # heights of the kunstwerken's decks there, NaN elsewhere

A kunstwerk file (data/kunstwerken/<name>.kunstwerk.json): origin_rd, positions (k, 3) in RD metres from it and NAP,
tris, materials (the game's), tri_material, decks [{ring (k, 2) RD metres from the origin, plane [a, b, c]: z = a + b x
+ c y}].
"""
import glob
import json
import logging
import math
import os
import threading
import time

import numpy as np

log = logging.getLogger("earthd.kunstwerken")

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "kunstwerken")
UV_M = 4.0                  # m per texture repeat on a kunstwerk's faces (concrete)
BOX_PAD_M = 2.0             # m round a kunstwerk's footprint its box reaches (the BGT's walls kept out of it)
# m round its floor its footprint reaches: what stands there of ours (the BGT's walls, props, lamps, utility wires, the
# AI's switchable signal heads) is left out, the kunstwerk has its own. The floor, not the box: the area is a rotated
# rectangle, its RD box took the generator's furniture out of the corners round it too (2026-10-22)
FOOTPRINT_PAD_M = 2.0
# m round a point the terrain under a kunstwerk takes the lowest floor of (cap): the terrain is a lattice of points 2 m
# apart, and between a point on a trench's floor and one on the street over it its triangle ran across the wall and
# stood in the trench as teeth along both walls, closing the passages under the decks (in game 2026-10-22). Kept under
# the lowest floor within a lattice's diagonal, that slope lies under the kunstwerk's street (its columns are solid)
CAP_REACH_M = 3
# m round a kunstwerk's rails the ground under it is bare (no grass of the game's ground cover): the terrain lies just
# under the tram's grass bed, and its grass stood through the rails (in game 2026-10-22: "de tram rails gaat door het
# gras heen")
RAIL_BARE_M = 2
RAIL_MATS = ("bne_rail_steel", "bne_steel_black")

_lock = threading.Lock()
_all = None


def _grow(mask, r):
    """mask with every cell within r cells of a set one set too (a disc)."""
    out = mask.copy()
    ny, nx = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r:
                out[max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)] |= \
                    mask[max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)]
    return out


def _lowest(z, r):
    """z (NaN: none) with every finite cell the lowest finite value within r cells of it (a disc); NaN stays NaN."""
    out = z.copy()
    ny, nx = z.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r:
                dst = out[max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)]
                np.fmin(dst, z[max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)], out=dst)
    out[~np.isfinite(z)] = np.nan
    return out


def _inside(ring, px, py):
    x, y = ring[:, 0], ring[:, 1]
    ins = np.zeros(len(px), bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        c = ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        ins ^= c
        j = i
    return ins


class Kunstwerk:
    def __init__(self, data):
        self.name = data.get("name", "?")
        self.origin = np.asarray(data["origin_rd"], np.float64)
        self.P = np.asarray(data["positions"], np.float64)
        self.T = np.asarray(data["tris"], np.int64)
        self.mats = list(data["materials"])
        self.tm = np.asarray(data["tri_material"], np.int64)
        self.decks = [(np.asarray(d["ring"], np.float64), np.asarray(d["plane"], np.float64)) for d in data.get("decks", ())]
        self.UV = np.asarray(data["uvs"], np.float64) if "uvs" in data else None
        # its traffic lights' heads (tools/kunstwerk_export.py): the middle of each, RD metres from the origin and NAP,
        # and the way it looks; the game switches them (hide_heads)
        self.signals = [(np.asarray(s["at"], np.float64), np.asarray(s["facing"], np.float64))
                        for s in data.get("signals", ())]
        self.wires = [np.asarray(w, np.float64) for w in data.get("wires", ()) if len(w) >= 2]    # (its overhead line)
        fl = data.get("floor")
        if fl:
            self.floor = np.asarray(fl["z"], np.float64).reshape(fl["ny"], fl["nx"])
            self.floor[self.floor < -900] = np.nan
            self.fx0, self.fy0 = float(fl["x0"]), float(fl["y0"])
            self.foot = _grow(np.isfinite(self.floor), int(round(FOOTPRINT_PAD_M)))     # (1 m cells)
            self.low = _lowest(self.floor, CAP_REACH_M)
            rail = np.zeros(self.floor.shape, bool)
            mi = [i for i, m in enumerate(self.mats) if m in RAIL_MATS]
            if mi:
                t = self.P[self.T[np.isin(self.tm, mi)]][:, :, :2]         # (k, 3, 2): along each edge every 0.5 m
                v = np.concatenate([a + (b - a) * s for a, b in ((t[:, 0], t[:, 1]), (t[:, 1], t[:, 2]),
                                                                 (t[:, 2], t[:, 0]))
                                    for s in np.linspace(0, 1, int(np.ceil(np.hypot(*(b - a).T).max() / 0.5)) + 1)
                                    [:, None, None]])
                v = v.reshape(-1, 2)
                i = np.floor(v[:, 0] - self.fx0).astype(np.int64)
                j = np.floor(v[:, 1] - self.fy0).astype(np.int64)
                k = (i >= 0) & (i < rail.shape[1]) & (j >= 0) & (j < rail.shape[0])
                rail[j[k], i[k]] = True
            self.bare = _grow(rail, RAIL_BARE_M)
        else:
            self.floor = self.foot = self.low = self.bare = None
        self.lo = self.P[:, :2].min(0) - BOX_PAD_M
        self.hi = self.P[:, :2].max(0) + BOX_PAD_M
        self._enu = {}

    def _rd(self, lat, lon):
        import ahn
        x, y = ahn.wgs_to_rd(lat, lon)
        return np.asarray(x) - self.origin[0], np.asarray(y) - self.origin[1]

    def enu(self, frame):
        """Its positions in the frame (cached per frame)."""
        key = (round(frame.lat0, 9), round(frame.lon0, 9)) if hasattr(frame, "lat0") else id(frame)
        got = self._enu.get(key)
        if got is None:
            import ahn
            la, lo = ahn.rd_to_wgs(self.P[:, 0] + self.origin[0], self.P[:, 1] + self.origin[1])
            e, n, u = frame.geodetic_to_enu(la, lo, self.P[:, 2])
            got = self._enu[key] = np.c_[np.asarray(e), np.asarray(n), np.asarray(u)]
        return got

    def heads(self, frame):
        """Its signal heads in the frame: [(x, y, z, facing x, facing y)]."""
        if not self.signals:
            return []
        import ahn
        at = np.array([s[0] for s in self.signals])
        f = np.array([s[1] for s in self.signals])
        f = f / np.maximum(np.hypot(f[:, 0], f[:, 1]), 1e-9)[:, None]
        la, lo = ahn.rd_to_wgs(at[:, 0] + self.origin[0], at[:, 1] + self.origin[1])
        e, n, u = frame.geodetic_to_enu(la, lo, at[:, 2])
        la2, lo2 = ahn.rd_to_wgs(at[:, 0] + f[:, 0] + self.origin[0], at[:, 1] + f[:, 1] + self.origin[1])
        e2, n2, _ = frame.geodetic_to_enu(la2, lo2, at[:, 2])
        de, dn = np.asarray(e2) - np.asarray(e), np.asarray(n2) - np.asarray(n)
        h = np.maximum(np.hypot(de, dn), 1e-9)
        return [(float(a), float(b), float(c), float(x), float(y))
                for a, b, c, x, y in zip(np.atleast_1d(e), np.atleast_1d(n), np.atleast_1d(u), de / h, dn / h)]

    def tile_wires(self, frame, box):
        """Its overhead wires' pieces whose segments' middles lie in the box (x0, y0, x1, y1), as the frame's
        polylines (k, 3) (the road tiles' utility wires: furniture.wire_mesh, render-only); each segment in one tile."""
        if not self.wires:
            return []
        import ahn
        x0, y0, x1, y1 = box
        out = []
        for w in self.wires:
            la, lo = ahn.rd_to_wgs(w[:, 0] + self.origin[0], w[:, 1] + self.origin[1])
            e, n, u = frame.geodetic_to_enu(la, lo, w[:, 2])
            p = np.c_[np.atleast_1d(e), np.atleast_1d(n), np.atleast_1d(u)]
            mid = (p[:-1] + p[1:]) / 2
            ins = (mid[:, 0] >= x0) & (mid[:, 0] < x1) & (mid[:, 1] >= y0) & (mid[:, 1] < y1)
            i = 0
            while i < len(ins):
                if not ins[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(ins) and ins[j + 1]:
                    j += 1
                out.append(p[i:j + 2])
                i = j + 1
        return out

    def over(self, frame, box):
        """Does it reach into the box (x0, y0, x1, y1) in the frame?"""
        V = self.enu(frame)
        x0, y0, x1, y1 = box
        return bool(((V[:, 0] >= x0 - BOX_PAD_M) & (V[:, 0] <= x1 + BOX_PAD_M) & (V[:, 1] >= y0 - BOX_PAD_M)
                     & (V[:, 1] <= y1 + BOX_PAD_M)).any())

    def tile_mesh(self, frame, box, origin):
        """Its triangles whose centre lies in the box, as a mesh relative to origin, or None."""
        V = self.enu(frame)
        c = V[self.T].mean(1)
        x0, y0, x1, y1 = box
        keep = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
        if not keep.any():
            return None
        T = self.T[keep]
        used = np.unique(T)
        remap = np.full(len(V), -1, np.int64)
        remap[used] = np.arange(len(used))
        P = V[used]
        uv = self.UV[used] if self.UV is not None else np.c_[(P[:, 0] + P[:, 1]) / UV_M, P[:, 2] / UV_M]
        return (P - np.asarray(origin, np.float64), remap[T], uv, list(self.mats), self.tm[keep])

    def in_box(self, lat, lon):
        """Which points lie in its footprint's box."""
        x, y = self._rd(lat, lon)
        return (x >= self.lo[0]) & (x <= self.hi[0]) & (y >= self.lo[1]) & (y <= self.hi[1])

    def _cells(self, lat, lon, grid, fill):
        x, y = self._rd(lat, lon)
        x, y = np.ravel(x), np.ravel(y)
        out = np.full(x.shape, fill, grid.dtype if grid is not None else type(fill))
        if grid is None:
            return out
        ny, nx = grid.shape
        i = np.floor(x - self.fx0).astype(np.int64)
        j = np.floor(y - self.fy0).astype(np.int64)
        k = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        out[k] = grid[j[k], i[k]]
        return out

    def floor_at(self, lat, lon):
        """Its floor's height (the lowest road surface it draws) per point, NaN where it draws none."""
        return self._cells(lat, lon, self.floor, np.nan)

    def low_at(self, lat, lon):
        """The lowest floor within CAP_REACH_M of each point (where it draws a floor), NaN elsewhere."""
        return self._cells(lat, lon, self.low, np.nan)

    def in_footprint(self, lat, lon):
        """Which points lie on its floor or within FOOTPRINT_PAD_M of it."""
        return self._cells(lat, lon, self.foot, False)

    def bare_at(self, lat, lon):
        """Which points lie within RAIL_BARE_M of its rails."""
        return self._cells(lat, lon, self.bare, False)

    def deck_top(self, lat, lon):
        x, y = self._rd(lat, lon)
        x, y = np.ravel(x), np.ravel(y)
        out = np.full(x.shape, np.nan)
        for ring, (a, b, c) in self.decks:
            lo_, hi_ = ring.min(0), ring.max(0)
            m = np.isnan(out) & (x >= lo_[0]) & (x <= hi_[0]) & (y >= lo_[1]) & (y <= hi_[1])
            if m.any():
                k = np.flatnonzero(m)
                k = k[_inside(ring, x[k], y[k])]
                out[k] = a + b * x[k] + c * y[k]
        return out


RELOAD_ENV = "BNE_KUNSTWERK_RELOAD"     # set (1): a kunstwerk file written anew is read again (tools/dashboard)
RELOAD_EVERY_S = 1.0
_stamp, _checked = None, 0.0


def _files_stamp():
    return tuple((p, os.stat(p).st_mtime_ns) for p in sorted(glob.glob(os.path.join(DIR, "*.kunstwerk.json"))))


def _reload_due():
    """With RELOAD_ENV set: has a kunstwerk file changed since they were read (checked once a RELOAD_EVERY_S)? A file
    still being written (not JSON yet) keeps the old ones until it is whole."""
    global _stamp, _checked
    if not os.environ.get(RELOAD_ENV) or _all is None or time.monotonic() - _checked < RELOAD_EVERY_S:
        return False
    _checked = time.monotonic()
    try:
        st = _files_stamp()
        if st == _stamp:
            return False
        for p, _ in st:
            with open(p, encoding="utf-8") as f:
                json.load(f)
    except (OSError, ValueError):
        return False
    log.info("kunstwerken changed on disk: read again")
    return True


def all_kunstwerken():
    global _all, _stamp
    with _lock:
        if _reload_due():
            _all = None
        if _all is None:
            _stamp = _files_stamp() if os.environ.get(RELOAD_ENV) else None
            _all = []
            for path in sorted(glob.glob(os.path.join(DIR, "*.kunstwerk.json"))):
                try:
                    with open(path, encoding="utf-8") as f:
                        _all.append(Kunstwerk(json.load(f)))
                except (OSError, ValueError, KeyError) as e:
                    log.warning("kunstwerk %s: %s: %s", os.path.basename(path), type(e).__name__, e)
        return _all


def near(frame, box):
    """The kunstwerken reaching into the box (x0, y0, x1, y1) in the frame."""
    return [k for k in all_kunstwerken() if k.over(frame, box)]


def deck_top(lat, lon, fallback=None):
    """The kunstwerken's deck heights at the points, else fallback(lat, lon) (the AHN's deck surface), else NaN."""
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    shape = np.broadcast(lat, lon).shape
    la, lo = np.broadcast_to(lat, shape).ravel(), np.broadcast_to(lon, shape).ravel()
    out = np.full(la.shape, np.nan)
    for k in all_kunstwerken():
        b = k.in_box(la, lo)
        if b.any():
            got = k.deck_top(la[b], lo[b])
            idx = np.flatnonzero(b)
            fill = np.isnan(out[idx]) & np.isfinite(got)
            out[idx[fill]] = got[fill]
    if fallback is not None:
        rest = np.isnan(out)                    # (a deck the kunstwerk has not modelled: the lidar's, as elsewhere)
        if rest.any():
            out[rest] = np.asarray(fallback(la[rest], lo[rest]), np.float64)
    return out.reshape(shape)


def covered(lat, lon):
    """Which points the kunstwerken draw a street on (their floor)."""
    lat, lon = np.ravel(np.asarray(lat, np.float64)), np.ravel(np.asarray(lon, np.float64))
    out = np.zeros(lat.shape, bool)
    for k in all_kunstwerken():
        b = k.in_box(lat, lon)
        if b.any():
            idx = np.flatnonzero(b)
            out[idx] |= np.isfinite(k.floor_at(lat[idx], lon[idx]))
    return out


def bare(lat, lon):
    """Which points the ground under a kunstwerk is bare at (round its rails: no ground cover through them)."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    shape = lat.shape
    lat, lon = lat.ravel(), lon.ravel()
    out = np.zeros(lat.shape, bool)
    for k in all_kunstwerken():
        b = k.in_box(lat, lon)
        if b.any():
            idx = np.flatnonzero(b)
            out[idx] |= k.bare_at(lat[idx], lon[idx])
    return out.reshape(shape)


def in_footprint(lat, lon, kws=None):
    """Which points lie in a kunstwerk's footprint (its floor, FOOTPRINT_PAD_M round it); kws: the ones to ask (all)."""
    lat, lon = np.ravel(np.asarray(lat, np.float64)), np.ravel(np.asarray(lon, np.float64))
    out = np.zeros(lat.shape, bool)
    for k in all_kunstwerken() if kws is None else kws:
        b = k.in_box(lat, lon)
        if b.any():
            idx = np.flatnonzero(b)
            out[idx] |= k.in_footprint(lat[idx], lon[idx])
    return out


def keep_out(frame, xy, kws=None):
    """in_footprint of the frame's points xy (k, 2)."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, bool)
    la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], np.zeros(len(xy)))
    return in_footprint(la, lo, kws)


HEAD_SHAPE = "/levels/beamng_earth/art/shapes/beamngEarth/bne_nl_signal.cdae"     # tools/nl_signal_head.py
HEAD_REACH_M = 30.0         # m from an approach's stop point a kunstwerk's head for it may stand
HEAD_FACING = 0.6           # ... looking at its traffic within ~53 degrees


def hide_heads(frame, signals):
    """navgraph's signals in a kunstwerk's footprint (their stop points): each approach's switchable head is the
    kunstwerk's own head that looks at its traffic nearest its stop point (its pole carries no head of its own: the
    game puts HEAD_SHAPE there and switches it, navgraph.lua), else none (the stock boom stood in its lanes,
    2026-10-22); the phases hold the AI either way. Its lights stood dead before: their heads were part of its mesh."""
    spots = [(i, j) for i, s in enumerate(signals) for j, a in enumerate(s["app"])]
    if not spots or not all_kunstwerken():
        return signals
    hit = keep_out(frame, [signals[i]["app"][j][1:3] for i, j in spots])
    heads = [h for k in all_kunstwerken() for h in k.heads(frame)]
    used = set()
    for (i, j), h in zip(spots, hit):
        if not h:
            continue
        a = signals[i]["app"][j]
        q, inc = np.asarray(a[1:3], np.float64), np.asarray(a[4:6], np.float64)
        best = None
        for n, (x, y, z, fx, fy) in enumerate(heads):
            d = float(np.hypot(x - q[0], y - q[1]))
            if -(fx * inc[0] + fy * inc[1]) >= HEAD_FACING and d <= HEAD_REACH_M:
                key = (n in used, d)            # (a head not yet given to another approach first)
                if best is None or key < best[0]:
                    best = (key, n)
        if best is None:
            a[7:11] = [None] * 4
            continue
        n = best[1]
        used.add(n)
        x, y, z, fx, fy = heads[n]
        # (navgraph.lua turns the head by -yaw: the yaw that takes its local +x to (fy, -fx) faces its +y along f)
        a[7:11] = [round(x, 2), round(y, 2), round(z, 2), round(math.degrees(math.atan2(-fx, fy)) % 360.0, 1)]
        a[11:] = [HEAD_SHAPE]
    return signals


CAP_M = 0.15                # m the terrain is kept under a kunstwerk's floor


def cap(ground):
    """ground(lat, lon) -> heights, CAP_M under the kunstwerken's floors where they draw one on the ground (not the
    lidar's bumps through them, nor the grading's cuts under them), and no higher than that under their decks."""
    if not all_kunstwerken():
        return ground

    def capped(lat, lon):
        z = np.asarray(ground(lat, lon), np.float64)
        shape = z.shape
        la, lo = np.ravel(np.asarray(lat, np.float64)), np.ravel(np.asarray(lon, np.float64))
        z = z.reshape(-1).copy()
        for k in all_kunstwerken():
            b = k.in_box(la, lo)
            if b.any():
                idx = np.flatnonzero(b)
                f = k.low_at(la[idx], lo[idx])     # (the lowest floor near: CAP_REACH_M)
                m = np.isfinite(f)
                dk = np.isfinite(k.deck_top(la[idx], lo[idx]))
                at = m & ~dk                    # on the ground: just under it (the grading's raise or cut undone)
                z[idx[at]] = f[at] - CAP_M
                un = m & dk                     # under a deck: no higher than under its slab
                z[idx[un]] = np.minimum(z[idx[un]], f[un] - CAP_M)
        return z.reshape(shape)
    return capped
