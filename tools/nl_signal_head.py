#!/usr/bin/env python3
"""The Dutch signal head the game switches (a TSStatic the traffic signals' controller drives, mod navgraph.lua): the
kunstwerken's head (tools/kunstwerk_props.py signal_head: a black stadium, its white rim, three bezels) with the stock
switchable boom's lenses in it. The game lights a lens by its vertex colours and UVs on the instanced_trafficlight
material (a palette the instance colour picks from), which is not documented: the lenses are the stock head's own
triangles (art/shapes/objects/s_trafficlight_boom_sn.dae, its pole head), moved and scaled into the three bezels.

    python3 tools/nl_signal_head.py [--game "<BeamNG.drive folder>"]

Writes mod/beamng_earth/levels/beamng_earth/art/shapes/beamngEarth/bne_nl_signal.cdae: its frame on the pole's axis at
the head's middle, its face toward +y (kunstwerk_props.head_mount: where it hangs on a pole); and
bne_nl_signal_bike.cdae, the cyclists' head under it (kunstwerk_props.bike_mount)."""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import cdae  # noqa: E402
import kunstwerk_props as kit  # noqa: E402
from kunstwerken import two_sided_apart  # noqa: E402

OUT = os.path.join(HERE, "..", "mod", "beamng_earth", "levels", "beamng_earth", "art", "shapes", "beamngEarth",
                   "bne_nl_signal.cdae")
STOCK = "art/shapes/objects/s_trafficlight_boom_sn.dae"
LENS_MAT = "instanced_trafficlight"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
GAME = r"C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive"


def stock_lenses(game):
    """The stock boom's pole head lenses (its finest detail): positions, normals, uvs, colours (RGBA 0..255) per
    triangle corner, with the three lenses' centres top to bottom."""
    with zipfile.ZipFile(os.path.join(game, "content", "art_shapes.zip")) as z:
        root = ET.fromstring(z.read(STOCK))
    src = {s.get("id"): np.array(s.find("c:float_array", NS).text.split(), float)
           for s in root.iter("{%s}source" % NS["c"]) if s.find("c:float_array", NS) is not None}
    def detail(g):                                      # ("..._L250": its detail size; colliders and others last)
        tail = (g.get("name") or "").rsplit("_L", 1)[-1]
        return -int(tail) if tail.isdigit() else 0
    geoms = sorted(root.iter("{%s}geometry" % NS["c"]), key=detail)
    m = geoms[0].find("c:mesh", NS)
    verts = {v.get("id"): v.find("c:input", NS).get("source")[1:] for v in m.findall("c:vertices", NS)}
    for tr in m.findall("c:triangles", NS):
        if tr.get("material") != LENS_MAT:
            continue
        ins = {i.get("semantic") + i.get("set", ""): (i.get("source")[1:], int(i.get("offset")))
               for i in tr.findall("c:input", NS)}
        stride = max(o for _, o in ins.values()) + 1
        p = np.array(tr.find("c:p", NS).text.split(), int).reshape(-1, stride)

        def get(key, dim):
            s, o = ins[key]
            s = verts.get(s, s)
            return src[s].reshape(-1, dim)[p[:, o]]
        P, N, UV = get("VERTEX", 3), get("NORMAL", 3), get("TEXCOORD0", 2)
        cs = ins["COLOR0"][0]
        C = get("COLOR0", 4 if len(src[cs]) % 4 == 0 and len(src[cs]) // 4 >= len(P) else 3)
        if C.shape[1] == 3:
            C = np.c_[C, np.ones(len(C))]
        tri = P.reshape(-1, 3, 3).mean(1)
        pole = tri[:, 0] < np.median(tri[:, 0])          # (the head on the pole, not the arm's)
        k = np.repeat(pole, 3)
        P, N, UV, C, tri = P[k], N[k], UV[k], C[k], tri[pole]
        zs = np.sort(tri[:, 2])
        gaps = np.argsort(np.diff(zs))[-2:]
        cuts = np.sort(zs[gaps] + np.diff(zs)[gaps] / 2)
        lens = np.digitize(tri[:, 2], cuts)             # 0 bottom .. 2 top
        centres = [tri[lens == i].mean(0) for i in (2, 1, 0)]
        return P, N, UV, np.clip(np.rint(C * 255), 0, 255).astype(np.uint8), centres
    raise SystemExit(f"{STOCK}: no {LENS_MAT} triangles")


def build(game, bike=False):
    """The head's mesh (positions, tris, uvs, normals, colours, tri materials, lens normals); bike: the cyclists' head
    lower on the pole (kunstwerk_props.bike_mount: its fixed lenses read as lit under the switchable head, the user
    2026-10-22), the same lenses at its scale."""
    # (its number plate is two-sided: the back on vertices of its own, else the engine's normals there are noise)
    P0, T0, UV0 = two_sided_apart(*kit.signal_head(lenses=False, bike=bike))
    P, N, UV, C, centres = stock_lenses(game)
    # the stock lenses face along their mean normal: turned to face +y, scaled to our lens pitch, centred in our bezels
    f = N.mean(0)
    f[2] = 0.0
    f /= np.linalg.norm(f)
    ang = np.arctan2(f[0], f[1])                        # rotation about z taking f to +y
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    if bike:
        yb, _, scale = kit.bike_mount()
    else:
        (yb, _), scale = kit.head_mount(), 1.0
    zs, lr = kit.head_lenses(scale)
    stock_pitch = (centres[0][2] - centres[2][2]) / 2
    s = (zs[0] - zs[2]) / 2 / stock_pitch
    mid = centres[1]
    Q = (P - mid) @ R.T * s
    Nq = N @ R.T
    front = yb + kit.HEAD[1] * scale + 0.006 * scale
    Q[:, 1] += front - Q[:, 1].max()                    # (just proud of the head's face)
    Q[:, 2] += zs[1]
    n0 = len(P0)
    positions = np.r_[P0, Q]
    tris = np.r_[T0, n0 + np.arange(len(Q)).reshape(-1, 3)]
    uvs = np.r_[UV0, UV]
    normals = None
    colors = np.r_[np.full((n0, 4), 255, np.uint8), C]
    tm = np.r_[np.zeros(len(T0), np.int64), np.ones(len(Q) // 3, np.int64)]
    what = "cyclists' head" if bike else "head"
    print(f"{what} {len(T0)} triangles, lenses {len(Q) // 3} (stock pitch "
          f"{stock_pitch:.3f} m, scale {s:.2f}, turned {np.degrees(ang):.0f} deg)")
    return positions, tris, uvs, normals, colors, tm, Nq


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", default=GAME, help="the BeamNG.drive folder (its content/art_shapes.zip)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    for bike, out in ((False, a.out), (True, a.out.replace(".cdae", "_bike.cdae"))):
        positions, tris, uvs, _, colors, tm, _ = build(a.game, bike)
        n = cdae.write(out, positions, tris, uvs=uvs, material=[kit.MATERIAL, LENS_MAT], tri_materials=tm,
                       colors=colors, object_name="signal")
        print(f"{os.path.relpath(out, os.path.join(HERE, '..'))}: {n} bytes")


if __name__ == "__main__":
    main()
