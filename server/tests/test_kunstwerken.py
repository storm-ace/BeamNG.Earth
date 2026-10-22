"""kunstwerken: a hand-built structure placed in the road tiles (its triangles per tile, its decks' planes)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ahn  # noqa: E402
import geodesy  # noqa: E402
import kunstwerken  # noqa: E402


class RdTest(unittest.TestCase):
    def test_round_trip(self):
        la, lo = ahn.rd_to_wgs(*ahn.wgs_to_rd(51.9076, 4.4660))
        self.assertAlmostEqual(float(la), 51.9076, places=7)
        self.assertAlmostEqual(float(lo), 4.4660, places=7)


class KunstwerkTest(unittest.TestCase):
    def setUp(self):
        # a 20 x 10 m deck slab at 3 m NAP round RD (91600, 435900), its plane rising 1 % eastward
        self.k = kunstwerken.Kunstwerk({
            "name": "t", "origin_rd": [91600, 435900],
            "positions": [[-10, -5, 3], [10, -5, 3], [10, 5, 3], [-10, 5, 3]], "tris": [[0, 1, 2], [0, 2, 3]],
            "materials": ["bne_bridge_concrete"], "tri_material": [0, 0],
            "decks": [{"ring": [[-10, -5], [10, -5], [10, 5], [-10, 5]], "plane": [3.0, 0.01, 0.0]}]})
        la, lo = ahn.rd_to_wgs(91600.0, 435900.0)
        self.frame = geodesy.EnuFrame(float(la), float(lo))

    def test_tile_mesh_and_deck(self):
        m = self.k.tile_mesh(self.frame, (-50, -50, 50, 50), (0.0, 0.0, 0.0))
        self.assertIsNotNone(m)
        self.assertEqual(len(m[1]), 2)
        # (RD's grid is turned ~1.2 degrees from true north at Rotterdam: corners move ~0.1 m, the size stays)
        self.assertAlmostEqual(float(np.hypot(*(m[0][1, :2] - m[0][0, :2]))), 20.0, delta=0.01)
        np.testing.assert_allclose(m[0][:, 0].min(), -10, atol=0.3)
        self.assertIsNone(self.k.tile_mesh(self.frame, (100, 100, 200, 200), (0.0, 0.0, 0.0)))
        la, lo = ahn.rd_to_wgs(np.array([91605.0, 91700.0]), np.array([435900.0, 435900.0]))
        z = self.k.deck_top(la, lo)
        self.assertAlmostEqual(float(z[0]), 3.05, places=3)
        self.assertTrue(np.isnan(z[1]))


class CapTest(unittest.TestCase):
    """The terrain under a kunstwerk: under the lowest floor within CAP_REACH_M, so the lattice's triangle between a
    trench's floor and the street over it lies under the street, not across the trench's wall."""

    def test_street_beside_a_trench_is_capped_to_its_floor(self):
        z = np.full((20, 20), 3.0)
        z[:, 8:12] = -2.5                                   # a trench 4 m wide, x 8..12 m
        k = kunstwerken.Kunstwerk({
            "name": "t", "origin_rd": [91600, 435900],
            "positions": [[-10, -10, 3], [10, -10, 3], [10, 10, 3], [-10, 10, 3]], "tris": [[0, 1, 2], [0, 2, 3]],
            "materials": ["bne_grass"], "tri_material": [0, 0],
            "floor": {"x0": 0, "y0": 0, "nx": 20, "ny": 20, "z": z.ravel().tolist()}})
        saved, kunstwerken._all = kunstwerken._all, [k]
        try:
            la, lo = ahn.rd_to_wgs(91600.0 + np.array([10.5, 6.5, 2.5]), 435900.0 + np.array([10.5, 10.5, 10.5]))
            got = kunstwerken.cap(lambda a, b: np.full(np.shape(a), 3.4))(np.atleast_1d(la), np.atleast_1d(lo))
            np.testing.assert_allclose(got, [-2.65, -2.65, 2.85], atol=1e-6,
                                       err_msg="in the trench; the street 2 m off its wall; the street 6 m off")
        finally:
            kunstwerken._all = saved


class FootprintTest(unittest.TestCase):
    """What stands of ours round a kunstwerk is kept out of its floor (padded), not out of its box: the area is a
    rotated rectangle and the box's corners reach past it."""

    def setUp(self):
        # its floor a diamond |x| + |y| <= 8 m in 1 m cells round RD (91600, 435900); its mesh the 20 x 20 m square
        c = np.arange(-10, 10) + 0.5
        z = np.where(np.abs(c)[None, :] + np.abs(c)[:, None] <= 8, 1.0, -999.0)
        self.k = kunstwerken.Kunstwerk({
            "name": "t", "origin_rd": [91600, 435900],
            "positions": [[-10, -10, 1], [10, -10, 1], [10, 10, 1], [-10, 10, 1]], "tris": [[0, 1, 2], [0, 2, 3]],
            "materials": ["bne_grass"], "tri_material": [0, 0],
            "floor": {"x0": -10, "y0": -10, "nx": 20, "ny": 20, "z": z.ravel().tolist()}})
        la, lo = ahn.rd_to_wgs(91600.0, 435900.0)
        self.frame = geodesy.EnuFrame(float(la), float(lo))
        self.saved, kunstwerken._all = kunstwerken._all, [self.k]

    def tearDown(self):
        kunstwerken._all = self.saved

    def ll(self, x, y):
        la, lo = ahn.rd_to_wgs(91600.0 + np.asarray(x, np.float64), 435900.0 + np.asarray(y, np.float64))
        return np.atleast_1d(la), np.atleast_1d(lo)

    def test_floor_padded_not_box(self):
        la, lo = self.ll([0.0, 7.5, 9.5, 9.5], [0.0, 0.0, 0.0, 9.5])
        # on it; on its rim; 1.5 m past its rim (within the pad); a box corner 6 m off it
        self.assertEqual(kunstwerken.in_footprint(la, lo).tolist(), [True, True, True, False])
        self.assertTrue(self.k.in_box(la[3:], lo[3:]).all())
        la, lo = self.ll([30.0], [0.0])
        self.assertFalse(kunstwerken.in_footprint(la, lo).any())

    def test_keep_out_and_heads(self):
        e, n, _ = self.frame.geodetic_to_enu(*self.ll([0.0, 40.0], [0.0, 0.0]), 0.0)
        on, off = [float(e[0]), float(n[0])], [float(e[1]), float(n[1])]
        self.assertEqual(kunstwerken.keep_out(self.frame, [on, off]).tolist(), [True, False])
        app = lambda p: ["c1"] + p + [0.0, 1.0, 0.0, 0] + p + [0.5, 90.0]     # noqa: E731  (stop point, head at p)
        sig = [{"j": "n1", "kind": "signal", "app": [app(on), app(off), ["c2"] + off + [0, 1, 0, 1, None, None, None, None]]}]
        kunstwerken.hide_heads(self.frame, sig)
        a = sig[0]["app"]
        self.assertEqual(a[0][7:], [None] * 4, "no stock boom in its footprint (it has no heads of its own)")
        self.assertEqual(a[0][:7], app(on)[:7], "the approach still holds the AI")
        self.assertEqual(a[1][7:9], off, "elsewhere as before")
        self.assertIsNone(a[2][7])

    def test_its_own_head_for_an_approach(self):
        """An approach in its footprint gets the kunstwerk's head that looks at its traffic, nearest its stop point,
        with its Dutch shape; one looking the other way is not it."""
        self.k.signals = [(np.array([3.0, -2.0, 4.2]), np.array([-1.0, 0.0])),     # looks west: at eastbound traffic? no
                          (np.array([4.0, 2.0, 4.2]), np.array([-1.0, 0.0])),      # looks west
                          (np.array([-4.0, 2.0, 4.2]), np.array([1.0, 0.0]))]      # looks east
        e, n, _ = self.frame.geodetic_to_enu(*self.ll([2.0], [0.0]), 0.0)
        stop = [float(e[0]), float(n[0])]
        # westbound traffic (travel -x) arriving at a stop point 2 m east of the middle: the heads looking east face it
        a = ["c1"] + stop + [0.0, -1.0, 0.0, 0, None, None, None, None]
        kunstwerken.hide_heads(self.frame, [{"j": "n1", "kind": "signal", "app": [a]}])
        he, hn, _ = self.frame.geodetic_to_enu(*self.ll([-4.0], [2.0]), 0.0)
        self.assertAlmostEqual(a[7], float(he[0]), places=1)
        self.assertAlmostEqual(a[8], float(hn[0]), places=1)
        self.assertAlmostEqual(a[9], 4.2, places=1)
        self.assertEqual(a[11], kunstwerken.HEAD_SHAPE)
        yaw = np.radians(a[10])         # its local +x along (fy, -fx) = (0, -1) for a head looking east: yaw 270
        self.assertAlmostEqual(float(np.cos(yaw)), 0.0, places=1)
        self.assertAlmostEqual(float(np.sin(yaw)), -1.0, places=1)


class ShippedPaintTest(unittest.TestCase):
    """The shipped kunstwerken's road paint faces up: a face wound the other way is culled from above and the paint is
    not seen (the trench's lane lines were all wound down, its floor bare asphalt in game, the user 2026-10-22)."""

    def test_paint_faces_up(self):
        import glob
        import json
        files = glob.glob(str(Path(kunstwerken.DIR) / "*.kunstwerk.json"))
        self.assertTrue(files)
        for f in files:
            d = json.load(open(f))
            P, T, M = np.asarray(d["positions"]), np.asarray(d["tris"]), np.asarray(d["tri_material"])
            for name in ("bne_paint_white", "bne_cycle_red"):
                if name not in d["materials"]:
                    continue
                t = T[M == d["materials"].index(name)]
                a, b, c = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
                nz = np.cross(b - a, c - a)[:, 2]
                flat = np.abs(nz) > 1e-6
                self.assertEqual(int((nz[flat] < 0).sum()), 0, f"{Path(f).name}: {name} faces wound down")

    def test_droogleever_middle_is_a_hollow(self):
        """Between Droogleever's two trench carriageways the ground lies in the cut (the AHN: -1.2 .. -1.6 m NAP), not
        on a berm at the streets' height walled in on both sides (+1.3 .. +3.5 m, the user's Street View 2026-10-22)."""
        import json
        d = json.load(open(Path(kunstwerken.DIR) / "droogleever.kunstwerk.json"))
        f = d["floor"]
        z = np.asarray(f["z"]).reshape(f["ny"], f["nx"])
        for x, y in ((8, 0), (12, 0), (16, -25), (20, -25)):
            self.assertLess(float(z[int(y - f["y0"]), int(x - f["x0"])]), 0.0, f"floor at ({x}, {y})")


if __name__ == "__main__":
    unittest.main()
