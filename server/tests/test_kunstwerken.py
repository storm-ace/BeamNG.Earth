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


    def enu(self, x, y):
        e, n, _ = self.frame.geodetic_to_enu(*self.ll([x], [y]), 0.0)
        return [float(e[0]), float(n[0])]

    def test_head_first_primary_and_repeater_not_one_behind(self):
        """Each head goes to the approach it stands at: the primary and the repeater both show it (one instance, its
        extra heads in a[12]); a head 11 m back up the approach's own road is not its primary (the Droogleever's
        Westzeedijk approach took one 21 m back, its own head unused, 2026-10-22)."""
        self.k.signals = [(np.array([13.0, 1.0, 4.2]), np.array([1.0, 0.0])),      # 11 m before the stop point
                          (np.array([-3.0, -2.0, 4.2]), np.array([1.0, 0.0])),     # the primary, on the right
                          (np.array([-3.0, 3.0, 4.2]), np.array([1.0, 0.0]))]      # the repeater, on the left
        stop = self.enu(2.0, 0.0)
        a = ["c1"] + stop + [0.0, -1.0, 0.0, 0, None, None, None, None]
        kunstwerken.hide_heads(self.frame, [{"j": "n1", "kind": "signal", "app": [a]}])
        prim, rep = self.enu(-3.0, -2.0), self.enu(-3.0, 3.0)
        self.assertAlmostEqual(a[7], prim[0], places=1)
        self.assertAlmostEqual(a[8], prim[1], places=1)
        self.assertEqual(a[11], kunstwerken.HEAD_SHAPE)
        got = [(round(e[0], 1), round(e[1], 1)) for e in a[12]]
        self.assertIn((round(rep[0], 1), round(rep[1], 1)), got, "the repeater shows the same light")
        for e in a[12]:
            self.assertEqual(e[4], kunstwerken.HEAD_SHAPE)

    def test_head_on_a_road_without_a_signalled_approach(self):
        """A head looking at a graph edge with a mapped signal on that road and no signalled approach gets one made on
        that edge (Overture's graph has no junction where the Droogleever's oval has its lights); a mapped signal off
        the road (a cycle path's) makes none."""
        from types import SimpleNamespace
        self.k.signals = [(np.array([0.0, 5.0, 4.2]), np.array([0.0, -1.0]))]      # looks south: at northbound cars
        h = self.enu(0.0, 5.0)
        graph = {"nodes": {"n1": [h[0] + 2.0, h[1] - 30.0, 1.0, 3.0], "n2": [h[0] + 2.0, h[1] + 30.0, 1.0, 3.0]},
                 "edges": [["n1", "n2", 1, 1, 0, 13.9, 1.0, 3.0]]}
        sig = []
        kunstwerken.hide_heads(self.frame, sig, graph, SimpleNamespace(pts={"signal": np.array([[h[0] + 9.0,
                                                                                                h[1]]])}))
        self.assertEqual(sig, [], "a mapped signal 7 m off the road: a cycle path's, no stop for the cars")
        kunstwerken.hide_heads(self.frame, sig, graph, SimpleNamespace(pts={"signal": np.array([[h[0] + 2.0,
                                                                                                h[1] + 1.0]])}))
        self.assertEqual(len(sig), 1)
        self.assertEqual(sig[0]["j"], "n2")
        a = sig[0]["app"][0]
        self.assertEqual(a[0], "n1")
        self.assertAlmostEqual(a[4], 0.0, places=3)
        self.assertAlmostEqual(a[5], 1.0, places=3)
        self.assertLess(a[2], h[1], "its stop point before the head")
        self.assertGreater(a[1], h[0] + 2.0, "... on the right half of the road")
        self.assertAlmostEqual(a[7], h[0], places=1)
        self.assertEqual(a[11], kunstwerken.HEAD_SHAPE)

    def test_cyclists_head_under_it_and_a_cycle_paths_light(self):
        """A pole's cyclists' head ("bike") switches with the head over it (its own shape); a cycle path's light
        ("cycle") gets no car approach made for it, though a mapped signal stands on the road beside it."""
        from types import SimpleNamespace
        self.k.signals = [(np.array([-3.0, -2.0, 4.2]), np.array([1.0, 0.0])),
                          (np.array([-3.0, -2.0, 2.6]), np.array([1.0, 0.0]))]
        self.k.signal_kinds = ["", "bike"]
        stop = self.enu(2.0, 0.0)
        a = ["c1"] + stop + [0.0, -1.0, 0.0, 0, None, None, None, None]
        kunstwerken.hide_heads(self.frame, [{"j": "n1", "kind": "signal", "app": [a]}])
        self.assertEqual(a[11], kunstwerken.HEAD_SHAPE)
        self.assertEqual([e[4] for e in a[12]], [kunstwerken.HEAD_SHAPE_BIKE])
        self.assertAlmostEqual(a[12][0][2], 2.6, places=1)
        self.k.signals = [(np.array([0.0, 5.0, 4.2]), np.array([0.0, -1.0]))]
        self.k.signal_kinds = ["cycle"]
        h = self.enu(0.0, 5.0)
        graph = {"nodes": {"n1": [h[0] + 2.0, h[1] - 30.0, 1.0, 3.0], "n2": [h[0] + 2.0, h[1] + 30.0, 1.0, 3.0]},
                 "edges": [["n1", "n2", 1, 1, 0, 13.9, 1.0, 3.0]]}
        sig = []
        kunstwerken.hide_heads(self.frame, sig, graph, SimpleNamespace(pts={"signal": np.array([[h[0] + 2.0, h[1]]])}))
        self.assertEqual(sig, [])

    def test_cyclists_head_shows_the_parallel_approach(self):
        """A head for no car road (a cycle path's light) shows the light of the approach beside it along its axis."""
        self.k.signals = [(np.array([-3.0, -6.0, 4.2]), np.array([1.0, 0.0]))]
        stop = self.enu(2.0, 0.0)
        a = ["c1"] + stop + [0.0, -1.0, 0.0, 0, None, None, None, None]
        kunstwerken.hide_heads(self.frame, [{"j": "n1", "kind": "signal", "app": [a]}])
        hd = self.enu(-3.0, -6.0)
        self.assertAlmostEqual(a[7], hd[0], places=1)
        self.assertAlmostEqual(a[8], hd[1], places=1)

    def test_crossing_heads_the_inverse_of_the_road_they_cross(self):
        """A signalled crossing's heads (the user 2026-10-22, "we missen stoplichten") go to the approach driving over
        it as its inverse ("inv": green while that traffic stands): the pedestrians' head with its own shape, not the
        approach's own head; a crossing of a road no approach drives along gets none (not the cross road's)."""
        self.k.signals = [(np.array([-3.0, -2.0, 4.2]), np.array([1.0, 0.0])),     # the car head: at westbound cars
                          (np.array([-6.0, 4.0, 2.45]), np.array([0.0, -1.0])),    # a crossing at x -6 over that road
                          (np.array([-6.0, -4.0, 1.75]), np.array([0.0, -1.0])),   # its cyclists' head at the south kerb
                          (np.array([30.0, 4.0, 2.45]), np.array([1.0, 0.0]))]     # a crossing over a north-south road
        self.k.signal_kinds = ["", "cross_ped", "cross_bike", "cross_ped"]
        self.k.signal_across = [None, np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        stop = self.enu(2.0, 0.0)
        a = ["c1"] + stop + [0.0, -1.0, 0.0, 0, None, None, None, None]
        kunstwerken.hide_heads(self.frame, [{"j": "n1", "kind": "signal", "app": [a]}])
        self.assertEqual(a[11], kunstwerken.HEAD_SHAPE, "the car head is the approach's own")
        inv = [e for e in a[12] if len(e) > 5 and e[5] == "inv"]
        self.assertEqual(sorted(e[4] for e in inv), sorted([kunstwerken.HEAD_SHAPE_PED, kunstwerken.HEAD_SHAPE_BIKE]))
        ped = next(e for e in inv if e[4] == kunstwerken.HEAD_SHAPE_PED)
        hp = self.enu(-6.0, 4.0)
        self.assertAlmostEqual(ped[0], hp[0], places=1)
        self.assertAlmostEqual(ped[2], 2.45, places=1)
        self.assertEqual(len(a[12]), 2, "the north-south road's crossing: no approach drives it, none")


class BareTest(unittest.TestCase):
    """The ground under a kunstwerk's rails is bare: the game's ground cover grew through them (in game 2026-10-22)."""

    def test_bare_round_its_rails_only(self):
        k = kunstwerken.Kunstwerk({
            "name": "t", "origin_rd": [91600, 435900],
            "positions": [[-10, -10, 3], [10, -10, 3], [10, 10, 3], [-10, 10, 3],
                          [-10, 0, 3.03], [10, 0, 3.03], [10, 0.07, 3.03], [-10, 0.07, 3.03]],
            "tris": [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            "materials": ["bne_grass", "bne_rail_steel"], "tri_material": [0, 0, 1, 1],
            "floor": {"x0": -10, "y0": -10, "nx": 20, "ny": 20, "z": [3.0] * 400}})
        saved, kunstwerken._all = kunstwerken._all, [k]
        try:
            la, lo = ahn.rd_to_wgs(91600.0 + np.array([-9.5, 0.3, 0.3, 0.3]), 435900.0 + np.array([0.5, 1.5, 6.0, -8.0]))
            got = kunstwerken.bare(np.asarray(la), np.asarray(lo))
            np.testing.assert_array_equal(got, [True, True, False, False],
                                          err_msg="at the rail's end; its middle 1.5 m off it; 6 m and 8 m off it")
        finally:
            kunstwerken._all = saved


class TwoSidedTest(unittest.TestCase):
    """A thin part seen from both sides (a sign's face: each triangle and its reverse) has its back on vertices of its
    own: on shared ones the engine's vertex normals summed +n and -n to rounding noise, a direction at random per
    vertex, and the E1 plate's face showed radial streaks in game (2026-10-22)."""

    def test_back_on_its_own_vertices(self):
        import cdae
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        P = np.r_[[[0.0, 0.0, 2.0]], np.c_[0.3 * np.cos(a), np.zeros(12), 2.0 + 0.3 * np.sin(a)]]
        P = P @ np.array([[np.cos(0.37), np.sin(0.37), 0], [-np.sin(0.37), np.cos(0.37), 0], [0, 0, 1]])
        T = np.array([[0, (i + 1) % 12 + 1, i + 1] for i in range(12)])
        T2 = np.r_[T, T[:, ::-1], [[1, 2, 3]]]          # (and one more, one-sided, not paired)
        UV = np.c_[np.arange(13.0), np.arange(13.0)]
        P2, T3, UV2 = kunstwerken.two_sided_apart(P, T2, UV)
        self.assertEqual(len(P2), 26)
        self.assertEqual(T3[:12].tolist(), T.tolist(), "the front as it was")
        np.testing.assert_array_equal(UV2[T3[12:24]], UV[T[:, ::-1]], "the back: the same UVs")
        N = cdae.face_normal_sum(P2, T3[:24])
        fn = np.cross(P[T[0, 1]] - P[T[0, 0]], P[T[0, 2]] - P[T[0, 0]])
        fn /= np.linalg.norm(fn)
        np.testing.assert_allclose(N[np.unique(T3[:12])] @ fn, 1.0, atol=1e-9)
        np.testing.assert_allclose(N[np.unique(T3[12:24])] @ fn, -1.0, atol=1e-9)
        self.assertIs(kunstwerken.two_sided_apart(P, T, UV)[1], T, "nothing two-sided: as it was")


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


class DroogleeverCycleTunnelTest(unittest.TestCase):
    """The cycle paths along 's-Gravendijkwal pass under Droogleever's ring and the Westzeedijk in passages of their own
    beside the trench (OSM's tunnel=yes, layer=-1 cycleways), not up the ring's embankment and through it as solid
    ground (in game 2026-10-22: "de fietspad gaat door de grond, niet ondergronds"). Their lines in the kunstwerk's
    local RD metres (ref/osm_map.xml): each end is a portal."""

    TUNNELS = {"45754201": ((3, -36), (10, -60)), "45754202": ((45, -51), (37, -21)),
               "45754204": ((16, 48), (6, 73)), "45754210": ((-33, 60), (-21, 32))}
    CLEARANCE = 2.5             # m over the path under a deck (tools/kunstwerk_solid.py CYCLE_CLEARANCE)

    @classmethod
    def setUpClass(cls):
        import json
        d = json.load(open(Path(kunstwerken.DIR) / "droogleever.kunstwerk.json"))
        cls.k = kunstwerken.Kunstwerk(d)
        cls.P = np.asarray(d["positions"], np.float64)
        cls.T = np.asarray(d["tris"], np.int64)

    def _centre(self, name):
        """The passage's middle line under its deck: the deck over the tunnel's middle (a ring round the band beside
        the trench, its corners at both ends' inner and outer edge), from end to end."""
        a, b = (np.asarray(v, np.float64) for v in self.TUNNELS[name])
        m = (a + b) / 2
        for ring, _ in self.k.decks:
            if len(ring) == 4 and kunstwerken._inside(ring, m[:1], m[1:])[0]:
                return (ring[0] + ring[3]) / 2, (ring[1] + ring[2]) / 2
        self.fail(f"{name}: no deck of its own over it")

    def _line(self, a, b, step=0.5, reach=0.0):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        u = (b - a) / np.hypot(*(b - a))
        s = np.arange(-reach, np.hypot(*(b - a)) + reach + 1e-6, step)
        return a + np.outer(s, u)

    def _wgs(self, xy):
        o = self.k.origin
        return ahn.rd_to_wgs(xy[:, 0] + o[0], xy[:, 1] + o[1])

    def _solid_over(self, xy, lo, hi):
        """Per point whether a triangle of the kunstwerk covers it between heights lo and hi (per point)."""
        A, B, C = (self.P[self.T[:, j]] for j in range(3))
        box_lo, box_hi = np.minimum(np.minimum(A, B), C), np.maximum(np.maximum(A, B), C)
        out = np.zeros(len(xy), bool)
        for i, (x, y) in enumerate(xy):
            m = np.flatnonzero((box_lo[:, 0] <= x) & (box_hi[:, 0] >= x) & (box_lo[:, 1] <= y) & (box_hi[:, 1] >= y)
                               & (box_hi[:, 2] > lo[i]) & (box_lo[:, 2] < hi[i]))
            for j in m:
                a, b, c = A[j], B[j], C[j]
                d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if abs(d) < 1e-12:
                    continue
                w1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / d
                w2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / d
                w3 = 1 - w1 - w2
                if min(w1, w2, w3) >= -1e-9 and lo[i] < w1 * a[2] + w2 * b[2] + w3 * c[2] < hi[i]:
                    out[i] = True
                    break
        return out

    def test_headroom_under_the_decks(self):
        """Along each passage's middle under its deck (its ends 1 m in): a floor (the kunstwerk's, the cycle path's red)
        with a deck over it at least CLEARANCE over its slab's underside, and nothing of the kunstwerk in between."""
        for name in self.TUNNELS:
            xy = self._line(*self._centre(name))
            xy = xy[2:-2]
            la, lo = self._wgs(xy)
            floor = self.k.floor_at(la, lo)
            top = self.k.deck_top(la, lo)
            self.assertTrue(np.isfinite(floor).all(), f"{name}: a floor all along")
            self.assertTrue(np.isfinite(top).all(), f"{name}: a deck all over it")
            self.assertLess(float(np.max(floor)), 1.0, f"{name}: the path in the cut, not on the ring")
            room = top - DECK_T - floor
            self.assertGreaterEqual(float(room.min()), self.CLEARANCE - 0.1, f"{name}: headroom {room.min():.2f} m")
            hit = self._solid_over(xy, floor + 0.15, floor + self.CLEARANCE - 0.1)
            self.assertFalse(hit.any(), f"{name}: solid in the passage at {xy[hit][:3].round(1).tolist()}")

    def test_ramps_without_a_hump(self):
        """From 25 m before a portal to 25 m past the other the floor along the passage climbs and drops no more than
        a ramp's (the path rose 3.5 m onto the ring before), and the terrain is kept under it."""
        for name in self.TUNNELS:
            xy = self._line(*self._centre(name), 1.0, reach=25.0)
            la, lo = self._wgs(xy)
            floor = self.k.floor_at(la, lo)
            ok = np.isfinite(floor)
            self.assertGreater(ok.mean(), 0.9, name)
            f = floor[ok]
            self.assertLess(float(np.ptp(f)), 1.2, f"{name}: floor {f.min():.2f}..{f.max():.2f}")
            self.assertLess(float(np.abs(np.diff(f)).max()), 0.2, f"{name}: a step along it")
            saved, kunstwerken._all = kunstwerken._all, [self.k]
            try:
                ground = kunstwerken.cap(lambda a_, b_: np.full(np.shape(a_), 3.4))(la[ok], lo[ok])
            finally:
                kunstwerken._all = saved
            self.assertTrue((ground <= f - kunstwerken.CAP_M + 1e-6).all(), f"{name}: the terrain over the path")

    def test_the_generators_path_left_out(self):
        """The BGT's fietspad there (level -1 under the deck, its ramps at 0) is the kunstwerk's: covered, so bgt.mesh
        leaves it out (skip) and blockworker's _kw_cut the rest; OSM's cycleways the generator draws not at all
        (roads.STYLE: test_roads)."""
        for name in self.TUNNELS:
            la, lo = self._wgs(self._line(*self._centre(name), 1.0, reach=10.0))
            saved, kunstwerken._all = kunstwerken._all, [self.k]
            try:
                self.assertTrue(kunstwerken.covered(la, lo).all(), name)
            finally:
                kunstwerken._all = saved


DECK_T = 0.9                # m: a deck's slab (tools/kunstwerk_solid.py)


if __name__ == "__main__":
    unittest.main()
