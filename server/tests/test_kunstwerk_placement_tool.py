"""tools/kunstwerk_placement.py: NDW's speed limits with their number, back-to-back signs on one post, and --check
(which way each placed sign and head looks against the traffic of the road it stands at)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))
import kunstwerk_placement as kp  # noqa: E402


class SpeedCodeTest(unittest.TestCase):
    def test_a1_with_its_number(self):
        """An A1 plate shows the limit NDW's register gives (its blackCode, else a number in its textSigns), not the
        Commons drawing's 50 every A1 showed (the user, 2026-10-22)."""
        self.assertEqual(kp.speed_code({"rvvCode": "A1", "blackCode": "30", "textSigns": []}), "A1_30")
        self.assertEqual(kp.speed_code({"rvvCode": "A1", "blackCode": None,
                                        "textSigns": [{"type": "VRIJ", "text": "controle"}, {"text": "70"}]}), "A1_70")
        self.assertEqual(kp.speed_code({"rvvCode": "A1", "blackCode": "35"}), "A1", "no cell for 35: the plain A1")
        self.assertEqual(kp.speed_code({"rvvCode": "B6", "blackCode": "30"}), "B6")


def ndw_feature(lat, lon, code, bearing):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"rvvCode": code, "placement": "L", "bearing": bearing}}


class NdwPostsTest(unittest.TestCase):
    def test_back_to_back_on_one_post(self):
        """NDW lists each face of a sign seen both ways (an E1 for each way of a street): they stand on one post, not
        on two 0.2 m apart with their plates in each other."""
        feats = [ndw_feature(0.0, 0.0, "E1", 0.0), ndw_feature(0.0, 1.0, "E1", 180.0)]

        def local(e):
            return np.array([0.0, 0.17 * e["lon"]])
        items = []
        kp.ndw_signs(feats, items, local, lambda p: True, lambda p: None)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["at"], items[1]["at"])
        self.assertLess(float(np.dot(items[0]["facing"], items[1]["facing"])), -0.9)
        self.assertEqual(items[0]["bearing"], 0.0)


class CheckTest(unittest.TestCase):
    """--check on a one-way road running north (RD y up) with a signal on it."""

    def setUp(self):
        A = np.array([[0.0, -50.0], [0.0, 0.0]])
        B = np.array([[0.0, 0.0], [0.0, 50.0]])
        self.ways = [("1", {"name": "Weg", "highway": "primary", "oneway": "yes"}, A, B, (1,))]
        self.sigs = [("7", np.array([0.0, -2.0]), {"highway": "traffic_signals"}, [("1", np.array([0.0, 1.0]))])]

    def check(self, items):
        return kp.check_items(items, self.ways, self.sigs)

    def test_a_light_facing_its_traffic(self):
        r = self.check([{"module": "traffic_light", "at": [3.0, 0.0], "facing": [0.0, -1.0]}])[0]
        self.assertEqual(r["flags"], [])
        self.assertEqual(r["face_err"], 0)
        self.assertEqual(r["approach"], "n7/w1")
        self.assertEqual(r["travel"], [0.0, 1.0])

    def test_turned_away_and_no_signal(self):
        r = self.check([{"module": "traffic_light", "at": [3.0, 0.0], "facing": [0.0, 1.0]}])[0]
        self.assertIn("away", r["flags"])
        self.assertIn("no signal", r["flags"], "its face looks at no way in of the signal")

    def test_two_heads_one_pole_one_approach(self):
        rows = self.check([{"module": "head", "at": [3.0, 0.0], "facing": [0.0, -1.0]},
                           {"module": "head", "at": [3.1, 0.0], "facing": [0.0, -1.0]},
                           {"module": "head", "code": "bike head", "at": [3.1, 0.0], "facing": [0.0, -1.0]}])
        self.assertEqual([r["flags"] for r in rows], [[], ["2 heads"], []])

    def test_ndw_bearing_and_no_entry(self):
        rows = self.check([{"module": "sign", "code": "B6", "at": [3.0, 0.0], "facing": [0.0, -1.0], "bearing": 0.0},
                           {"module": "sign", "code": "B6", "at": [3.0, 9.0], "facing": [0.0, -1.0], "bearing": 90.0},
                           {"module": "sign", "code": "C2", "at": [-3.0, 30.0], "facing": [0.0, 1.0],
                            "bearing": 180.0}])
        self.assertEqual(rows[0]["flags"], [])
        self.assertEqual(rows[1]["flags"], ["ndw"], "NDW: its traffic drives east, the road runs north")
        self.assertEqual(rows[2]["flags"], [], "no entry: it faces the way the one-way road forbids")

    def test_one_sign_twice(self):
        rows = self.check([{"module": "sign", "code": "E1", "at": [3.0, 0.0], "facing": [0.0, -1.0]},
                           {"module": "sign", "code": "E1", "at": [3.0, 0.9], "facing": [0.0, -1.0]},
                           {"module": "sign", "code": "E1", "at": [3.0, 0.2], "facing": [0.0, 1.0]}])
        self.assertEqual(rows[1]["flags"], ["double of 0"])
        self.assertNotIn("double of 0", rows[2]["flags"], "back to back: two faces, not one sign twice")


OSM = """<osm>
 <node id="1" lat="0" lon="-1"/><node id="2" lat="0" lon="0"/><node id="3" lat="0" lon="1"/>
 <node id="4" lat="0" lon="0.5"><tag k="highway" v="traffic_signals"/></node>
 <node id="10" lat="-1" lon="-0.6"/><node id="11" lat="1" lon="-0.6"/>
 <node id="20" lat="-1" lon="-0.4"/><node id="21" lat="1" lon="-0.4"/>
 <node id="30" lat="0" lon="-5"/><node id="31" lat="0" lon="-4.6"/><node id="32" lat="0" lon="-4"/>
 <node id="40" lat="-1" lon="-4.5"/><node id="41" lat="1" lon="-4.5"/>
 <way id="100"><nd ref="1"/><nd ref="2"/><nd ref="4"/><nd ref="3"/>
  <tag k="highway" v="primary"/><tag k="name" v="Weg"/></way>
 <way id="101"><nd ref="10"/><nd ref="11"/><tag k="highway" v="footway"/></way>
 <way id="102"><nd ref="20"/><nd ref="21"/><tag k="highway" v="cycleway"/></way>
 <way id="103"><nd ref="30"/><nd ref="31"/><nd ref="32"/><tag k="highway" v="residential"/></way>
 <way id="104"><nd ref="40"/><nd ref="41"/><tag k="highway" v="footway"/></way>
</osm>"""


def local10(e):
    """lat/lon as tens of metres (x east, y north)."""
    return np.array([10.0 * e["lon"], 10.0 * e["lat"]])


class CrossingsTest(unittest.TestCase):
    """Signalled crossings from OSM (the user 2026-10-22, "we missen stoplichten"): a road running east with a car
    signal at x 5, a footway and a cycle path crossing it at x -6 and -4 (one crossing, 2 m apart), and a footway
    over a side road 45 m off (no signal near: not a signalled one)."""

    def test_found_merged_signalled(self):
        cs = kp.osm_crossings(OSM, local10)
        sig = [c for c in cs if c["signalled"]]
        self.assertEqual(len(sig), 1)
        c = sig[0]
        self.assertEqual(c["kinds"], ["bike", "ped"], "the footway and its cycle path: one crossing")
        self.assertEqual(c["way"], "100")
        self.assertLess(abs(float(c["at"][1])), 1e-6)
        self.assertAlmostEqual(abs(float(c["road"][0])), 1.0)
        self.assertIn("car signal", c["why"])
        self.assertFalse(next(c for c in cs if c["way"] == "103")["signalled"], "45 m from any signal")

    def test_a_pole_at_each_kerb(self):
        """Its poles at both kerbs (the road 8 m wide), back from them; the pedestrians' head looking across the road,
        the cyclists' at those arriving; a cycle path's own light near one end leaves its cyclists' head out."""
        c = next(c for c in kp.osm_crossings(OSM, local10) if c["signalled"])

        def kind(x, y):
            return np.where(np.abs(np.asarray(y)) <= 4.0, 0, 1)        # 0: the road
        items = []
        n, skipped = kp.crossing_lights([c], items, lambda p: True, kind, 0)
        self.assertEqual((n, skipped), (2, []))
        for it in items:
            o = np.asarray(it["facing"])
            self.assertGreater(abs(it["at"][1]), 4.0, "off the road")
            self.assertGreater(float(o @ np.sign(it["at"])), 0.5 * abs(o[1]), "facing away from the road")
            heads = {h["kind"]: np.asarray(h["facing"]) for h in it["heads"]}
            self.assertLess(float(heads["ped"] @ o), -0.9, "the pedestrians' head looks across the road")
            self.assertGreater(float(heads["bike"] @ o), 0.9, "the cyclists' at those arriving here")
        north = next(it for it in items if it["at"][1] > 0)
        items2 = [{"module": "traffic_light", "bike": True, "cycle": True, "at": [north["at"][0] + 2.0, 6.0],
                   "facing": north["facing"]}]
        kp.crossing_lights([c], items2, lambda p: True, kind, 0)
        got = {round(it["at"][1]): [h["kind"] for h in it["heads"]] for it in items2[1:]}
        self.assertEqual(sorted(got.values()), [["ped"], ["ped", "bike"]])

    def test_coverage(self):
        """--check's coverage: a signal node with no head near (unless a head's approach is it), a crossing with heads
        on one side only."""
        c = next(c for c in kp.osm_crossings(OSM, local10) if c["signalled"])
        sigs = [("4", np.array([5.0, 0.0]), {"highway": "traffic_signals"}, [])]
        lone, sides = kp.coverage([[-5.0, 6.0]], sigs, [c], lambda p: True)
        self.assertEqual([x[0] for x in lone], ["4"])
        self.assertEqual(len(sides), 1)
        self.assertEqual(sorted(sides[0][1:]), [0, 1], "heads on one side only")
        lone, sides = kp.coverage([[-5.0, 6.0], [-5.0, -6.0], [6.0, -9.0]], sigs, [c], lambda p: True, {"4"})
        self.assertEqual((lone, sides), ([], []))


if __name__ == "__main__":
    unittest.main()
