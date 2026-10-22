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


if __name__ == "__main__":
    unittest.main()
