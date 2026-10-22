"""tools/ndw_signs.py: NDW's national sign file read as a stream, the signs round a point kept."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import ndw_signs  # noqa: E402


def feature(i, lon, lat, code="B6"):
    return {"type": "Feature", "id": f"s{i}", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"rvvCode": code, "placement": "L", "bearing": 90, "textSigns": []}}


class AroundTest(unittest.TestCase):
    def test_keeps_the_signs_in_reach_across_chunks(self):
        feats = [feature(i, 4.4658 + (i % 7 - 3) * 0.004, 51.908 + (i % 5 - 2) * 0.004) for i in range(3000)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "national.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": feats}, f, separators=(",", ":"))
            saved = ndw_signs.CHUNK
            ndw_signs.CHUNK = 4096                       # (many chunk edges cutting features)
            try:
                got = ndw_signs.around(path, 51.908, 4.4658, 300.0)
            finally:
                ndw_signs.CHUNK = saved
        want = {f["id"] for f in feats if abs(f["geometry"]["coordinates"][1] - 51.908) < 0.0027
                and abs(f["geometry"]["coordinates"][0] - 4.4658) < 0.0044}
        self.assertTrue(want)
        self.assertEqual({f["id"] for f in got}, want)


if __name__ == "__main__":
    unittest.main()
