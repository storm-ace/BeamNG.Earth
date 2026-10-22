"""tools/kunstwerk_solid.py: a cycle path's passage under a deck (passage_floor: its floor along it and its open run)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))
import kunstwerk_solid as ks  # noqa: E402


def hump(ts, d0, d1, path=-0.3, deck=3.5, blur=8.0, rise=0.004):
    """The plan's surface over a passage's line: the path at its own level (rising rise per m), the deck's street over
    [d0, d1], a cosine blur of blur m between them (Droogleever's ring, 2026-10-22)."""
    base = path + rise * ts
    dist = np.where(ts < d0, d0 - ts, np.where(ts > d1, ts - d1, 0.0))
    w = np.where(dist < blur, 0.5 + 0.5 * np.cos(np.pi * np.minimum(dist / blur, 1.0)), 0.0)
    return base * (1 - w) + deck * w


class PassageFloorTest(unittest.TestCase):
    def setUp(self):
        self.ts = np.arange(-60.0, 60.0 + 1e-6, 0.5)
        self.d0, self.d1 = -12.0, 12.0

    def soffit(self, top):
        return np.where((self.ts >= self.d0) & (self.ts <= self.d1), top - ks.DECK_T, np.nan)

    def test_under_the_hump_not_over_it(self):
        """The floor is the path's own level read past the blur, under the deck too, never over the surface; its open
        run reaches over the whole hump and no farther."""
        surf = hump(self.ts, self.d0, self.d1)
        floor, ta, tb = ks.passage_floor(self.ts, surf, self.d0, self.d1, self.soffit(3.5))
        np.testing.assert_allclose(floor, -0.3 + 0.004 * self.ts, atol=1e-6)
        self.assertTrue((floor <= surf + 1e-9).all(), "the floor under the plan's surface")
        self.assertLess(ta, self.d0 - 6.0)
        self.assertGreater(ta, self.d0 - 8.5, "the open run ends where the blur does")
        self.assertGreater(tb, self.d1 + 6.0)
        self.assertLess(tb, self.d1 + 8.5)
        run = (self.ts >= ta) & (self.ts <= tb)
        self.assertTrue((surf[run] - floor[run] > ks.PASSAGE_MEET - 1e-9).all())

    def test_headroom_and_ramps(self):
        """A low deck: the floor under it CYCLE_CLEARANCE under its soffit, dipping to that at no more than
        CYCLE_GRADE, its ramps monotonic: down from either open end to the deck, level under it."""
        surf = hump(self.ts, self.d0, self.d1, path=0.2, deck=3.2, rise=0.0)
        soffit = self.soffit(3.2)
        floor, ta, tb = ks.passage_floor(self.ts, surf, self.d0, self.d1, soffit)
        under = np.isfinite(soffit)
        self.assertLessEqual(float(np.max(floor[under] - (soffit[under] - ks.CYCLE_CLEARANCE))), 1e-9)
        self.assertAlmostEqual(float(floor[under].max()), 3.2 - ks.DECK_T - ks.CYCLE_CLEARANCE, places=6)
        slope = np.abs(np.diff(floor)) / np.diff(self.ts)
        self.assertLessEqual(float(slope.max()), ks.CYCLE_GRADE * np.pi / 2 + 1e-9, "a cosine of CYCLE_GRADE's run")
        before = (self.ts >= ta) & (self.ts <= self.d0)
        after = (self.ts >= self.d1) & (self.ts <= tb)
        self.assertTrue((np.diff(floor[before]) <= 1e-9).all(), "down to the deck")
        self.assertTrue((np.diff(floor[after]) >= -1e-9).all(), "up from it")
        self.assertTrue((floor <= surf + 1e-9).all())

    def test_no_reading_past_the_ends(self):
        """No surface read past the deck (an area's edge): its lowest, level, and no open run past the samples."""
        ts = np.arange(-14.0, 14.0 + 1e-6, 0.5)
        surf = hump(ts, self.d0, self.d1)
        floor, ta, tb = ks.passage_floor(ts, surf, self.d0, self.d1, np.full(len(ts), np.nan))
        np.testing.assert_allclose(floor, surf.min())
        self.assertTrue(ts[0] <= ta <= self.d0 and self.d1 <= tb <= ts[-1], (ta, tb))


if __name__ == "__main__":
    unittest.main()
