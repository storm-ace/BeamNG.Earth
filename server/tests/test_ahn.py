"""ahn: the AHN's holes filled (under a deck the road under it, under a building the ground round it), RD coordinates,
and AhnDem's ground (the AHN's where it has it, water from the base DEM)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ahn  # noqa: E402


class RdTest(unittest.TestCase):
    def test_amersfoort_and_the_junction(self):
        x, y = ahn.wgs_to_rd(52.15517440, 5.38720621)
        self.assertAlmostEqual(float(x), 155000.0, delta=0.01)
        self.assertAlmostEqual(float(y), 463000.0, delta=0.01)
        x, y = ahn.wgs_to_rd(51.9076, 4.4660)            # Droogleever Fortuynplein: where the AHN tile showed it
        self.assertAlmostEqual(float(x), 91610, delta=5)
        self.assertAlmostEqual(float(y), 435857, delta=5)


class FillTest(unittest.TestCase):
    def deck(self):
        """A trench (floor -1.5, 10 px wide) along the rows under a deck (rows 20-29: no ground; surface 3.2) in a
        street at 3.2."""
        dtm = np.full((50, 40), 3.2, np.float32)
        dtm[:, 15:25] = -1.5
        dtm[20:30, :] = np.nan
        dsm = np.where(np.isfinite(dtm), dtm, 3.2).astype(np.float32)
        return dtm, dsm

    def test_under_a_deck_the_trench_runs_on(self):
        dtm, dsm = self.deck()
        g = ahn.ground(dtm, dsm)
        self.assertTrue(np.isfinite(g).all())
        np.testing.assert_allclose(g[20:30, 18:22], -1.5, err_msg="the floor under the deck, not the street's mean")
        self.assertGreater(float((dsm - g)[25, 20]), ahn.DECK_RISE_M)

    def test_under_a_building_the_ground_round_it(self):
        dtm = np.full((40, 40), 1.0, np.float32)
        dtm[10:30, 10:30] = np.nan
        dtm[0, :] = -3.0                                   # a ditch along the edge: no reason to sink the house
        dsm = np.where(np.isfinite(dtm), dtm, 12.0).astype(np.float32)     # its roof
        g = ahn.ground(dtm, dsm)
        self.assertGreater(float(g[20, 20]), 0.5)

    def test_fill_low_takes_the_lowest_of_the_rim(self):
        a = np.array([[3.0, 3.0, 3.0, 3.0], [3.0, np.nan, np.nan, -1.0], [3.0, 3.0, 3.0, 3.0]], np.float32)
        np.testing.assert_allclose(ahn.fill_low(a)[1, 1:3], -1.0)


def grid_fill(a, rounds=ahn.FILL_ROUNDS, smooth=ahn.SMOOTH_ROUNDS):
    """fill as it was, round by round over the whole grid (the reference: fill looks at the last ring's neighbours
    only, for the same result)."""
    a = np.asarray(a, np.float32)
    hole = ~np.isfinite(a)
    if not hole.any() or hole.all():
        return a
    out = np.where(hole, 0.0, a).astype(np.float32)
    known = ~hole
    for _ in range(rounds):
        todo = ~known
        p = np.pad(out, 1, mode="edge")
        v = np.pad(known, 1, mode="edge").astype(np.float32)
        s = (p[:-2, 1:-1] * v[:-2, 1:-1] + p[2:, 1:-1] * v[2:, 1:-1] + p[1:-1, :-2] * v[1:-1, :-2]
             + p[1:-1, 2:] * v[1:-1, 2:])
        n = v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:]
        got = todo & (n > 0)
        if not got.any():
            break
        out[got] = s[got] / n[got]
        known |= got
    filled = hole & known
    for _ in range(smooth):
        p = np.pad(out, 1, mode="edge")
        out[filled] = ((p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0)[filled]
    out[~known] = np.nan
    return out


def grid_fill_low(a, rounds=ahn.FILL_ROUNDS):
    a = np.asarray(a, np.float32)
    hole = ~np.isfinite(a)
    if not hole.any() or hole.all():
        return a
    out = np.where(hole, np.inf, a).astype(np.float32)
    for _ in range(rounds):
        p = np.pad(out, 1, mode="constant", constant_values=np.inf)
        m = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        lower = hole & (m < out)
        if not lower.any():
            break
        out[lower] = m[lower]
    out[~np.isfinite(out)] = np.nan
    return out


class FrontierTest(unittest.TestCase):
    def test_same_as_round_by_round_over_the_grid(self):
        """Random holes (large ones, at the edges, a few rounds only so some stay NaN): bit for bit the grid's."""
        rng = np.random.default_rng(7)
        for k in range(8):
            a = rng.normal(0, 3, (40 + 9 * k, 35 + 5 * k)).astype(np.float32)
            a[rng.random(a.shape) < 0.2 + 0.08 * k] = np.nan
            a[3:25, 0:20] = np.nan
            for rounds in (ahn.FILL_ROUNDS, 4):
                for got, want in ((ahn.fill(a, rounds), grid_fill(a, rounds)),
                                  (ahn.fill_low(a, rounds), grid_fill_low(a, rounds)),
                                  (ahn.fill_low(-a, rounds), grid_fill_low(-a, rounds))):
                    np.testing.assert_array_equal(np.nan_to_num(got, nan=-1e9), np.nan_to_num(want, nan=-1e9))


class AhnDemTest(unittest.TestCase):
    class Base:
        def sample(self, lat, lon, zoom, progress=None):
            return np.full(np.shape(lat), 0.4)          # the base DEM: the water's level

    class Fake:
        def sample(self, lat, lon):
            n = np.shape(lat)
            return (np.array([2.0, 2.0, np.nan])[:n[0]], np.array([False, True, False])[:n[0]],
                    np.array([2.0, 2.0, np.nan])[:n[0]])

    def test_ground_water_and_no_data(self):
        d = ahn.AhnDem(self.Base(), self.Fake())
        z = d.sample(np.array([51.9, 51.9, 51.9]), np.array([4.4, 4.4, 4.4]), 15)
        np.testing.assert_allclose(z, [2.0, 0.4, 0.4], err_msg="ground; a hole over water: the base; no AHN: the base")
        np.testing.assert_allclose(d.sample(np.array([51.9]), np.array([4.4]), 10), [0.4], err_msg="far: the base")


if __name__ == "__main__":
    unittest.main()
