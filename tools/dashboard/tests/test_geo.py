import math

import pytest

from dashboard.geo import Frame, facing_from_bearing, facing_of, heading_of

ORIGIN_RD = [91597, 435902]                     # droogleever's ref.json
ORIGIN_WGS = [51.90800136211493, 4.4658091118043775]


def test_origin_is_local_zero():
    f = Frame(ORIGIN_RD)
    x, y = f.from_wgs(*ORIGIN_WGS)
    assert abs(x) < 0.5 and abs(y) < 0.5        # (ref.json's origin_wgs: the RD origin through the < 1 m polynomial)


@pytest.mark.parametrize("x,y", [(0.0, 0.0), (120.5, -80.25), (-150.0, 160.0)])
def test_round_trip(x, y):
    f = Frame(ORIGIN_RD)
    lat, lon = f.to_wgs(x, y)
    x2, y2 = f.from_wgs(lat, lon)
    assert abs(x2 - x) < 1e-3 and abs(y2 - y) < 1e-3


def test_axes_east_north():
    f = Frame(ORIGIN_RD)
    lat0, lon0 = f.to_wgs(0, 0)
    lat_e, lon_e = f.to_wgs(100, 0)
    lat_n, lon_n = f.to_wgs(0, 100)
    assert lon_e > lon0 and abs(lat_e - lat0) < 2e-5       # +x east
    assert lat_n > lat0 and abs(lon_n - lon0) < 3e-5       # +y north
    assert abs((lat_n - lat0) * 111_200 - 100) < 1.0       # metres


def test_matches_the_pipeline():
    """The same conversion as tools/kunstwerk_placement.py local()."""
    import ahn
    f = Frame(ORIGIN_RD)
    rx, ry = ahn.wgs_to_rd(51.9074729, 4.4654893)
    assert f.from_wgs(51.9074729, 4.4654893) == pytest.approx((rx - 91597, ry - 435902))


@pytest.mark.parametrize("facing,heading", [([0, 1], 0), ([1, 0], 90), ([0, -1], 180), ([-1, 0], 270)])
def test_heading(facing, heading):
    assert heading_of(facing) == pytest.approx(heading)
    f = facing_of(heading)
    assert f[0] == pytest.approx(facing[0], abs=1e-9) and f[1] == pytest.approx(facing[1], abs=1e-9)


def test_ndw_bearing_faces_its_traffic():
    # traffic driving north (bearing 0) sees a sign facing south
    f = facing_from_bearing(0)
    assert f[0] == pytest.approx(0, abs=1e-9) and f[1] == pytest.approx(-1)
    assert heading_of(facing_from_bearing(90)) == pytest.approx(270)
    assert math.hypot(*facing_from_bearing(33)) == pytest.approx(1)
