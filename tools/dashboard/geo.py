"""Coordinates for the kunstwerk dashboard.

The kunstwerk's local frame (every file under design/kunstwerken/<name>/ref and server/data/kunstwerken/<name>.kunstwerk.json):

- x, y: metres in RD New (EPSG:28992) minus the kunstwerk's origin_rd (ref/ref.json; droogleever: 91597, 435902):
  x east, y north (RD grid north; it differs from true north by under a degree here);
- z: metres above NAP (the AHN's and the model's heights).

WGS84 <-> RD is server/ahn.py's (wgs_to_rd: the RD polynomial, < 1 m; rd_to_wgs its Newton inverse), the same the
pipeline uses (tools/kunstwerk_placement.py local()), so a point converts here exactly as there.

A facing is a unit vector [fx, fy] in the local frame: the way a sign's face looks (toward the traffic it speaks to).
A heading is that facing as a compass bearing in degrees: 0 = north (+y), 90 = east (+x), clockwise.
The browser draws the local frame as is (three.js with z up).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _ahn(repo):
    server = os.path.join(repo, "server")
    if server not in sys.path:
        sys.path.insert(0, server)
    import ahn
    return ahn


class Frame:
    """A kunstwerk's local frame: origin_rd [x, y] (RD metres)."""

    def __init__(self, origin_rd, repo=None):
        self.origin = (float(origin_rd[0]), float(origin_rd[1]))
        self._ahn = _ahn(repo or os.path.normpath(os.path.join(HERE, "..", "..")))

    def from_wgs(self, lat, lon):
        rx, ry = self._ahn.wgs_to_rd(lat, lon)
        return float(rx) - self.origin[0], float(ry) - self.origin[1]

    def to_wgs(self, x, y):
        lat, lon = self._ahn.rd_to_wgs(x + self.origin[0], y + self.origin[1])
        return float(lat), float(lon)


def heading_of(facing):
    """A facing [fx, fy] as a compass heading in degrees [0, 360)."""
    return math.degrees(math.atan2(float(facing[0]), float(facing[1]))) % 360.0


def facing_of(heading):
    """A compass heading (degrees) as a unit facing [fx, fy]."""
    b = math.radians(float(heading))
    return [math.sin(b), math.cos(b)]


def facing_from_bearing(bearing_deg):
    """NDW's register bearing (the way its traffic drives) as the sign's facing: toward that traffic
    (tools/kunstwerk_placement.py ndw_signs)."""
    b = math.radians(float(bearing_deg or 0.0))
    return [-math.sin(b), -math.cos(b)]
