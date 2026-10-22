#!/usr/bin/env python3
"""NDW's register of the Dutch road signs (wegkenmerken, open data, CC-BY 4.0) round a kunstwerk: each sign where it
stands (GPS, from the photo survey), its RVV code, whether it stands beside the road (L) or over it (B), and the way the
traffic it speaks to drives (bearing: a compass quarter, toward the junction for a give-way sign). The user,
2026-10-22: "op de kaart kan je precies zien op gps data waar elke paal bordje etc moet staan".

The national file (~1.2 GB, refreshed monthly by NDW) is fetched once a month into the cache and read as a stream.

    python3 tools/ndw_signs.py ../design/kunstwerken/droogleever/ref [--reach 260]

Writes ref/ndw/signs.json: [GeoJSON Feature, ...] within reach (m) of the kunstwerk's origin (ref.json origin_wgs).
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request

URL = "https://data.ndw.nu/api/rest/static-road-data/traffic-signs/v4/current-state"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth", "ndw", "traffic-signs-current-state.json")
MAX_AGE_S = 31 * 86400
CHUNK = 1 << 24             # characters read at a time
COORDS = re.compile(r'"coordinates":\[([0-9.\-]+),([0-9.\-]+)')


def national(path=CACHE, url=URL):
    """The national file, fetched when missing or older than MAX_AGE_S."""
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > MAX_AGE_S:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        req = urllib.request.Request(url, headers={"User-Agent": "beamng-earth-tools/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, path)
    return path


def around(path, lat0, lon0, reach_m):
    """The features within reach_m of (lat0, lon0), read chunk by chunk (the file is one line)."""
    dlat = reach_m / 111320.0
    dlon = reach_m / (111320.0 * math.cos(math.radians(lat0)))
    dec = json.JSONDecoder()
    out = []
    tail = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK)
            buf = tail + chunk
            start = 0
            while True:
                i = buf.find('{"type":"Feature"', start)
                if i < 0:
                    tail = buf[start:] if chunk else ""
                    break
                m = COORDS.search(buf, i, i + 400)
                try:
                    feat, end = dec.raw_decode(buf, i)
                except ValueError:
                    tail = buf[i:]                      # (cut by the chunk: read on)
                    break
                if m and abs(float(m.group(2)) - lat0) <= dlat and abs(float(m.group(1)) - lon0) <= dlon:
                    out.append(feat)
                start = end
            if not chunk:
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref")
    ap.add_argument("--reach", type=float, default=260.0, help="m round the origin")
    ap.add_argument("--source", default=None, help="a national file already fetched")
    a = ap.parse_args(argv)
    ref = json.load(open(os.path.join(a.ref, "ref.json"), encoding="utf-8"))
    lat0, lon0 = ref["origin_wgs"]
    feats = around(a.source or national(), lat0, lon0, a.reach)
    os.makedirs(os.path.join(a.ref, "ndw"), exist_ok=True)
    dst = os.path.join(a.ref, "ndw", "signs.json")
    json.dump(feats, open(dst, "w", encoding="utf-8"))
    print(f"{dst}: {len(feats)} signs")


if __name__ == "__main__":
    main()
