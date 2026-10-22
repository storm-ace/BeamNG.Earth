"""The exported kunstwerk (server/data/kunstwerken/<name>.kunstwerk.json, tools/kunstwerk_export.py) for the browser:
one binary, its triangles grouped by material.

    [u32 header length][header JSON, padded to 4 bytes][positions f32 (n, 3)][indices u32 (m, 3)]

header: {"materials", "groups": [{"material", "start", "count"} (in indices)], "nverts", "ntris", "origin_rd",
"origin_wgs", "signals", "wires", "mtime"}. Positions in the local frame (geo.py), as the file has them.
"""
import json
import os
import struct
import threading

import numpy as np

_lock = threading.Lock()
_cache = {}


def load(path):
    """{"head": {...}, "bin": bytes} of the kunstwerk file at path (cached by mtime), or None when there is none."""
    if not os.path.exists(path):
        return None
    key = os.stat(path).st_mtime_ns
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == key:
            return hit[1]
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    val = build(d, key)
    with _lock:
        _cache[path] = (key, val)
    return val


def build(d, mtime=0):
    """The binary of a kunstwerk dict (pure)."""
    P = np.asarray(d.get("positions", []), np.float32).reshape(-1, 3)
    T = np.asarray(d.get("tris", []), np.uint32).reshape(-1, 3)
    TM = np.asarray(d.get("tri_material", np.zeros(len(T))), np.int64)
    order = np.argsort(TM, kind="stable")
    T, TM = T[order], TM[order]
    mats = d.get("materials", [])
    groups = []
    for m in np.unique(TM):
        idx = np.flatnonzero(TM == m)
        groups.append({"material": mats[m] if m < len(mats) else str(m), "start": int(idx[0]) * 3,
                       "count": int(len(idx)) * 3})
    head = {"name": d.get("name"), "materials": mats, "groups": groups, "nverts": int(len(P)), "ntris": int(len(T)),
            "origin_rd": d.get("origin_rd"), "origin_wgs": d.get("origin_wgs"), "signals": d.get("signals", []),
            "wires": d.get("wires", []), "mtime": mtime}
    hb = json.dumps(head).encode()
    hb += b" " * (-len(hb) % 4)
    blob = struct.pack("<I", len(hb)) + hb + P.tobytes() + T.tobytes()
    return {"head": head, "bin": blob}


def head(path):
    m = load(path)
    return m["head"] if m else {}
