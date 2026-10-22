"""ref/overrides.json: the hand fixes made in the dashboard, a list keyed by each object's stable id
(tools/kunstwerk_placement.py item_id; applied by its apply_overrides in tools/kunstwerk_solid.py).

    [{"id": "sign:C2@12.3,-4.5", "at": [x, y], "heading": 90.0, "code": "C2", "approach": 3, "remove": false,
      "note": "..."}]
"""
import json
import os
import threading

FIELDS = ("at", "heading", "facing", "code", "approach", "remove", "note")
_lock = threading.Lock()


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: not a list")
    return data


def save(path, items):
    """Written whole (a temporary file renamed over it), sorted by id; the previous one kept as .bak."""
    items = sorted(items, key=lambda o: o["id"])
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)
        f.write("\n")
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)
    return items


def clean(o):
    """An override as stored: its id and the fields it sets (empty ones left out); ValueError when malformed."""
    if not isinstance(o, dict) or not isinstance(o.get("id"), str) or not o["id"]:
        raise ValueError("an override needs an id")
    out = {"id": o["id"]}
    for k in FIELDS:
        v = o.get(k)
        if v is None or v == "" or (k == "remove" and not v):
            continue
        if k in ("at", "facing"):
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(f"{k}: [x, y]")
            v = [round(float(v[0]), 3), round(float(v[1]), 3 if k == "at" else 4)]
        elif k == "heading":
            v = round(float(v) % 360.0, 2)
        elif k == "approach":
            v = int(v)
        elif k == "remove":
            v = True
        elif k == "code":
            v = [str(c) for c in v] if isinstance(v, (list, tuple)) else str(v).strip()
        out[k] = v
    return out


def upsert(path, o):
    o = clean(o)
    with _lock:
        items = [x for x in load(path) if x.get("id") != o["id"]]
        if len(o) > 1:                                  # (nothing but the id: the override dropped)
            items.append(o)
        return save(path, items)


def delete(path, id_):
    with _lock:
        items = load(path)
        rest = [x for x in items if x.get("id") != id_]
        if len(rest) == len(items):
            return None
        return save(path, rest)
