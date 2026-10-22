import json
import os

import pytest

import kunstwerk_placement as kp
from dashboard import overrides
from dashboard.data import Data

ITEMS = [{"module": "sign", "code": "C2", "at": [1.04, 2.0], "facing": [0, 1]},
         {"module": "traffic_light", "role": "primary", "at": [5.0, 5.0], "facing": [1, 0]},
         {"module": "traffic_light", "role": "primary", "at": [5.0, 5.0], "facing": [-1, 0]}]


def test_item_ids():
    out = kp.apply_overrides(ITEMS, [])
    assert [o["id"] for o in out] == ["sign:C2@1.0,2.0", "traffic_light:primary@5.0,5.0",
                                      "traffic_light:primary@5.0,5.0#2"]
    assert kp.item_id({"module": "sign", "code": ["B6", "OB01"], "at": [0, -0.04]}) == "sign:B6+OB01@0.0,-0.0"
    assert "id" not in ITEMS[0]                                   # (pure)


def test_apply_overrides():
    aps = [{"name": "a", "left": [0, 0], "right": [0, 10]}]     # (a way in westward: its lights face east)
    ovs = [{"id": "sign:C2@1.0,2.0", "at": [3, 4], "heading": 90, "code": "C3"},
           {"id": "traffic_light:primary@5.0,5.0", "approach": 0},
           {"id": "traffic_light:primary@5.0,5.0#2", "remove": True}]
    out = kp.apply_overrides(ITEMS, ovs, aps)
    assert len(out) == 2
    assert out[0]["at"] == [3.0, 4.0] and out[0]["code"] == "C3"
    assert out[0]["facing"] == pytest.approx([1.0, 0.0], abs=1e-4)
    assert out[1]["facing"] == pytest.approx([1.0, 0.0], abs=1e-4) and out[1]["approach"] == 0


def test_round_trip(tmp_path):
    p = str(tmp_path / "ref" / "overrides.json")
    assert overrides.load(p) == []
    overrides.upsert(p, {"id": "b", "at": [1.23456, 2], "heading": 370, "note": "x", "code": ""})
    overrides.upsert(p, {"id": "a", "remove": True, "approach": "2"})
    items = overrides.load(p)
    assert items == [{"id": "a", "approach": 2, "remove": True},
                     {"id": "b", "at": [1.235, 2.0], "heading": 10.0, "note": "x"}]
    overrides.upsert(p, {"id": "b", "heading": 45})           # (replaces b whole)
    assert overrides.load(p)[1] == {"id": "b", "heading": 45.0}
    assert os.path.exists(p + ".bak")
    assert overrides.delete(p, "a") is not None and [o["id"] for o in overrides.load(p)] == ["b"]
    assert overrides.delete(p, "nope") is None
    overrides.upsert(p, {"id": "b"})                          # (nothing left to set: dropped)
    assert overrides.load(p) == []
    with open(p, encoding="utf-8") as f:
        assert json.load(f) == []


@pytest.mark.parametrize("bad", [{}, {"id": ""}, {"id": "x", "at": [1]}, {"id": "x", "heading": "noord"}])
def test_clean_rejects(bad):
    with pytest.raises(ValueError):
        overrides.clean(bad)


def test_pipeline_view_round_trip(world):
    """An override saved is what the dashboard shows and what kunstwerk_solid would place."""
    d = Data(world)
    ours = {o["id"]: o for o in d.ours()[0]}
    # placement_fix moved the D2 (20, 20) to (21, 22): its id is where the fix put it
    assert "sign:D2@21.0,22.0" in ours
    assert set(ours) == d.known_ids()
    overrides.upsert(world.overrides_path, {"id": "sign:C2@-3.0,7.0", "at": [-2, 8], "heading": 180})
    overrides.upsert(world.overrides_path, {"id": "finger_post:@1.0,2.0", "remove": True})
    ours = {o["id"]: o for o in d.ours()[0]}
    c2 = ours["sign:C2@-3.0,7.0"]
    assert c2["at"] == [-2.0, 8.0] and c2["heading"] == pytest.approx(180)
    assert c2["props"]["base_at"] == [-3.04, 7.0] and c2["props"]["override"]["heading"] == 180.0
    assert ours["finger_post:@1.0,2.0"]["props"]["removed"] is True
