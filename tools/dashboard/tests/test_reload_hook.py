"""server/kunstwerken.py's opt-in reload (BNE_KUNSTWERK_RELOAD), which Push to game relies on."""
import json
import os

import pytest

import kunstwerken as kwmod


class _Stub:
    def __init__(self, d):
        self.name = d["name"]


@pytest.fixture
def kwdir(tmp_path, monkeypatch):
    monkeypatch.setattr(kwmod, "DIR", str(tmp_path))
    monkeypatch.setattr(kwmod, "Kunstwerk", _Stub)
    monkeypatch.setattr(kwmod, "_all", None)
    monkeypatch.setattr(kwmod, "_checked", 0.0)
    monkeypatch.setattr(kwmod, "RELOAD_EVERY_S", 0.0)
    return tmp_path


def _write(path, name, mtime):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name}, f)
    os.utime(path, (mtime, mtime))


def test_off_by_default(kwdir, monkeypatch):
    monkeypatch.delenv(kwmod.RELOAD_ENV, raising=False)
    p = str(kwdir / "a.kunstwerk.json")
    _write(p, "one", 1000)
    assert [k.name for k in kwmod.all_kunstwerken()] == ["one"]
    _write(p, "two", 2000)
    assert [k.name for k in kwmod.all_kunstwerken()] == ["one"]


def test_reload_when_changed(kwdir, monkeypatch):
    monkeypatch.setenv(kwmod.RELOAD_ENV, "1")
    p = str(kwdir / "a.kunstwerk.json")
    _write(p, "one", 1000)
    first = kwmod.all_kunstwerken()
    assert [k.name for k in first] == ["one"]
    assert kwmod.all_kunstwerken() is first                  # (unchanged: the same list)
    _write(p, "two", 2000)
    assert [k.name for k in kwmod.all_kunstwerken()] == ["two"]
    with open(p, "w", encoding="utf-8") as f:                 # (half written: the old ones kept)
        f.write('{"name": "thr')
    os.utime(p, (3000, 3000))
    assert [k.name for k in kwmod.all_kunstwerken()] == ["two"]
