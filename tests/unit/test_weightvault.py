"""Tests for the weight vault (agent.security.weightvault)."""
from __future__ import annotations

import os
import time
import types

from agent.security import weightvault as wv


def _cfg(tmp_path):
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent")
    )


def _weight(tmp_path, name="model.gguf", data=b"GGUF" + b"\x00" * 4096):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_pin_records_sha_and_size(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    res = wv.pin(cfg, str(p), source="huggingface://foo/bar")
    assert res["ok"]
    assert len(res["sha256"]) == 64
    assert res["size"] == p.stat().st_size


def test_pin_missing_file(tmp_path):
    res = wv.pin(_cfg(tmp_path), str(tmp_path / "nope.gguf"))
    assert "error" in res


def test_verify_clean(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    wv.pin(cfg, str(p))
    res = wv.verify(cfg)
    assert res["ok"]
    assert str(p.resolve()) in res["verified"]


def test_verify_detects_content_swap(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    wv.pin(cfg, str(p))
    p.write_bytes(b"DIFFERENT WEIGHTS" + b"\x00" * 4096)  # same-ish size, diff content
    res = wv.verify(cfg)
    assert not res["ok"]
    assert str(p.resolve()) in res["mismatched"]


def test_verify_detects_missing(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    wv.pin(cfg, str(p))
    os.remove(p)
    res = wv.verify(cfg)
    assert str(p.resolve()) in res["missing"]
    assert not res["ok"]


def test_quickcheck_size_change(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    wv.pin(cfg, str(p))
    assert wv.quickcheck(cfg)["ok"]
    p.write_bytes(b"x" * 100)  # size changed
    res = wv.quickcheck(cfg)
    assert not res["ok"]
    assert str(p.resolve()) in res["changed"]


def test_unpinned_is_ok(tmp_path):
    assert wv.verify(_cfg(tmp_path))["ok"]
    assert wv.quickcheck(_cfg(tmp_path))["ok"]
    assert wv.warn_if_drift(_cfg(tmp_path)) is None


def test_warn_if_drift(tmp_path):
    cfg = _cfg(tmp_path)
    p = _weight(tmp_path)
    wv.pin(cfg, str(p))
    assert wv.warn_if_drift(cfg) is None
    os.remove(p)
    msg = wv.warn_if_drift(cfg)
    assert msg and "WEIGHT VAULT WARNING" in msg


def test_human_sizes():
    assert wv._human(512) == "512B"
    assert wv._human(2048) == "2.0KB"
    assert wv._human(5 * 1024**3) == "5.0GB"
