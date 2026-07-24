"""Unit tests for agent/core/model_state_store.py."""
from __future__ import annotations

from agent.core.model_state_store import load_disabled, save_disabled


def test_load_missing_returns_empty(tmp_path):
    assert load_disabled(str(tmp_path)) == set()


def test_save_load_round_trip(tmp_path):
    save_disabled(str(tmp_path), {"gpu-a", "gpu-b"})
    assert load_disabled(str(tmp_path)) == {"gpu-a", "gpu-b"}


def test_save_overwrites_previous(tmp_path):
    save_disabled(str(tmp_path), {"gpu-a"})
    save_disabled(str(tmp_path), {"gpu-b"})
    assert load_disabled(str(tmp_path)) == {"gpu-b"}


def test_load_corrupt_json_returns_empty(tmp_path):
    p = tmp_path / "model_state.json"
    p.write_text("{not valid json")
    assert load_disabled(str(tmp_path)) == set()


def test_save_creates_parent_dir(tmp_path):
    target = tmp_path / "nested" / "agent_dir"
    save_disabled(str(target), {"x"})
    assert load_disabled(str(target)) == {"x"}
