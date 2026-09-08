"""Unit tests for /perf rendering of global model health (turn_metrics).

The session block is covered indirectly by summarize(); these tests pin the
new global capability/reliability block and the fact that it is optional.
"""
from __future__ import annotations

import pytest

from agent.metrics import model_reliability as mr
from agent.metrics import turn_metrics


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point capability/reliability at a scratch DB and drop cached conns."""
    monkeypatch.setattr(mr, "_DB_PATH", tmp_path / "model_reliability.db")
    monkeypatch.setattr(mr, "_schema_ready", False)
    if hasattr(mr._local, "conn"):
        del mr._local.conn
    yield
    if hasattr(mr._local, "conn"):
        del mr._local.conn


def test_health_lines_empty_without_data():
    assert turn_metrics._model_health_lines("gpu-a") == []


def test_health_lines_report_capability_and_reliability():
    for _ in range(9):
        mr.record_capability("gpu-a", "ok")
    mr.record_capability("gpu-a", "schema_error")
    mr.record_outcome("gpu-a", "success")

    lines = turn_metrics._model_health_lines("gpu-a")
    assert any("malformed calls (10.0%)" in ln for ln in lines)
    assert any("reliability (gpu-a, 24h)" in ln for ln in lines)


def test_perf_without_session_dir_still_reports_health():
    mr.record_capability("gpu-a", "ok")
    out = turn_metrics.run_perf_command(None, "gpu-a")
    assert "no active session side-log" in out
    assert "capability (gpu-a, 7d)" in out


def test_perf_without_entry_name_has_no_health_block():
    mr.record_capability("gpu-a", "ok")
    assert "capability" not in turn_metrics.run_perf_command(None)
