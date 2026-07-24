"""Unit tests for agent/metrics/model_reliability.py."""
from __future__ import annotations

import pytest

from agent.metrics import model_reliability as mr


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the module at a scratch DB and drop any cached connection so
    each test starts clean."""
    monkeypatch.setattr(mr, "_DB_PATH", tmp_path / "model_reliability.db")
    monkeypatch.setattr(mr, "_schema_ready", False)
    if hasattr(mr._local, "conn"):
        del mr._local.conn
    yield
    if hasattr(mr._local, "conn"):
        del mr._local.conn


def test_summary_empty_when_no_calls():
    s = mr.reliability_summary("gpu-a")
    assert s == {"success": 0, "failure": 0, "rate_limited": 0, "total": 0, "success_rate": None}


def test_record_and_summarize_success_and_failure():
    mr.record_outcome("gpu-a", "success")
    mr.record_outcome("gpu-a", "success")
    mr.record_outcome("gpu-a", "failure")
    s = mr.reliability_summary("gpu-a")
    assert s["success"] == 2
    assert s["failure"] == 1
    assert s["total"] == 3
    assert s["success_rate"] == pytest.approx(2 / 3, abs=1e-3)


def test_rate_limited_excluded_from_success_rate_denominator():
    mr.record_outcome("gpu-a", "success")
    mr.record_outcome("gpu-a", "rate_limited")
    mr.record_outcome("gpu-a", "rate_limited")
    s = mr.reliability_summary("gpu-a")
    assert s["rate_limited"] == 2
    assert s["total"] == 3
    # success_rate is judged over success+failure only, not rate_limited
    assert s["success_rate"] == 1.0


def test_entries_are_isolated_by_name():
    mr.record_outcome("gpu-a", "success")
    mr.record_outcome("gpu-b", "failure")
    assert mr.reliability_summary("gpu-a")["success"] == 1
    assert mr.reliability_summary("gpu-b")["failure"] == 1


def test_window_hours_excludes_old_rows():
    import time
    conn = mr._conn()
    old_ts = int(time.time()) - 100 * 3600
    conn.execute(
        "INSERT INTO calls (entry_name, ts, outcome, role) VALUES (?, ?, ?, ?)",
        ("gpu-a", old_ts, "success", ""),
    )
    conn.commit()
    mr.record_outcome("gpu-a", "success")
    s = mr.reliability_summary("gpu-a", window_hours=24)
    assert s["total"] == 1


def test_invalid_outcome_is_ignored():
    mr.record_outcome("gpu-a", "bogus")
    assert mr.reliability_summary("gpu-a")["total"] == 0


def test_concurrent_writers_do_not_lose_rows():
    """Two threads recording outcomes concurrently should not lose writes —
    the reason for using WAL-mode sqlite over atomic-replace JSON."""
    import threading

    def _write(n):
        for _ in range(n):
            mr.record_outcome("gpu-a", "success")

    threads = [threading.Thread(target=_write, args=(20,)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mr.reliability_summary("gpu-a")["success"] == 100
