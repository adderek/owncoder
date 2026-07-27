"""Unit tests for agent/metrics/model_history.py (long-term throughput log)."""
from __future__ import annotations

import time

import pytest

from agent.metrics import model_history as mh


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(mh, "_DB_PATH", tmp_path / "model_throughput.db")
    monkeypatch.setattr(mh, "_schema_ready", False)
    if hasattr(mh._local, "conn"):
        del mh._local.conn
    yield
    if hasattr(mh._local, "conn"):
        del mh._local.conn


def _insert(entry, ts, in_tokens=0, out_tokens=0, gen=0.0, ttft=0.0):
    """Insert a sample at an explicit timestamp (record_sample stamps now())."""
    conn = mh._conn()
    conn.execute("INSERT INTO samples (entry_name, ts, in_tokens, out_tokens, gen_sec, ttft) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (entry, ts, in_tokens, out_tokens, gen, ttft))
    conn.commit()


def test_empty_history():
    assert mh.known_entries() == []
    assert mh.window_summary("gpu-a")["calls"] == 0
    assert all(b["out_tps"] is None for b in mh.series("gpu-a"))


def test_record_and_summarize():
    mh.record_sample("gpu-a", out_tokens=100, gen_sec=2.0, in_tokens=4000, ttft=0.5)
    mh.record_sample("gpu-a", out_tokens=100, gen_sec=4.0, in_tokens=4000, ttft=0.5)
    s = mh.window_summary("gpu-a")
    assert s["calls"] == 2
    # Token-weighted: 200 tokens over 6s, not the mean of 50 and 25.
    assert s["out_tps"] == pytest.approx(33.3, abs=0.1)
    assert s["in_tps"] == pytest.approx(8000.0, abs=0.1)
    assert (s["out_tps_min"], s["out_tps_max"]) == (25.0, 50.0)


def test_degenerate_sample_is_dropped():
    mh.record_sample("gpu-a", out_tokens=0, gen_sec=0.0, in_tokens=0, ttft=0.0)
    assert mh.window_summary("gpu-a")["calls"] == 0


def test_input_only_sample_is_kept():
    """A reply too short to time decode still measured a real prefill."""
    mh.record_sample("gpu-a", out_tokens=3, gen_sec=0.0, in_tokens=2000, ttft=0.4)
    s = mh.window_summary("gpu-a")
    assert s["calls"] == 1
    assert s["in_tps"] == 5000.0
    assert s["out_tps"] is None


def test_series_buckets_by_time():
    now = int(time.time())
    _insert("gpu-a", now - 20 * 3600, out_tokens=100, gen=1.0)   # ~fast, older
    _insert("gpu-a", now - 60, out_tokens=100, gen=10.0)         # slow, recent
    buckets = mh.series("gpu-a", hours=24, buckets=24)
    assert len(buckets) == 24
    filled = [b for b in buckets if b["calls"]]
    assert len(filled) == 2
    assert filled[0]["out_tps"] == 100.0
    assert filled[-1]["out_tps"] == 10.0
    # Quiet stretches stay in the series as gaps, keeping the x-axis linear.
    assert any(b["calls"] == 0 and b["out_tps"] is None for b in buckets)


def test_window_excludes_older_samples():
    now = int(time.time())
    _insert("gpu-a", now - 40 * 3600, out_tokens=100, gen=1.0)
    _insert("gpu-a", now - 60, out_tokens=100, gen=2.0)
    assert mh.window_summary("gpu-a", hours=24)["calls"] == 1
    assert mh.window_summary("gpu-a", hours=168)["calls"] == 2


def test_entries_are_separate_and_listed_by_volume():
    mh.record_sample("gpu-a", out_tokens=100, gen_sec=1.0)
    mh.record_sample("cloud-b", out_tokens=100, gen_sec=1.0)
    mh.record_sample("cloud-b", out_tokens=100, gen_sec=1.0)
    assert mh.known_entries() == ["cloud-b", "gpu-a"]
    assert mh.window_summary("gpu-a")["calls"] == 1


def test_update_stats_feeds_history(tmp_path, monkeypatch):
    from agent.metrics import model_stats as ms
    monkeypatch.setattr(ms, "_stats_path", lambda: tmp_path / "model_stats.json")
    ms.update_stats("gpu-a", 100, 2.0, in_tokens=1000, ttft=0.5)
    assert mh.window_summary("gpu-a")["calls"] == 1
