"""Unit tests for agent/metrics/model_stats.py throughput tracking."""
from __future__ import annotations

import pytest

from agent.metrics import model_stats as ms


@pytest.fixture(autouse=True)
def _isolated_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "_stats_path", lambda: tmp_path / "model_stats.json")


def test_no_data_returns_empty():
    assert ms.stats_for("gpu-a") == {}
    assert ms.get_tps("gpu-a") == 0.0


def test_output_sample_records_tps_and_totals():
    ms.update_stats("gpu-a", 100, 2.0)
    rec = ms.stats_for("gpu-a")
    assert rec["tps_ewma"] == 50.0
    assert rec["tps_last"] == 50.0
    assert rec["samples"] == 1
    assert rec["tokens_out"] == 100
    assert rec["tps_avg"] == 50.0


def test_input_sample_records_prefill_rate_and_ttft():
    ms.update_stats("gpu-a", 0, 0.0, in_tokens=4000, ttft=0.5)
    rec = ms.stats_for("gpu-a")
    assert rec["in_tps_ewma"] == 8000.0
    assert rec["in_samples"] == 1
    assert rec["ttft_ewma"] == 0.5
    assert rec["in_tps_avg"] == 8000.0
    # An input-only sample must not fabricate an output rate.
    assert "tps_ewma" not in rec


def test_short_reply_still_contributes_a_prefill_sample():
    """The _MIN_TOKENS guard gates the decode rate only — a two-word answer
    still measured a real prefill."""
    ms.update_stats("gpu-a", 3, 0.1, in_tokens=2000, ttft=0.4)
    rec = ms.stats_for("gpu-a")
    assert "tps_ewma" not in rec
    assert rec["in_tps_ewma"] == 5000.0


def test_ewma_blends_successive_samples():
    ms.update_stats("gpu-a", 100, 1.0)   # 100 tok/s
    ms.update_stats("gpu-a", 100, 2.0)   # 50 tok/s
    rec = ms.stats_for("gpu-a")
    assert rec["tps_last"] == 50.0
    assert 50.0 < rec["tps_ewma"] < 100.0
    assert rec["samples"] == 2
    # Cumulative average is over tokens/seconds, not a mean of the rates.
    assert rec["tps_avg"] == pytest.approx(200 / 3.0, abs=0.1)


def test_stats_for_accepts_a_shared_snapshot():
    ms.update_stats("gpu-a", 100, 2.0)
    snap = ms.load_stats()
    assert ms.stats_for("gpu-a", snap)["tps_ewma"] == 50.0
    assert ms.stats_for("missing", snap) == {}


def test_entries_are_kept_separate():
    ms.update_stats("gpu-a", 100, 1.0)
    ms.update_stats("cloud-b", 100, 10.0)
    assert ms.get_tps("gpu-a") == 100.0
    assert ms.get_tps("cloud-b") == 10.0
