"""Per-tier model-call counters (metrics/model_calls.py)."""
from types import SimpleNamespace as N

import pytest

from agent.metrics import model_calls as mc


@pytest.fixture(autouse=True)
def _reset():
    mc.reset_round()
    mc._session.clear()
    yield
    mc.reset_round()
    mc._session.clear()


def _entry(**kw):
    base = dict(tier="", local=False, base_url="https://api.x/v1", model="m",
                cost_in_per_1k=0.0, cost_out_per_1k=0.0)
    base.update(kw)
    return N(**base)


def test_record_entry_classifies_tiers():
    mc.record_entry(_entry(local=True, base_url="http://localhost:8080/v1"))  # local
    mc.record_entry(_entry())                                                  # free (cloud, no price)
    mc.record_entry(_entry(tier="bundled"))                                    # bundled (explicit)
    mc.record_entry(_entry(cost_in_per_1k=1.0))                                # paid
    assert mc.round_counts() == {"local": 1, "free": 1, "bundled": 1, "paid": 1}


def test_reset_round_keeps_session_total():
    mc.record("local")
    mc.record("paid")
    mc.reset_round()
    assert mc.round_counts() == {}
    assert mc.session_counts() == {"local": 1, "paid": 1}


def test_format_line_order_and_empty():
    assert mc.format_line({}) == ""
    mc.record("paid")
    mc.record("local")
    mc.record("local")
    # display order is local/free/bundled/paid regardless of insertion order
    assert mc.format_line(mc.round_counts()) == "models: 3 calls (local=2 paid=1)"


def test_record_main_uses_config_llm_entry_tier():
    entry = _entry(tier="bundled", model="big", base_url="https://api.x/v1")
    cfg = N(llm=N(base_url="https://api.x/v1", model="big"),
            model_entries={"big": entry})
    mc.record_main(cfg)
    assert mc.round_counts() == {"bundled": 1}


def test_format_line_with_duration():
    mc.record("local")
    line = mc.format_line(mc.round_counts(), duration=8.24)
    assert line == "models: 1 call (local=1) in 8.2s"
    # zero/None duration → no suffix
    assert mc.format_line(mc.round_counts()) == "models: 1 call (local=1)"
    assert mc.format_line(mc.round_counts(), duration=0) == "models: 1 call (local=1)"


def test_format_duration_units():
    assert mc.format_duration(8.24) == "8.2s"
    assert mc.format_duration(64) == "1m 04s"
    assert mc.format_duration(3725) == "1h 02m"


def test_round_duration_and_detail_offsets():
    mc.reset_round()
    assert mc.round_duration() >= 0
    mc.record("local", role="main", model="m1")
    d = mc.round_detail()
    assert len(d) == 1 and d[0]["t"] >= 0
    assert d[0]["role"] == "main" and d[0]["tier"] == "local"


def test_run_command_reset():
    mc.record("free")
    assert "free=1" in mc.run_modelcalls_command("")
    assert "reset" in mc.run_modelcalls_command("reset").lower()
    assert mc.session_counts() == {}
