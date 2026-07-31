"""Session-health metrics and the budget they drive (P3).

tool_call_rate / token_usage_rate describe what a turn is *spending*, as opposed
to the error/null/dup fractions that describe whether it is making progress. The
budget reacts to waste_rate so a turn full of empty or repeated tool results
sheds them before it reaches the context ceiling.
"""
import pytest

from agent.config.models import Config
from agent.core import context_budget as cb
from agent.core.confidence import ConfidenceMonitor, ConfidenceSignal
from agent.core.context_budget import health_adjusted_budget, input_token_budget


def _monitor(**kw):
    kw.setdefault("window", 8)
    return ConfidenceMonitor(**kw)


def _cfg(ctx_window=32768, max_output_tokens=4096):
    c = Config()
    c.llm.ctx_window = ctx_window
    c.llm.max_output_tokens = max_output_tokens
    return c


# ── cost rates ────────────────────────────────────────────────────────────────

class TestCostRates:
    def test_zero_before_any_iteration_completes(self):
        m = _monitor()
        m.observe_result("x" * 400, is_error=False)
        sig = m.signal()
        assert sig.tool_call_rate == 0.0
        assert sig.token_usage_rate == 0.0

    def test_calls_per_iteration(self):
        m = _monitor()
        for _ in range(3):
            m.observe_result("unique-a" + "x" * 100, is_error=False)
        m.tick_iter()
        assert m.signal().tool_call_rate == 3.0

    def test_rate_is_averaged_across_iterations(self):
        m = _monitor()
        for _ in range(4):
            m.observe_result("a" * 100, is_error=False)
        m.tick_iter()
        m.observe_result("b" * 100, is_error=False)
        m.tick_iter()
        assert m.signal().tool_call_rate == 2.5

    def test_token_rate_approximates_four_chars_per_token(self):
        m = _monitor()
        m.observe_result("x" * 400, is_error=False)
        m.tick_iter()
        assert m.signal().token_usage_rate == 100.0

    def test_idle_iteration_counts_as_zero(self):
        m = _monitor()
        for _ in range(2):
            m.observe_result("x" * 100, is_error=False)
        m.tick_iter()
        m.tick_iter()             # a round with no tool calls
        assert m.signal().tool_call_rate == 1.0

    def test_window_bounds_the_history(self):
        m = _monitor(window=2)
        for _ in range(5):
            m.observe_result("x" * 100, is_error=False)
            m.tick_iter()
        assert len(m._calls_per_iter) == 2

    def test_rates_survive_the_empty_result_window(self):
        """signal() returns early when no results are recorded; rates still report."""
        m = _monitor()
        m.tick_iter()
        sig = m.signal()
        assert sig.score == 1.0 and sig.tool_call_rate == 0.0


# ── waste_rate ────────────────────────────────────────────────────────────────

class TestWasteRate:
    def _sig(self, null=0.0, dup=0.0):
        return ConfidenceSignal(score=1.0, error_rate=0.0, null_rate=null,
                                dup_rate=dup, triggered=False)

    def test_combines_null_and_dup(self):
        assert self._sig(null=0.25, dup=0.25).waste_rate == 0.5

    def test_clamped_to_one(self):
        assert self._sig(null=0.8, dup=0.8).waste_rate == 1.0

    def test_errors_do_not_count_as_waste(self):
        sig = ConfidenceSignal(score=0.1, error_rate=1.0, null_rate=0.0,
                               dup_rate=0.0, triggered=True)
        assert sig.waste_rate == 0.0

    def test_real_monitor_reports_duplicates_as_waste(self):
        m = _monitor()
        for _ in range(4):
            m.observe_result("identical result payload", is_error=False)
        m.tick_iter()
        assert m.signal().waste_rate > 0.0


# ── health-adjusted budget ────────────────────────────────────────────────────

class TestHealthAdjustedBudget:
    def test_no_signal_matches_the_plain_budget(self):
        cfg = _cfg()
        assert health_adjusted_budget(cfg, None) == input_token_budget(cfg)

    def test_healthy_turn_is_not_tightened(self):
        cfg = _cfg()
        sig = ConfidenceSignal(score=1.0, error_rate=0.0, null_rate=0.1,
                               dup_rate=0.1, triggered=False)
        assert health_adjusted_budget(cfg, sig) == input_token_budget(cfg)

    def test_at_the_floor_nothing_changes(self):
        cfg = _cfg()
        sig = ConfidenceSignal(score=1.0, error_rate=0.0, null_rate=0.5,
                               dup_rate=0.0, triggered=False)
        assert sig.waste_rate == cb._WASTE_FLOOR
        assert health_adjusted_budget(cfg, sig) == input_token_budget(cfg)

    def test_total_waste_tightens_by_the_maximum(self):
        cfg = _cfg()
        sig = ConfidenceSignal(score=0.0, error_rate=0.0, null_rate=0.5,
                               dup_rate=0.5, triggered=True)
        expected = int(input_token_budget(cfg) * (1.0 - cb._MAX_TIGHTEN))
        assert health_adjusted_budget(cfg, sig) == expected

    def test_tightening_scales_between_floor_and_full(self):
        cfg = _cfg()
        plain = input_token_budget(cfg)
        half = ConfidenceSignal(score=0.0, error_rate=0.0, null_rate=0.75,
                                dup_rate=0.0, triggered=False)
        got = health_adjusted_budget(cfg, half)
        assert got < plain
        assert got > int(plain * (1.0 - cb._MAX_TIGHTEN))

    def test_never_below_the_floor(self):
        cfg = _cfg(ctx_window=512, max_output_tokens=512)
        sig = ConfidenceSignal(score=0.0, error_rate=0.0, null_rate=1.0,
                               dup_rate=1.0, triggered=True)
        assert health_adjusted_budget(cfg, sig) >= 1024

    def test_tightening_is_bounded_so_compaction_is_not_forced_constantly(self):
        """Guards the P5 interaction: earlier compaction costs a cached prefix."""
        cfg = _cfg()
        sig = ConfidenceSignal(score=0.0, error_rate=0.0, null_rate=1.0,
                               dup_rate=1.0, triggered=True)
        assert health_adjusted_budget(cfg, sig) >= input_token_budget(cfg) * 0.7


# ── intervention text ─────────────────────────────────────────────────────────

def test_intervention_mentions_tool_churn():
    sig = ConfidenceSignal(score=0.2, error_rate=0.0, null_rate=0.0,
                           dup_rate=0.0, triggered=True, tool_call_rate=6.0)
    assert "6 tool calls per round" in ConfidenceMonitor.intervention_message(sig)


def test_intervention_omits_churn_when_calls_are_normal():
    sig = ConfidenceSignal(score=0.2, error_rate=0.7, null_rate=0.0,
                           dup_rate=0.0, triggered=True, tool_call_rate=2.0)
    assert "tool calls per round" not in ConfidenceMonitor.intervention_message(sig)
