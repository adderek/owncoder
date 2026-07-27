"""Unit tests for the per-model TTFT expectation (metrics/ttft_expect.py)."""
from __future__ import annotations

from agent.metrics.ttft_expect import (
    FLOOR_S, MIN_SAMPLES, TtftExpectation, expect, _percentile,
)


def _samples(rate: float, n: int = 20, overhead: float = 2.0,
             lo: int = 500, hi: int = 40000) -> list[tuple[int, float]]:
    """n calls spread over prompt sizes lo..hi at *rate* s/token + *overhead*."""
    out = []
    for i in range(n):
        tokens = lo + (hi - lo) * i // max(1, n - 1)
        out.append((tokens, overhead + tokens * rate))
    return out


def _flat_samples(ttft: float, n: int = 20, tokens: int = 4000):
    """n calls that all used the same prompt size — no slope to identify."""
    return [(tokens, ttft) for _ in range(n)]


class TestPercentile:
    def test_bounds(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert _percentile(vals, 0.0) == 1.0
        assert _percentile(vals, 1.0) == 4.0

    def test_empty(self):
        assert _percentile([], 0.5) == 0.0


class TestFallback:
    def test_too_few_samples_uses_default(self):
        e = expect("x", 5000, 600.0, samples=_samples(0.001, n=MIN_SAMPLES - 1))
        assert e.basis == "default"
        assert e.budget_s == 600.0
        assert not e.adaptive

    def test_no_samples_uses_default(self):
        e = expect("x", 5000, 600.0, samples=[])
        assert e.basis == "default"
        # A quiet threshold is still offered so the UI has something to show.
        assert 8.0 <= e.quiet_s <= 45.0

    def test_zero_token_samples_ignored(self):
        junk = [(0, 1.0)] * 50 + [(100, 0.0)] * 50
        assert expect("x", 1000, 600.0, samples=junk).basis == "default"


class TestPrediction:
    def test_scales_with_prompt_size(self):
        s = _samples(0.002)          # 2ms per prompt token
        small = expect("x", 1000, 600.0, samples=s)
        big = expect("x", 100000, 600.0, samples=s)
        assert small.adaptive and big.adaptive
        # Linear in tokens on top of a fixed floor, so 100× the prompt is a
        # little under 100× the prediction.
        assert big.p50_s > small.p50_s * 30
        assert big.budget_s > small.budget_s

    def test_predicted_p50_matches_the_line(self):
        # Clean linear data: the fit should recover overhead + rate·tokens.
        e = expect("x", 10000, 600.0, samples=_samples(0.002, overhead=4.0))
        assert abs(e.p50_s - (4.0 + 0.002 * 10000)) < 0.5

    def test_overhead_not_extrapolated_as_rate(self):
        # The bug this model exists to avoid: calls whose time was mostly fixed
        # overhead (cold model load on small prompts) must not predict minutes
        # for a moderate prompt. ttft/tokens on the 200-token samples is 0.1
        # s/token, which naively extrapolates to 400s at 4k tokens.
        s = [(200, 20.0)] * 10 + [(20000, 30.0)] * 10
        e = expect("x", 4000, 600.0, samples=s)
        assert e.p50_s < 60.0

    def test_uniform_prompt_sizes_fall_back_to_flat(self):
        # No variation in prompt size → no identifiable slope; predict the
        # typical measured time instead of inventing a rate.
        e = expect("x", 6000, 600.0, samples=_flat_samples(12.0))
        assert e.adaptive
        assert abs(e.p50_s - 12.0) < 0.5

    def test_slow_tail_widens_the_budget(self):
        fast = _samples(0.002, n=18)
        slow = fast + [(1000, 1000 * 0.05)] * 6      # a quarter of calls crawl
        wide = expect("x", 10000, 600.0, samples=slow)
        tight = expect("x", 10000, 600.0, samples=fast)
        assert wide.budget_s > tight.budget_s

    def test_budget_floor_for_fast_models(self):
        # A model that always answers instantly must not get a 1s fuse.
        e = expect("x", 10, 600.0, samples=_samples(0.0001))
        assert e.budget_s >= 30.0

    def test_budget_capped(self):
        e = expect("x", 100000, 600.0, samples=_samples(0.05, hi=100000))
        assert e.budget_s <= 1800.0

    def test_no_extrapolation_far_past_measured_sizes(self):
        # Only small prompts measured → a 40k prompt is outside what the data
        # supports; the fixed budget is used rather than a scaled-up guess.
        e = expect("x", 40000, 600.0, samples=_samples(0.02, lo=30, hi=800))
        assert e.basis == "default" and e.budget_s == 600.0

    def test_quiet_below_budget(self):
        e = expect("x", 20000, 600.0, samples=_samples(0.003))
        assert e.quiet_s < e.budget_s
        assert e.quiet_s >= 8.0

    def test_as_dict_round_trips(self):
        d = expect("x", 1000, 600.0, samples=_samples(0.002)).as_dict()
        assert d["basis"] == "stats" and d["samples"] == 20
        assert set(d) == {"quiet_s", "budget_s", "p50_s", "p90_s", "samples", "basis"}


class TestExpectForConfig:
    def test_unknown_endpoint_falls_back(self):
        from agent.config import Config
        from agent.metrics.ttft_expect import expect_for_config
        cfg = Config()
        cfg.llm.stream_ttft_seconds = 600
        e = expect_for_config(cfg, 1000)
        assert isinstance(e, TtftExpectation)
        assert e.budget_s > 0
