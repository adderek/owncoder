"""Expected time-to-first-token per model, learned from measured samples.

A fixed TTFT budget has to be set for the worst case (a 200k-token prompt on
a cold LAN box), which makes it useless as a "something is wrong" signal for
the common case (a small prompt on a warm endpoint): by the time a 600s fuse
blows, the user has been staring at a frozen-looking UI for ten minutes.

``model_history`` already stores every call's ``(in_tokens, ttft)`` pair per
entry, so the budget can be derived instead. This module fits

    ttft ≈ overhead + rate · prompt_tokens

per entry (Theil–Sen, so a few cold-load outliers don't bend the line) and
reads two levels off it: the median residual for "this is taking longer than
usual" and the p90 residual for "this has left the normal range entirely".

Splitting fixed overhead from per-token cost is the point. A rate computed as
``ttft / prompt_tokens`` is dominated by overhead on small prompts, and
extrapolating it to a big one predicts minutes where seconds are due — the
first version of this module did exactly that and produced a 600s budget for a
4k-token call.

Everything degrades to the caller's fixed default when there is not enough
history: a fresh install, a new endpoint, or a model whose samples are all
from cached-prefill calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_SAMPLES = 8        # below this the percentiles are noise — use the default
LOOKBACK_DAYS = 30
MAX_SAMPLES = 400     # Theil-Sen is O(n^2) in pairs — cap the window
EXTRAPOLATE_FACTOR = 3.0  # predict at most this far past the largest measured prompt
FLOOR_S = 3.0          # connection setup + queueing, independent of prompt size
BUDGET_SAFETY = 2.0    # p90 prediction × this = "declare it wedged"
QUIET_SAFETY = 1.5     # p50 prediction × this = "mention that it is slow"
MIN_BUDGET_S = 30.0    # never fuse faster than this, however quick the model
MAX_BUDGET_S = 1800.0
MIN_QUIET_S = 8.0


@dataclass
class TtftExpectation:
    """Thresholds for one upcoming call. Seconds; 0 budget = no fuse."""
    quiet_s: float       # past this, say the backend is slow
    budget_s: float      # past this, call it stalled
    p50_s: float         # predicted typical TTFT for this prompt size
    p90_s: float         # predicted slow-but-normal TTFT
    samples: int         # samples the prediction is based on (0 = default)
    basis: str           # "stats" | "default"

    @property
    def adaptive(self) -> bool:
        return self.basis == "stats"

    def as_dict(self) -> dict:
        return {"quiet_s": round(self.quiet_s, 1), "budget_s": round(self.budget_s, 1),
                "p50_s": round(self.p50_s, 1), "p90_s": round(self.p90_s, 1),
                "samples": self.samples, "basis": self.basis}


def _median(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _fit(pairs: "list[tuple[int, float]]") -> "tuple[float, float]":
    """Robust ``(sec_per_token, fixed_overhead)`` fit of ttft against prompt size.

    Theil–Sen (median of pairwise slopes) rather than least squares: a handful
    of calls that hit a cold model load or a queued GPU are normal in this
    data and would drag a least-squares line badly. When the samples carry
    little variation in prompt size the slope is not identifiable — fall back
    to a flat model (all cost as fixed overhead), which is the honest answer
    for "this endpoint has only ever seen 4k-token prompts".
    """
    ns = [n for n, _ in pairs]
    if max(ns) < 2 * min(ns):
        return 0.0, _median([t for _, t in pairs])
    slopes = [
        (t2 - t1) / (n2 - n1)
        for i, (n1, t1) in enumerate(pairs)
        for (n2, t2) in pairs[i + 1:]
        if n2 != n1
    ]
    slope = max(0.0, _median(slopes))
    base = max(0.0, _median([t - slope * n for n, t in pairs]))
    return slope, base


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile of a non-empty sorted list (q in 0..1)."""
    if not sorted_vals:
        return 0.0
    idx = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


def expect(entry_name: str, prompt_tokens: int, default_budget_s: float,
           *, samples: "list[tuple[int, float]] | None" = None,
           min_samples: int = MIN_SAMPLES) -> TtftExpectation:
    """Predict TTFT thresholds for *prompt_tokens* on *entry_name*.

    *samples* (``[(in_tokens, ttft), …]``) is for tests; production reads the
    history db. *default_budget_s* is the configured fixed fuse, used verbatim
    when history is too thin to model — with a quiet threshold derived from it
    so the UI still has something to show.
    """
    fallback = TtftExpectation(
        quiet_s=max(MIN_QUIET_S, min(45.0, default_budget_s * 0.1)),
        budget_s=float(default_budget_s), p50_s=0.0, p90_s=0.0,
        samples=0, basis="default",
    )
    if samples is None:
        try:
            from agent.metrics.model_history import ttft_samples
            samples = ttft_samples(entry_name, days=LOOKBACK_DAYS)
        except Exception:
            logger.debug("ttft history unavailable for %s", entry_name, exc_info=True)
            return fallback
    pairs = [(int(n), float(t)) for n, t in (samples or []) if n > 0 and t > 0]
    if len(pairs) < min_samples:
        return fallback
    if len(pairs) > MAX_SAMPLES:
        pairs = pairs[-MAX_SAMPLES:]

    tokens = max(0, int(prompt_tokens or 0))
    # Refuse to extrapolate far past the prompt sizes actually measured. An
    # endpoint that has only ever seen 800-token prompts says nothing useful
    # about a 40k one, and a slope fitted in the small regime multiplies into
    # a wildly wrong budget. The fixed setting is the honest fallback there.
    if tokens > EXTRAPOLATE_FACTOR * max(n for n, _ in pairs):
        return fallback
    slope, base = _fit(pairs)
    pred = base + slope * tokens
    # Spread of what the fit did not explain (a cold model load, a busy GPU,
    # a queue) — added on top rather than scaled, since it does not grow with
    # the prompt.
    resid = sorted(t - (base + slope * n) for n, t in pairs)
    p50 = max(FLOOR_S, pred + _percentile(resid, 0.5))
    p90 = max(p50, pred + max(0.0, _percentile(resid, 0.9)))
    budget = min(MAX_BUDGET_S, max(MIN_BUDGET_S, p90 * BUDGET_SAFETY))
    # The "slow" mark has to stay clear of the fuse, or the UI would jump
    # straight from "fine" to "stalled" with nothing in between.
    quiet = min(budget * 0.6, max(MIN_QUIET_S, p50 * QUIET_SAFETY))
    return TtftExpectation(quiet_s=quiet, budget_s=budget, p50_s=p50, p90_s=p90,
                           samples=len(pairs), basis="stats")


def expect_for_config(config, prompt_tokens: int) -> TtftExpectation:
    """``expect()`` for the config's active endpoint and fixed fuse."""
    default_budget = float(getattr(config.llm, "stream_ttft_seconds", 0) or 0)
    # A config with no model entries is not a real deployment (bare Config() in
    # tests, minimal embedded use): don't reach into the shared history db for
    # it — the prediction would depend on whatever this machine happens to have
    # recorded for a same-named model.
    if not getattr(config, "model_entries", None):
        return expect("", prompt_tokens, default_budget, samples=[])
    try:
        from agent.metrics.model_stats import resolve_entry_name
        entry = resolve_entry_name(config)
    except Exception:
        entry = ""
    if not entry:
        return expect("", prompt_tokens, default_budget, samples=[])
    return expect(entry, prompt_tokens, default_budget)
