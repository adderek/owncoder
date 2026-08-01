"""Confidence monitor: detect when model is walking circles blind.

Tracks behavioral signals (tool failure rate, duplicate results, null results)
over a sliding window. When non-convergence exceeds threshold, fires an
intervention so the harness forces the model to articulate uncertainty — since
the model itself won't notice it's stuck.

The model doesn't know it's guessing; the harness must.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Result text shorter than this is treated as null/empty. Kept small so a
# legitimate compact tool ack (e.g. a short JSON success blob) is not
# mis-flagged as a null result — only genuinely empty/near-empty payloads
# (e.g. "", "{}", "[]", '{"results":[]}') fall under it.
_NULL_RESULT_CHARS = 20


def _result_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


# Substrings identifying an error caused by a malformed *call* rather than by
# the model reasoning badly: wrong/missing arguments, unparseable argument
# JSON, a tool that does not exist. A stronger model is the wrong lever for
# these — the fix is a precise schema reminder, and paying a metered tier to
# re-roll broken JSON is the worst possible reason to escalate.
_SCHEMA_ERROR_MARKERS = (
    "missing required argument",
    "invalid json arguments",
    "empty_args",
    "unknown tool",
    "unexpected argument",
    "args_json_decode_error",
    "missing_required_args",
)


# At or above this share of schema-shaped errors, the failures are a call-format
# problem: intervene with the schema reminder and do NOT escalate the tier.
SCHEMA_DOMINANT_SHARE = 0.5


def classify_error(result_text: str) -> str:
    """Classify an error result: "schema" (bad call) or "other"."""
    low = result_text.lower()
    return "schema" if any(m in low for m in _SCHEMA_ERROR_MARKERS) else "other"


@dataclass
class ConfidenceSignal:
    score: float          # 0.0 = totally lost, 1.0 = converging
    error_rate: float     # fraction of recent tool calls that returned errors
    null_rate: float      # fraction of recent results that were empty/null
    dup_rate: float       # fraction of recent results that were identical to a prior one
    triggered: bool       # True when score < threshold
    # Session-health rates. Unlike the three above they are not fractions and
    # do not feed the score — they describe *cost*, not confusion, and are read
    # by the context budget (see core/context_budget.health_adjusted_budget).
    tool_call_rate: float = 0.0    # tool calls per completed iteration
    token_usage_rate: float = 0.0  # approx result tokens per completed iteration
    # Fraction of the *errors* in the window that were malformed calls rather
    # than bad reasoning. Read by the auto-tier gate: schema-shaped failures
    # do not justify a costlier model (see classify_error).
    schema_error_share: float = 0.0

    @property
    def waste_rate(self) -> float:
        """Fraction of recent results that bought nothing: empty or repeated.

        Errors are excluded — an error message is short and usually tells the
        model something actionable. A duplicate full-file read is what quietly
        fills the window.
        """
        return min(1.0, self.null_rate + self.dup_rate)


class ConfidenceMonitor:
    """Per-turn sliding-window monitor for non-convergence signals.

    Call observe_result() after each tool call result. Call should_intervene()
    to check whether the model is circling blind. Call acknowledge() to reset
    cooldown after injecting an intervention.
    """

    def __init__(
        self,
        window: int = 8,
        error_rate_threshold: float = 0.6,
        null_rate_threshold: float = 0.6,
        dup_rate_threshold: float = 0.5,
        score_threshold: float = 0.35,
        inject_cooldown: int = 3,
    ) -> None:
        self.window = max(2, window)
        # Clamp thresholds away from 0 — signal() divides by each, so a config
        # value of 0 would raise ZeroDivisionError and crash the turn.
        self.error_rate_threshold = max(1e-6, error_rate_threshold)
        self.null_rate_threshold = max(1e-6, null_rate_threshold)
        self.dup_rate_threshold = max(1e-6, dup_rate_threshold)
        self.score_threshold = score_threshold
        self.inject_cooldown = inject_cooldown

        self._errors: list[bool] = []    # True = error result
        self._schema_errors: list[bool] = []  # True = error caused by a malformed call
        self._nulls: list[bool] = []     # True = empty/null result
        self._dups: list[bool] = []      # True = result hash seen before
        self._seen_hashes: set[str] = set()
        self._iters_since_last: int = 999  # starts past cooldown

        # Cost accounting, per completed iteration rather than per result: what
        # the turn is spending, as opposed to whether it is making progress.
        self._calls_this_iter: int = 0
        self._tokens_this_iter: int = 0
        self._calls_per_iter: list[int] = []
        self._tokens_per_iter: list[int] = []

    def observe_result(self, result_text: str, is_error: bool) -> None:
        """Record one tool call result. Call once per tool call."""
        h = _result_hash(result_text)
        is_null = not is_error and len(result_text.strip()) < _NULL_RESULT_CHARS
        is_dup = h in self._seen_hashes and not is_error and not is_null

        self._errors.append(is_error)
        self._schema_errors.append(is_error and classify_error(result_text) == "schema")
        self._nulls.append(is_null)
        self._dups.append(is_dup)
        self._seen_hashes.add(h)

        self._calls_this_iter += 1
        # 4 chars ≈ 1 token, the same approximation the turn's budget uses.
        self._tokens_this_iter += len(result_text) // 4

        # Keep only last `window` entries.
        if len(self._errors) > self.window:
            self._errors = self._errors[-self.window:]
            self._schema_errors = self._schema_errors[-self.window:]
            self._nulls = self._nulls[-self.window:]
            self._dups = self._dups[-self.window:]

    def tick_iter(self) -> None:
        """Call once per tool-call iteration (after all results processed)."""
        self._iters_since_last += 1
        self._calls_per_iter.append(self._calls_this_iter)
        self._tokens_per_iter.append(self._tokens_this_iter)
        self._calls_this_iter = 0
        self._tokens_this_iter = 0
        if len(self._calls_per_iter) > self.window:
            self._calls_per_iter = self._calls_per_iter[-self.window:]
            self._tokens_per_iter = self._tokens_per_iter[-self.window:]

    def _cost_rates(self) -> tuple[float, float]:
        """(tool calls, result tokens) per completed iteration."""
        iters = len(self._calls_per_iter)
        if iters == 0:
            return 0.0, 0.0
        return (sum(self._calls_per_iter) / iters,
                sum(self._tokens_per_iter) / iters)

    def signal(self) -> ConfidenceSignal:
        calls_rate, tokens_rate = self._cost_rates()
        n = len(self._errors)
        if n == 0:
            return ConfidenceSignal(score=1.0, error_rate=0.0, null_rate=0.0,
                                    dup_rate=0.0, triggered=False,
                                    tool_call_rate=round(calls_rate, 2),
                                    token_usage_rate=round(tokens_rate, 1))

        error_rate = sum(self._errors) / n
        n_err = sum(self._errors)
        schema_share = (sum(self._schema_errors) / n_err) if n_err else 0.0
        null_rate = sum(self._nulls) / n
        dup_rate = sum(self._dups) / n

        # Score: weighted inverse of non-convergence. Any single dimension
        # being bad is enough to drag score down.
        worst = max(
            error_rate / self.error_rate_threshold,
            null_rate / self.null_rate_threshold,
            dup_rate / self.dup_rate_threshold,
        )
        score = max(0.0, 1.0 - worst * 0.5)

        triggered = (
            n >= max(2, self.window // 2)  # need enough data first
            and score < self.score_threshold
            and self._iters_since_last >= self.inject_cooldown
        )
        return ConfidenceSignal(
            score=round(score, 3),
            error_rate=round(error_rate, 3),
            null_rate=round(null_rate, 3),
            dup_rate=round(dup_rate, 3),
            triggered=triggered,
            tool_call_rate=round(calls_rate, 2),
            token_usage_rate=round(tokens_rate, 1),
            schema_error_share=round(schema_share, 3),
        )

    def should_intervene(self) -> ConfidenceSignal:
        """Return signal; only call once per iteration (not after acknowledge)."""
        return self.signal()

    def acknowledge(self) -> None:
        """Reset cooldown after injecting an intervention."""
        self._iters_since_last = 0

    @staticmethod
    def intervention_message(sig: ConfidenceSignal) -> str:
        parts = []
        if sig.error_rate > 0.3:
            parts.append(f"error rate {sig.error_rate:.0%}")
        if sig.null_rate > 0.3:
            parts.append(f"null/empty results {sig.null_rate:.0%}")
        if sig.dup_rate > 0.3:
            parts.append(f"duplicate results {sig.dup_rate:.0%}")
        if sig.tool_call_rate >= 4:
            parts.append(f"{sig.tool_call_rate:.0f} tool calls per round")
        detail = ", ".join(parts) or f"score {sig.score:.2f}"
        if sig.schema_error_share >= SCHEMA_DOMINANT_SHARE and sig.error_rate > 0.3:
            return (
                f"[confidence-guard: {detail}. Most of those failures were malformed "
                "tool calls — missing required arguments, unparseable argument JSON, or a "
                "tool name that does not exist — not missing information. Before the next "
                "call: re-read the tool's schema, emit every required field, and emit the "
                "arguments as one valid JSON object. Call find_tools if unsure of the schema.]"
            )
        return (
            f"[confidence-guard: non-convergence detected ({detail}). "
            "State explicitly: (1) what you know for certain from tool output so far, "
            "(2) what you are currently guessing or assuming, "
            "(3) the single piece of information that would resolve the uncertainty. "
            "Then call the most targeted tool to get that information, or emit >>>BLOCKED: <reason>.]"
        )
