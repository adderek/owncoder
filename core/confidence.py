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


@dataclass
class ConfidenceSignal:
    score: float          # 0.0 = totally lost, 1.0 = converging
    error_rate: float     # fraction of recent tool calls that returned errors
    null_rate: float      # fraction of recent results that were empty/null
    dup_rate: float       # fraction of recent results that were identical to a prior one
    triggered: bool       # True when score < threshold


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
        self.error_rate_threshold = error_rate_threshold
        self.null_rate_threshold = null_rate_threshold
        self.dup_rate_threshold = dup_rate_threshold
        self.score_threshold = score_threshold
        self.inject_cooldown = inject_cooldown

        self._errors: list[bool] = []    # True = error result
        self._nulls: list[bool] = []     # True = empty/null result
        self._dups: list[bool] = []      # True = result hash seen before
        self._seen_hashes: set[str] = set()
        self._iters_since_last: int = 999  # starts past cooldown

    def observe_result(self, result_text: str, is_error: bool) -> None:
        """Record one tool call result. Call once per tool call."""
        h = _result_hash(result_text)
        is_null = not is_error and len(result_text.strip()) < _NULL_RESULT_CHARS
        is_dup = h in self._seen_hashes and not is_error and not is_null

        self._errors.append(is_error)
        self._nulls.append(is_null)
        self._dups.append(is_dup)
        self._seen_hashes.add(h)

        # Keep only last `window` entries.
        if len(self._errors) > self.window:
            self._errors = self._errors[-self.window:]
            self._nulls = self._nulls[-self.window:]
            self._dups = self._dups[-self.window:]

    def tick_iter(self) -> None:
        """Call once per tool-call iteration (after all results processed)."""
        self._iters_since_last += 1

    def signal(self) -> ConfidenceSignal:
        n = len(self._errors)
        if n == 0:
            return ConfidenceSignal(score=1.0, error_rate=0.0, null_rate=0.0, dup_rate=0.0, triggered=False)

        error_rate = sum(self._errors) / n
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
        detail = ", ".join(parts) or f"score {sig.score:.2f}"
        return (
            f"[confidence-guard: non-convergence detected ({detail}). "
            "State explicitly: (1) what you know for certain from tool output so far, "
            "(2) what you are currently guessing or assuming, "
            "(3) the single piece of information that would resolve the uncertainty. "
            "Then call the most targeted tool to get that information, or emit >>>BLOCKED: <reason>.]"
        )
