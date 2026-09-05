"""Context budget: how many prompt tokens a turn may use before compaction.

One number, derived from the model window, the output reserve and the
configured compaction threshold, so the trigger the turn loop applies is also
the number the model is told about in its budget notice.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# --- config-derived budgets -------------------------------------------------
# ctx_window == 0 means "auto" (probe has not filled it in yet, or the probe
# failed). Callers that subtract an output reserve from it produced degenerate
# budgets (e.g. 0 - 8192 - 500 -> clamped to 1), which made every turn look
# over-budget and compact/truncate pointlessly on every iteration.

DEFAULT_CTX_WINDOW = 32768   # conservative stand-in for an unprobed window
_PROMPT_OVERHEAD = 500       # room for the system/tool preamble the estimate misses

_warned_auto_ctx = False


def effective_ctx_window(config) -> int:
    """config.llm.ctx_window, or a conservative default when it is 0/auto."""
    global _warned_auto_ctx
    ctx = int(getattr(getattr(config, "llm", None), "ctx_window", 0) or 0)
    if ctx > 0:
        return ctx
    if not _warned_auto_ctx:
        _warned_auto_ctx = True
        logger.warning(
            "ctx_window is 0 (auto/unprobed) for model '%s' — assuming %d; "
            "set it explicitly in agent.toml or run the model probe",
            getattr(getattr(config, "llm", None), "model", "?"), DEFAULT_CTX_WINDOW,
        )
    return DEFAULT_CTX_WINDOW


def input_token_budget(config) -> int:
    """Tokens usable by the prompt: window minus output reserve and overhead.

    The output reserve is clamped to half the window so a misconfigured
    max_output_tokens >= ctx_window cannot drive the budget to ~0.
    """
    ctx = effective_ctx_window(config)
    max_out = int(getattr(getattr(config, "llm", None), "max_output_tokens", 0) or 0)
    reserve = min(max_out, ctx // 2)
    return max(1024, ctx - reserve - _PROMPT_OVERHEAD)


# --- health-reactive budget -------------------------------------------------
# A turn whose tool results are mostly empty or repeated is filling the window
# with text that bought nothing. Tightening the budget in that state makes the
# turn shed the junk sooner instead of carrying it to the context ceiling.

#: waste_rate below this leaves the budget alone — some repetition is normal.
_WASTE_FLOOR = 0.5
#: Most the budget may be tightened by, at waste_rate 1.0. Deliberately modest:
#: tightening pulls compaction forward, which costs an LLM call and (on
#: providers with a prompt cache) invalidates the cached prefix.
_MAX_TIGHTEN = 0.25


def health_adjusted_budget(config, signal=None) -> int:
    """input_token_budget(), tightened when the turn is wasting context.

    *signal* is a ConfidenceSignal (or None to skip the adjustment). Scales
    linearly from no change at waste_rate 0.5 to -25% at 1.0.
    """
    budget = input_token_budget(config)
    waste = float(getattr(signal, "waste_rate", 0.0) or 0.0) if signal is not None else 0.0
    if waste <= _WASTE_FLOOR:
        return budget
    over = (waste - _WASTE_FLOOR) / (1.0 - _WASTE_FLOOR)
    tightened = int(budget * (1.0 - _MAX_TIGHTEN * min(1.0, over)))
    logger.debug("context budget tightened %d -> %d (waste_rate %.2f)",
                 budget, tightened, waste)
    return max(1024, tightened)


def compaction_trigger_budget(config, signal=None) -> int:
    """The usage at which compaction actually fires.

    Two limits bound a turn and the earlier one wins:
      - input_token_budget(): what physically fits once the output reserve is
        set aside (tightened by health_adjusted_budget when the turn is
        wasting context);
      - llm.compaction_threshold x window: the configured comfort limit.

    They used to be applied at different points in the turn loop, so the
    threshold that fired depended on where in the turn you were — and neither
    matched the headroom reported to the model.
    """
    ctx = effective_ctx_window(config)
    threshold = float(getattr(getattr(config, "llm", None), "compaction_threshold", 0.75) or 0.75)
    configured = int(ctx * threshold)
    return max(1024, min(health_adjusted_budget(config, signal), configured))
