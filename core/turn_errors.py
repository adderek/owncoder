"""Model-call failure handling for the turn loop: reliability bookkeeping,
endpoint cooldowns, failover, and the "no model can serve this turn" state.

Split out of core/turn.py so the endpoint-error policy is readable on its own —
run_turn still owns the control flow (continue / raise), this module owns the
decisions and the side effects each one implies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Rate-limit classification lives in config.model_probe (shared with the
# background-task failover helper in core.llm_retry).
from agent.config.model_probe import (
    retry_after_seconds,
    is_daily_quota_429,
)

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "NoUsableModelError",
    "no_usable_model_error",
    "record_model_outcome",
    "record_model_capability",
    "mark_endpoint_cooldown",
    "clear_endpoint_cooldown",
    "try_failover",
    "retry_after_seconds",
    "is_daily_quota_429",
]


class NoUsableModelError(Exception):
    """No model could serve the turn: the active endpoint failed (rate-limit /
    outage) and self-hosted failover found nothing live to degrade to.

    Carries the original endpoint error (``cause``) and the names of configured
    entries the user could enable to recover (``candidates``), so the UI can
    offer a retry + an enable-a-model choice instead of crash-reporting it. This
    is an expected operational state, not a bug — UIs should surface it, not
    write a crash report.
    """
    def __init__(self, cause: BaseException, candidates: list[str], reason: str = ""):
        self.cause = cause
        self.candidates = candidates
        hint = (f" — enable one to continue: {', '.join(candidates)}"
                if candidates else "")
        super().__init__(
            (reason or ("no enabled model is reachable (active endpoint failed and "
                        "no live local/LAN model to fall back to)")) + hint)


def no_usable_model_error(config, cause: BaseException, reason: str = "") -> NoUsableModelError:
    """Build a NoUsableModelError listing disabled entries the user could enable."""
    from agent.core.model_control import is_disabled
    entries = config.model_entries or {}
    candidates = [n for n in entries if is_disabled(config, n)]
    return NoUsableModelError(cause, candidates, reason)


def record_model_outcome(config: "Config", outcome: str) -> None:
    """Record success / failure / rate_limited against the active model entry.

    Best-effort: reliability stats must never be able to fail a turn.
    """
    try:
        from agent.metrics.model_stats import resolve_entry_name
        from agent.metrics.model_reliability import record_outcome
        record_outcome(resolve_entry_name(config), outcome)
    except Exception:
        logger.debug("record_model_outcome(%s) failed (ignored)", outcome, exc_info=True)


def record_model_capability(config: "Config", ok: bool, result_text: str) -> None:
    """Record one tool-call capability sample against the active model entry.

    Only successful calls and malformed calls are stored (see
    model_reliability.CAPABILITY_VERDICTS): a tool error such as a missing
    file is a legitimate exploration outcome, not a model defect.

    Best-effort: capability stats must never be able to fail a turn.
    """
    try:
        from agent.core.confidence import classify_error
        from agent.metrics.model_reliability import record_capability
        from agent.metrics.model_stats import resolve_entry_name
        if ok:
            verdict = "ok"
        elif classify_error(result_text) == "schema":
            verdict = "schema_error"
        else:
            return
        record_capability(resolve_entry_name(config), verdict)
    except Exception:
        logger.debug("record_model_capability failed (ignored)", exc_info=True)


def mark_endpoint_cooldown(config: "Config", cooldown_s: float | None = None) -> None:
    """Keep tier ladders and escalation off the active endpoint for a while.

    Without *cooldown_s* the probe's default applies (retry-to-revive: the ladder
    stays off the endpoint until a fresh availability probe says it works again).
    """
    try:
        from agent.config.model_probe import mark_rate_limited
        if cooldown_s is None:
            mark_rate_limited(config.llm.base_url, config.llm.model)
        else:
            mark_rate_limited(config.llm.base_url, config.llm.model, cooldown_s=cooldown_s)
    except Exception:
        logger.debug("mark_endpoint_cooldown failed (ignored)", exc_info=True)


def clear_endpoint_cooldown(config: "Config") -> None:
    """Undo ``mark_endpoint_cooldown`` for the active endpoint.

    Cooling an endpoint down is only meaningful when something else can take
    over. When the caller just found out that nothing can, the cooldown has
    stopped being a retry-to-revive hedge and become the thing that ends the
    turn — so the turn loop drops it before retrying the only endpoint left.
    """
    try:
        from agent.config.model_probe import clear_rate_limited
        clear_rate_limited(config.llm.base_url, config.llm.model)
    except Exception:
        logger.debug("clear_endpoint_cooldown failed (ignored)", exc_info=True)


def try_failover(config: "Config"):
    """Find another endpoint to finish the turn on; returns a client or None.

    Order matters: cloud→cloud first, because another mode-allowed cloud entry
    (e.g. a different free provider) beats degrading to a local model. Only when
    no peer exists do we fall to local, and then — if already local, e.g. a
    router whose preset fails to load — to another live local entry.

    When the active entry was pinned by the user, ``failover.pinned_policy``
    decides how far off that pin we may drift: "ask" refuses to switch at all,
    "free-only" keeps paid tiers out of the candidate set, "fallback" (default)
    behaves as before. Returning None makes the caller surface the recoverable
    no-model state, which is how the user gets asked what to do.

    Callers own the retry budget (``failover.max_retries``); this just picks.
    """
    from agent.core import model_routing
    allow_paid = True
    if model_routing.is_active_default_pinned(config):
        policy = (getattr(getattr(config, "failover", None), "pinned_policy", "")
                  or "fallback").strip().lower()
        if policy == "ask":
            logger.warning("failover: active model is pinned (pinned_policy=ask) — "
                           "not switching; surfacing the no-model state instead")
            return None
        if policy == "free-only":
            allow_paid = False
            logger.info("failover: pinned entry failed, pinned_policy=free-only — "
                        "paid endpoints excluded from the candidate set")
    new_client = model_routing.failover_to_peer(config, allow_paid=allow_paid)
    if new_client is None:
        new_client = model_routing.failover_to_local(config, allow_paid=allow_paid)
    if new_client is None:
        new_client = model_routing.failover_to_alternative(config, allow_paid=allow_paid)
    return new_client
