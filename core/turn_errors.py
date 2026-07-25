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
    "mark_endpoint_cooldown",
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
    def __init__(self, cause: BaseException, candidates: list[str]):
        self.cause = cause
        self.candidates = candidates
        hint = (f" — enable one to continue: {', '.join(candidates)}"
                if candidates else "")
        super().__init__(
            "no enabled model is reachable (active endpoint failed and no live "
            f"local/LAN model to fall back to){hint}")


def no_usable_model_error(config, cause: BaseException) -> NoUsableModelError:
    """Build a NoUsableModelError listing disabled entries the user could enable."""
    from agent.core.model_control import is_disabled
    entries = config.model_entries or {}
    candidates = [n for n in entries if is_disabled(config, n)]
    return NoUsableModelError(cause, candidates)


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


def try_failover(config: "Config"):
    """Find another endpoint to finish the turn on; returns a client or None.

    Order matters: cloud→cloud first, because another mode-allowed cloud entry
    (e.g. a different free provider) beats degrading to a local model. Only when
    no peer exists do we fall to local, and then — if already local, e.g. a
    router whose preset fails to load — to another live local entry.

    Callers own the retry budget (``failover.max_retries``); this just picks.
    """
    from agent.core import model_routing
    new_client = model_routing.failover_to_peer(config)
    if new_client is None:
        new_client = model_routing.failover_to_local(config)
    if new_client is None:
        new_client = model_routing.failover_to_alternative(config)
    return new_client
