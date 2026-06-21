"""Deferred ("idle") action queue.

Work that should happen while the agent is otherwise doing nothing — waiting on
the user — runs here. The first consumers are session auto-naming and a
backfill pass that names older unnamed sessions. New deferrable actions register
with :func:`register_idle_action` and are picked up automatically.

Design notes:
- Actions are async ``fn(agent) -> bool`` (True = did meaningful work).
- They must be fail-soft: never raise; log and return False on error.
- They should be cheap or self-limiting; the queue runs them one at a time so a
  burst of idle work never stampedes the local model.
- A run is abandoned if a new turn starts (checked via ``agent._last_turn_time``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from agent.core.agent import Agent

logger = logging.getLogger(__name__)

IdleAction = Callable[["Agent"], Awaitable[bool]]

# Registered actions, in run order. Tuples of (name, fn).
_ACTIONS: list[tuple[str, IdleAction]] = []

# Guard so only one idle sweep runs at a time even if scheduled concurrently.
_running = asyncio.Lock()


def register_idle_action(name: str, fn: IdleAction) -> None:
    """Register a deferrable action. Idempotent on name."""
    for i, (existing, _) in enumerate(_ACTIONS):
        if existing == name:
            _ACTIONS[i] = (name, fn)
            return
    _ACTIONS.append((name, fn))


def registered_actions() -> list[str]:
    return [name for name, _ in _ACTIONS]


async def run_pending(agent: "Agent") -> int:
    """Run registered idle actions once. Returns count that did work.

    Abandons the sweep if a new turn begins partway through.
    """
    if _running.locked():
        return 0
    async with _running:
        stamp = getattr(agent, "_last_turn_time", None)
        did = 0
        for name, fn in list(_ACTIONS):
            # Bail out if the user started a new turn while we were working.
            if stamp is not None and getattr(agent, "_last_turn_time", None) != stamp:
                logger.debug("idle_tasks: new turn started; abandoning sweep")
                break
            try:
                if await fn(agent):
                    did += 1
            except Exception:
                logger.debug("idle_tasks: action %s failed", name, exc_info=True)
        return did


# ── Built-in actions ─────────────────────────────────────────────────────────


async def _action_name_current(agent: "Agent") -> bool:
    """Generate metadata for the active session if it lacks any."""
    config = getattr(agent, "config", None)
    if config is None or not getattr(config.agent, "auto_name_sessions", True):
        return False
    session = getattr(agent, "session", None)
    if session is None:
        return False

    from agent.memory.session_namer import needs_meta, generate_session_meta, apply_meta
    from agent.memory.session import save_session

    if not needs_meta(session):
        return False
    meta = await generate_session_meta(session, agent.messages, config)
    if not meta:
        return False
    if apply_meta(session, meta):
        await asyncio.to_thread(save_session, session, agent.messages)
        logger.debug("idle_tasks: named current session %s -> %r", session.id, session.name)
        return True
    return False


async def _action_backfill(agent: "Agent") -> bool:
    """Name one older unnamed session per sweep (rate-limited)."""
    config = getattr(agent, "config", None)
    if config is None or not getattr(config.agent, "idle_backfill", True):
        return False

    from agent.memory.session import list_sessions, load_session, save_session
    from agent.memory.session_namer import needs_meta, generate_session_meta, apply_meta

    current_id = getattr(getattr(agent, "session", None), "id", None)
    # Oldest first: prefer to fill in the long tail of unnamed history.
    for summary in list_sessions(oldest_first=True):
        sid = summary.get("id")
        if not sid or sid == current_id:
            continue
        if summary.get("message_count", 0) < 2:
            continue
        # Cheap pre-filter on the listing dict before loading the full session.
        if summary.get("name") and summary.get("description") and summary.get("tags") \
                and summary.get("classification"):
            continue
        session, messages = load_session(sid)
        if session is None or not needs_meta(session):
            continue
        meta = await generate_session_meta(session, messages, config)
        if not meta:
            continue
        if apply_meta(session, meta):
            await asyncio.to_thread(save_session, session, messages)
            logger.debug("idle_tasks: backfilled session %s -> %r", sid, session.name)
            return True
        # Generated nothing usable; stop here to avoid hammering the model.
        return False
    return False


def register_builtins() -> None:
    """Register the built-in idle actions. Safe to call multiple times."""
    register_idle_action("name-current", _action_name_current)
    register_idle_action("backfill", _action_backfill)


register_builtins()
