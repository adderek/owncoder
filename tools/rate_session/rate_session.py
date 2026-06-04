"""`rate_session` tool — record outcome quality for the current (or named) session.

Updates ``user_outcome`` or ``agent_outcome`` on the session file and re-tags
all ``session_summary`` MemoryStore entries so ``recall_sessions`` can filter
by outcome in future sessions.
"""
from __future__ import annotations

from typing import Any

from agent.tools import register

_VALID_OUTCOMES = {"good", "bad", "ok"}

_config = None
_session_id: str | None = None


def setup(config) -> None:
    global _config
    _config = config


def set_session(session_id: str) -> None:
    global _session_id
    _session_id = session_id


def _get_store():
    if _config is None:
        return None
    from pathlib import Path
    from agent.memory.store import MemoryStore
    agent_dir = Path(_config.tools.working_dir) / _config.tools.agent_dir
    db_path = agent_dir / "memory.db"
    if not db_path.exists():
        return None
    return MemoryStore(db_path)


@register(
    "rate_session",
    {
        "description": (
            "Record session outcome. Call on task complete: 'good'|'bad'|'ok'. "
            "Good sessions get injected as context in future related tasks. "
            "voter='agent' default; 'user' only when relaying explicit user feedback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": ["good", "bad", "ok"],
                    "description": "Quality rating for this session.",
                },
                "voter": {
                    "type": "string",
                    "enum": ["agent", "user"],
                    "description": "Who is rating. Default 'agent'.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session to rate. Defaults to current session.",
                },
            },
            "required": ["outcome"],
        },
    },
)
def rate_session(
    outcome: str,
    voter: str = "agent",
    session_id: str | None = None,
) -> dict[str, Any]:
    outcome = (outcome or "").strip().lower()
    if outcome not in _VALID_OUTCOMES:
        return {"error": f"outcome must be one of {sorted(_VALID_OUTCOMES)}"}

    voter = (voter or "agent").strip().lower()
    if voter not in ("agent", "user"):
        voter = "agent"

    sid = session_id or _session_id
    if not sid:
        return {"error": "No session active. Provide session_id."}

    if _config is None:
        return {"error": "rate_session not configured."}

    # Update session file
    from agent.memory.session import load_session, save_session, configure as _configure_session
    _configure_session(_config.tools.working_dir, _config.tools.agent_dir)
    session, messages = load_session(sid)
    if session is None:
        return {"error": f"Session not found: {sid}"}

    if voter == "user":
        session.user_outcome = outcome
    else:
        session.agent_outcome = outcome

    save_session(session, messages)

    # Re-tag MemoryStore entries for this session
    store = _get_store()
    updated = 0
    if store is not None:
        # Effective outcome: prefer user_outcome, fall back to agent_outcome
        effective = session.user_outcome or session.agent_outcome
        if effective:
            updated = store.update_source_tags(
                scope="session_summary",
                source=sid,
                tags=[f"outcome:{effective}"],
            )

    return {
        "session_id": sid,
        "voter": voter,
        "outcome": outcome,
        "store_entries_updated": updated,
    }
