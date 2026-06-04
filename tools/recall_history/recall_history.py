"""`recall_history` tool — read raw user messages from the current session's Q/A log.

Useful when the agent suspects goal drift or needs to verify its understanding
of the user's original intent against the verbatim chat history on disk.
"""
from __future__ import annotations

from typing import Any

from agent.tools import register

_qa_logger = None  # Set by setup() when a session is active.


def setup(qa_logger) -> None:
    """Wire the per-session QALogger. Called by Agent.set_session_id."""
    global _qa_logger
    _qa_logger = qa_logger


@register(
    "recall_history",
    {
        "description": (
            "Read verbatim user messages from Q/A log. "
            "Use when session summary may have drifted or to verify exact earlier wording. "
            "Never compacted. For derived facts/decisions use recall_facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "turns": {
                    "type": "integer",
                    "description": "Most-recent turns to retrieve (0=all, default 5).",
                },
                "from_turn": {
                    "type": "integer",
                    "description": "Return only turns >= this turn_id.",
                },
                "include_responses": {
                    "type": "boolean",
                    "description": "Include agent responses (default: false).",
                },
            },
            "required": [],
        },
    },
)
def recall_history(
    turns: int = 5,
    from_turn: int | None = None,
    include_responses: bool = False,
) -> dict[str, Any]:
    if _qa_logger is None:
        return {
            "error": "No session active.",
            "hint": "recall_history requires an active session with a QALogger.",
        }

    try:
        turns = max(0, int(turns or 5))
    except Exception:
        turns = 5

    import asyncio

    async def _read() -> list[tuple]:
        result = []
        async for entry in _qa_logger.read_history():
            result.append(entry)
        return result

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _read())
                history = future.result(timeout=10)
        else:
            history = loop.run_until_complete(_read())
    except Exception as e:
        return {"error": f"Failed to read history: {e}"}

    if from_turn is not None:
        history = [(tid, q, a) for tid, q, a in history if tid >= from_turn]

    if turns > 0:
        history = history[-turns:]

    out = []
    for tid, q_data, a_data in history:
        entry: dict[str, Any] = {
            "turn_id": tid,
            "user": q_data.get("content", "") if q_data else "",
            "timestamp": q_data.get("timestamp", "") if q_data else "",
        }
        if include_responses and a_data:
            resp = a_data.get("content", "")
            entry["agent"] = resp[:800] + ("…" if len(resp) > 800 else "")
        out.append(entry)

    return {
        "turns_returned": len(out),
        "history": out,
        "hint": (
            "These are verbatim user messages from disk — unaffected by compaction. "
            "Use turn_id=1 content to verify the original session goal."
        ),
    }
