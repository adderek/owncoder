"""`submit_idea` tool — agent-callable idea capture.

Agent uses this to record improvement ideas, feature suggestions, bug reports,
or any other backlog item encountered during work.
"""
from __future__ import annotations

import logging
from typing import Any

from agent.tools import register

logger = logging.getLogger(__name__)

_config = None


def setup(config) -> None:
    global _config
    _config = config
    from agent import ideas as _ideas_mod
    _ideas_mod.configure(config.tools.working_dir, config.tools.agent_dir)


@register(
    "submit_idea",
    {
        "description": (
            "Record an improvement idea, feature request, bug, or optimization "
            "for this project. Ideas are stored in .agent/ideas.db and can be "
            "reviewed, evaluated, and eventually auto-implemented. "
            "Use when you spot something worth doing but out of current scope."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title (≤120 chars).",
                },
                "body": {
                    "type": "string",
                    "description": "Details: what, why, how. Be specific.",
                },
                "type": {
                    "type": "string",
                    "enum": ["feature", "bug", "optimization", "integration", "module", "idea"],
                    "description": "Idea category.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Topic tags (e.g. ['performance', 'ui']).",
                },
                "priority": {
                    "type": "integer",
                    "description": "1 (low) – 5 (critical). Default 3.",
                },
            },
            "required": ["title"],
        },
    },
)
def submit_idea(
    title: str,
    body: str = "",
    type: str = "idea",
    tags: list[str] | None = None,
    priority: int = 3,
) -> dict[str, Any]:
    from agent import ideas as _ideas_mod

    store = _ideas_mod.get_store()
    if store is None:
        if _config is not None:
            _ideas_mod.configure(_config.tools.working_dir, _config.tools.agent_dir)
            store = _ideas_mod.get_store()
    if store is None:
        return {"error": "Ideas store not configured."}

    session_ref = ""

    try:
        idea_id = store.add(
            title=title.strip(),
            body=body.strip(),
            type=type,
            tags=tags or [],
            source="agent",
            priority=max(1, min(5, int(priority))),
            session_ref=session_ref,
        )
        return {"saved": True, "id": idea_id, "title": title.strip()}
    except Exception as e:
        logger.exception("submit_idea: store.add failed")
        return {"error": str(e)}
