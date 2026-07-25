"""Agent-callable surface for the immutable core rules: read and propose only.

The core is the one part of the prompt with a human in the loop, so the agent
gets no write path to it at all — not a guarded one, not a confirm-first one.
What it gets instead is a way to *propose*: `propose_core_change` files the
proposal into the idea backlog as `type="core_change"`, where a human reviews
it. The backlog entry is the change request; the change happens when a person
edits the file.

`core_rules_history` answers "when did the core change, and why" from the ledger
in `.agent/core_history.jsonl`, which records the git commit message behind each
observed change.
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


def _ideas_store():
    from agent import ideas as _ideas_mod

    store = _ideas_mod.get_store()
    if store is None and _config is not None:
        _ideas_mod.configure(_config.tools.working_dir, _config.tools.agent_dir)
        store = _ideas_mod.get_store()
    return store


@register(
    "propose_core_change",
    {
        "description": (
            "Propose a change to the agent's immutable core rules. You cannot "
            "edit the core yourself — this files the proposal in the idea "
            "backlog for a human to review and apply. Use when experience in "
            "this session suggests a core rule is wrong, missing, or harmful."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string",
                          "description": "One line: the rule change you want."},
                "rationale": {"type": "string",
                              "description": "What happened that makes this necessary. "
                                             "Cite the concrete episode, not a generality."},
                "proposed_text": {"type": "string",
                                  "description": "The exact rule text you propose, as it "
                                                 "would appear in the core."},
                "replaces": {"type": "string",
                             "description": "Existing core rule this would replace or "
                                            "amend; empty for a new rule."},
                "priority": {"type": "integer",
                             "description": "1 (low) – 5 (critical). Default 3."},
            },
            "required": ["title", "rationale"],
        },
    },
)
def propose_core_change(title: str, rationale: str, proposed_text: str = "",
                        replaces: str = "", priority: int = 3) -> dict[str, Any]:
    store = _ideas_store()
    if store is None:
        return {"error": "Ideas store not configured; proposal not saved."}

    body_parts = [f"Rationale:\n{rationale.strip()}"]
    if replaces.strip():
        body_parts.append(f"Replaces / amends:\n{replaces.strip()}")
    if proposed_text.strip():
        body_parts.append(f"Proposed rule text:\n{proposed_text.strip()}")
    body_parts.append(
        "Filed by the agent. The core is human input only: applying this means "
        "a human editing prompts/core.txt (or .agent/core.md) in a commit whose "
        "message says why."
    )
    try:
        idea_id = store.add(
            title=title.strip()[:120],
            body="\n\n".join(body_parts),
            type="core_change",
            tags=["core"],
            source="agent",
            priority=max(1, min(5, int(priority))),
        )
    except Exception as e:
        logger.exception("propose_core_change: store.add failed")
        return {"error": str(e)}
    return {
        "proposed": True,
        "id": idea_id,
        "note": "Filed for human review. The core rules are unchanged.",
    }


@register(
    "core_rules_history",
    {
        "description": (
            "When did the immutable core rules change, and why. Reads the "
            "append-only ledger, which records each observed change with the "
            "commit message of the human edit behind it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "Most recent N entries. Default 10."},
            },
        },
    },
)
def core_rules_history(limit: int = 10) -> dict[str, Any]:
    from agent.core import core_rules

    if _config is None:
        return {"error": "not configured"}
    entries = core_rules.history(_config, limit=max(1, int(limit)))
    return {
        "current_digest": core_rules.digest(_config),
        "count": len(entries),
        "entries": entries,
        "note": ("Empty history means the core has not changed since this "
                 "project first recorded it." if not entries else ""),
    }
