"""`save_note` tool — persist cross-session facts to project notes store.

Notes survive session boundaries. They are loaded at agent startup and
injected into context, giving the model memory of prior decisions, preferences,
and facts that aren't derivable from the codebase alone.

Storage: project-scoped MemoryStore at <agent_dir>/memory.db, scope='note'.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agent.tools import register

logger = logging.getLogger(__name__)

_config = None
_embedder = None
_notes_store = None  # MemoryStore, initialized lazily


def setup(config, embedder=None) -> None:
    global _config, _embedder, _notes_store
    _config = config
    _embedder = embedder
    _notes_store = None  # reset so _get_store() re-creates on next call


def _get_store():
    global _notes_store
    if _notes_store is not None:
        return _notes_store
    if _config is None:
        return None
    from pathlib import Path
    from agent.memory.store import MemoryStore
    agent_dir = Path(_config.tools.working_dir) / _config.tools.agent_dir
    _notes_store = MemoryStore(agent_dir / "memory.db")
    return _notes_store


@register(
    "save_note",
    {
        "description": (
            "Save a fact, decision, or preference that should persist beyond this "
            "session. Use for: architectural decisions, user preferences, recurring "
            "constraints, resolved ambiguities, 'always do X / never do Y' rules. "
            "Notes are injected into future sessions so the agent remembers them "
            "without having to re-derive them from code or history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title (≤80 chars). Used as note heading.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "Note content. Be specific: name files, functions, decisions, "
                        "rationale. Avoid conversational padding."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional topic tags for retrieval (e.g. ['auth','database']).",
                },
            },
            "required": ["title", "body"],
        },
    },
)
def save_note(
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    store = _get_store()
    if store is None:
        return {"error": "Notes store not configured. Call setup(config) first."}
    if not title.strip():
        return {"error": "`title` must be non-empty."}
    if not body.strip():
        return {"error": "`body` must be non-empty."}

    embedding = None
    if _embedder is not None:
        try:
            text = f"{title}\n\n{body}"
            embedding = _embedder.embed_one(text[:4000])
        except Exception:
            pass

    try:
        eid = store.add(
            scope="note",
            body=body.strip(),
            title=title.strip(),
            tags=tags or [],
            embedding=embedding,
        )
        return {"saved": True, "id": eid, "title": title.strip()}
    except Exception as e:
        logger.exception("save_note: store.add failed")
        return {"error": str(e)}


def load_notes_context(config=None, limit: int = 50) -> str | None:
    """Return all saved notes formatted for system context injection.

    Called at agent startup to load the full notes corpus. Commit 3 will
    replace this with selective per-turn injection based on query relevance.
    """
    cfg = config or _config
    if cfg is None:
        return None
    from pathlib import Path
    from agent.memory.store import MemoryStore
    agent_dir = Path(cfg.tools.working_dir) / cfg.tools.agent_dir
    db_path = agent_dir / "memory.db"
    if not db_path.exists():
        return None
    try:
        store = MemoryStore(db_path)
        entries = store.list_entries(scope="note", limit=limit, order_by="created_at ASC")
    except Exception:
        return None
    if not entries:
        return None

    lines = ["# Saved notes (cross-session memory)\n"]
    for e in entries:
        title = e.get("title") or "(untitled)"
        body = e.get("body") or ""
        tags = e.get("tags")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"## {title}{tag_str}\n{body}\n")

    return "\n".join(lines).strip() or None
