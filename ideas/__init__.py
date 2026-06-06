"""Ideas subsystem — per-project idea capture, tracking, and pipeline staging.

Storage: .agent/ideas.db (separate from RAG/memory DBs — future migration target).
Public API: configure, get_store, add_idea, list_ideas, get_idea, update_idea.
"""
from __future__ import annotations

from pathlib import Path

from agent.ideas.store import IdeasStore, IDEA_TYPES, IDEA_STATUSES

_store: IdeasStore | None = None
_db_path: Path | None = None


def configure(working_dir: str, agent_dir: str = ".agent") -> None:
    global _store, _db_path
    _db_path = Path(working_dir) / agent_dir / "ideas.db"
    _store = None  # lazy-init on first use


def get_store() -> IdeasStore | None:
    global _store
    if _store is not None:
        return _store
    if _db_path is None:
        return None
    _store = IdeasStore(_db_path)
    return _store


def add_idea(
    title: str,
    body: str = "",
    type: str = "idea",
    tags: list[str] | None = None,
    source: str = "human",
    priority: int = 3,
    session_ref: str = "",
    project: str = "",
) -> str | None:
    store = get_store()
    if store is None:
        return None
    return store.add(
        title=title,
        body=body,
        type=type,
        tags=tags,
        source=source,
        priority=priority,
        session_ref=session_ref,
        project=project,
    )


def list_ideas(status: str | None = None, limit: int = 100) -> list[dict]:
    store = get_store()
    if store is None:
        return []
    return store.list(status=status, limit=limit)


def get_idea(idea_id: str) -> dict | None:
    store = get_store()
    if store is None:
        return None
    return store.get(idea_id)


def update_idea(idea_id: str, **fields) -> bool:
    store = get_store()
    if store is None:
        return False
    return store.update(idea_id, **fields)


__all__ = [
    "configure",
    "get_store",
    "add_idea",
    "list_ideas",
    "get_idea",
    "update_idea",
    "IDEA_TYPES",
    "IDEA_STATUSES",
]
