"""`recall_sessions` tool — semantic search across past session summaries.

Each compaction round indexes its summary into the project-level MemoryStore
(scope='session_summary'). This tool lets the model find past decisions,
errors, or solutions from sessions days or weeks ago without reading old
session files.
"""
from __future__ import annotations

from typing import Any

from agent.tools import register

_config = None
_embedder = None


def setup(config, embedder=None) -> None:
    global _config, _embedder
    _config = config
    _embedder = embedder


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
    "recall_sessions",
    {
        "description": (
            "Search past session summaries for prior decisions, errors, solutions. "
            "Use for: 'what did we decide about X?', 'how did we fix Y before?'. "
            "Ranked by relevance. Unlike recall_facts (current session only), searches ALL past sessions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords describing what to find.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 5).",
                },
                "outcome_filter": {
                    "type": "string",
                    "enum": ["good", "bad", "ok"],
                    "description": "Filter by session outcome. Omit for all.",
                },
            },
            "required": ["query"],
        },
    },
)
def recall_sessions(
    query: str,
    max_results: int = 5,
    outcome_filter: str | None = None,
) -> dict[str, Any]:
    if not (query or "").strip():
        return {"error": "`query` must be non-empty."}
    try:
        max_results = max(1, min(int(max_results or 5), 20))
    except Exception:
        max_results = 5

    store = _get_store()
    if store is None:
        return {
            "query": query,
            "matches": [],
            "hint": "No session history indexed yet. History accumulates as sessions run.",
        }

    embedding = None
    if _embedder is not None:
        try:
            embedding = _embedder.embed_one(query[:2000])
        except Exception:
            pass

    tags_filter = [f"outcome:{outcome_filter}"] if outcome_filter else None
    hits = store.hybrid_search(
        query,
        embedding=embedding,
        scope="session_summary",
        top_k=max_results,
        tags_filter=tags_filter,
    )

    if not hits:
        return {
            "query": query,
            "matches": [],
            "hint": "No matching session summaries. Try broader terms.",
        }

    matches = []
    for h in hits:
        matches.append({
            "session_id": h.get("source", ""),
            "title": h.get("title", ""),
            "snippet": (h.get("body") or "")[:600],
            "score": h.get("combined_score", h.get("score", 0.0)),
        })

    return {"query": query, "matches": matches}
