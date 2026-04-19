from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules

if TYPE_CHECKING:
    from agent.config import Config
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder


_config = None
_store = None
_embedder = None
_asm_store = None
_archive_store = None


def setup(config, store, embedder, asm_store=None) -> None:
    global _config, _store, _embedder, _asm_store, _archive_store
    _config = config
    _store = store
    _embedder = embedder
    _asm_store = asm_store
    _archive_store = None  # lazy-open on first archive search


@register(
    "search_code",
    {
        "description": "Search the codebase using semantic + keyword search. Always call this before read_file to find relevant code.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language or keyword search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 8)",
                },
            },
            "required": ["query"],
        },
    },
)
def search_code(query: str, top_k: int | None = None) -> dict:
    if _store is None:
        return {"error": "Index not loaded. Run 'agent init' first."}

    k = top_k or (_config.rag.top_k if _config else 8)

    embedding = None
    if _embedder:
        try:
            embedding = _embedder.embed_one(query)
        except Exception as e:
            pass

    if embedding and _config and _config.rag.hybrid:
        results = _store.hybrid_search(query, embedding, top_k=k)
    elif embedding:
        results = _store.vector_search(embedding, top_k=k)
    else:
        results = _store.fts_search(query, top_k=k)

    # Clean up results for LLM consumption
    cleaned = []
    for r in results:
        cleaned.append(
            {
                "path": r.get("path"),
                "name": r.get("name"),
                "language": r.get("language"),
                "node_type": r.get("node_type"),
                "start_line": r.get("start_line"),
                "end_line": r.get("end_line"),
                "content": r.get("content", "")[:800],  # trim for context
            }
        )

    # Also query asm semantic search if available
    if _asm_store is not None and embedding is not None:
        try:
            asm_results = _asm_store.semantic_search(embedding, top_k=k)
            for r in asm_results:
                if r.get("description"):
                    cleaned.append(
                        {
                            "path": r.get("path"),
                            "name": r.get("inferred_name"),
                            "language": "asm",
                            "node_type": f"asm_unit_level{r.get('level', 0)}",
                            "start_line": r.get("start_line"),
                            "end_line": r.get("end_line"),
                            "content": r.get("description", ""),
                            "score": r.get("score"),
                        }
                    )
        except Exception:
            pass

    # Rule check: filter out .agent.ignore paths
    rules = get_rules()
    if not rules.ignore.empty:
        cleaned = [r for r in cleaned if not rules.ignore.matches(r.get("path", ""))]

    return {"results": cleaned, "count": len(cleaned), "query": query}


@register(
    "search_archive",
    {
        "description": "Search ARCHIVED (discarded) index entries — files that were removed or hidden via .agent.ignore. Only use when the user explicitly asks about content that is no longer in the live index.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword search query (FTS)",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 8)",
                },
            },
            "required": ["query"],
        },
    },
)
def search_archive(query: str, top_k: int | None = None) -> dict:
    global _archive_store
    if _config is None:
        return {"error": "Config not loaded."}
    if _archive_store is None:
        from agent.rag.archive import ArchiveStore

        _archive_store = ArchiveStore(_config.rag.archive_db_path)
    k = top_k or (_config.rag.top_k if _config else 8)
    rows = _archive_store.search(query, top_k=k)
    import datetime as _dt

    cleaned = []
    for r in rows:
        archived_at = r.get("archived_at")
        cleaned.append(
            {
                "path": r.get("path"),
                "name": r.get("name"),
                "language": r.get("language"),
                "node_type": r.get("node_type"),
                "start_line": r.get("start_line"),
                "end_line": r.get("end_line"),
                "content": (r.get("content") or "")[:800],
                "archived_at": _dt.datetime.fromtimestamp(archived_at).isoformat(
                    timespec="seconds"
                )
                if archived_at
                else None,
                "reason": r.get("reason"),
            }
        )
    return {
        "results": cleaned,
        "count": len(cleaned),
        "query": query,
        "source": "archive",
    }
