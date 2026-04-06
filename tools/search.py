from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder

_config = None
_store = None
_embedder = None


def setup(config, store, embedder) -> None:
    global _config, _store, _embedder
    _config = config
    _store = store
    _embedder = embedder


@register("search_code", {
    "description": "Search the codebase using semantic + keyword search. Always call this before read_file to find relevant code.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language or keyword search query"},
            "top_k": {"type": "integer", "description": "Number of results to return (default: 8)"},
        },
        "required": ["query"],
    },
})
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
        cleaned.append({
            "path": r.get("path"),
            "name": r.get("name"),
            "language": r.get("language"),
            "node_type": r.get("node_type"),
            "start_line": r.get("start_line"),
            "end_line": r.get("end_line"),
            "content": r.get("content", "")[:800],  # trim for context
        })

    return {"results": cleaned, "count": len(cleaned), "query": query}
