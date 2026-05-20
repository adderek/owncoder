from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules

if TYPE_CHECKING:
    from agent.config import Config
    from agent.data_provider import DataProviderProtocol


_config = None
_data_provider: "DataProviderProtocol | None" = None
_archive_store = None


def setup(config, data_provider) -> None:
    global _config, _data_provider, _archive_store
    _config = config
    _data_provider = data_provider
    _archive_store = None  # lazy-open on first archive search


@register(
    "search_code",
    {
        "description": (
            "Search the codebase index using semantic + keyword search. "
            "Results are excerpts from a summarized index — use them to locate files and line ranges, "
            "then verify with read_file before making changes. "
            "Falls back to grep when the index is not ready."
        ),
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
    if _data_provider is None or not _data_provider.is_available():
        from agent.tools.search.grep import grep_code as _grep_code
        result = _grep_code(query, fixed_string=False)
        result["note"] = "Index not ready — results are from grep fallback. Run 'agent index' to build the semantic index."
        return result

    k = top_k or (_config.rag.top_k if _config else 8)

    results = _data_provider.search(query, top_k=k)

    # Clean up results for LLM consumption
    _CONTENT_LIMIT = 800
    cleaned = []
    for r in results:
        raw = r.get("content", "")
        truncated = len(raw) > _CONTENT_LIMIT
        cleaned.append(
            {
                "path": r.get("path"),
                "name": r.get("name"),
                "language": r.get("language"),
                "node_type": r.get("node_type"),
                "start_line": r.get("start_line"),
                "end_line": r.get("end_line"),
                "content": raw[:_CONTENT_LIMIT],
                "truncated": truncated,
            }
        )

    # ASM semantic search via DataProvider.asm_search().
    for r in _data_provider.asm_search(query, top_k=k):
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
