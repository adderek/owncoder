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
            "Semantic + keyword search over codebase index. "
            "Results are index excerpts — locate files/lines, then read_file to verify before editing. "
            "Falls back to grep when index not ready. "
            "Best first step when the location or exact name is unknown. "
            "Exact text → grep_code; known symbol → find_symbol."
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
def _mark_stale(results: list[dict]) -> list[str]:
    """Flag hits whose file changed (or vanished) since indexing; one stat per file."""
    import os
    root = getattr(getattr(_config, "tools", None), "working_dir", None) or "."
    getter = getattr(_data_provider, "indexed_mtime", None)
    if getter is None:
        return []
    seen: dict[str, str] = {}
    for r in results:
        path = r.get("path") or ""
        if not path or r.get("language") == "asm":
            continue
        if path not in seen:
            indexed = getter(path)
            full = path if os.path.isabs(path) else os.path.join(root, path)
            try:
                disk = os.stat(full).st_mtime
                seen[path] = "changed" if indexed is not None and abs(disk - indexed) > 1.0 else ""
            except OSError:
                seen[path] = "deleted"
        if seen[path]:
            r["stale"] = seen[path] + " since indexed"
    return [p for p, v in seen.items() if v]


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

    stale = _mark_stale(cleaned)
    result = {"results": cleaned, "count": len(cleaned), "query": query}
    if stale:
        result["stale_note"] = (
            f"{len(stale)} result file(s) changed on disk since they were indexed "
            f"({', '.join(stale[:5])}{' …' if len(stale) > 5 else ''}) — read the file for current "
            "content; index_status shows overall freshness.")
    # Tell the model when it is getting keyword results from a semantic tool —
    # otherwise it reads a thin result set as "nothing matches" and stops.
    mismatch = getattr(_data_provider, "embedding_mismatch", lambda: "")()
    if mismatch == "dims":
        result["note"] = (
            "Semantic search is disabled: the index holds vectors of a different "
            "dimensionality than the configured embedding model, so these results "
            "come from keyword search only. Run 'agent index --reembed' to repair."
        )
    elif mismatch == "model":
        result["note"] = (
            "The index vectors come from a different embedding model of the same "
            "width — results are keyword-searchable but ranking is degraded. "
            "Run 'agent index --reembed' to repair."
        )
    return result


@register(
    "search_archive",
    {
        "description": "Search archived/discarded index entries (removed or .agent.ignore-hidden files). Only when user asks about content no longer in live index.",
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
