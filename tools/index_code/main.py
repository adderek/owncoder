"""index_code tool — let the agent trigger indexing of a path mid-session.

After indexing completes, rewires the live data_provider so semantic search
becomes available immediately (without restarting the session).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config
    from agent.data_provider import LocalDataProvider

logger = logging.getLogger(__name__)

_config: "Config | None" = None
_data_provider: "LocalDataProvider | None" = None


def setup(config: "Config", data_provider: "LocalDataProvider | None" = None) -> None:
    global _config, _data_provider
    _config = config
    _data_provider = data_provider


def _validate_path(root: str, working_dir: str) -> Path | None:
    """Return resolved Path if it's inside working_dir, else None."""
    try:
        resolved = (Path(working_dir) / root).resolve()
        wd_resolved = Path(working_dir).resolve()
        resolved.relative_to(wd_resolved)  # raises ValueError if outside
        return resolved
    except (ValueError, Exception):
        return None


@register(
    "index_code",
    {
        "description": (
            "Index a directory for semantic code search. "
            "Call this when the user approves indexing a path. "
            "After indexing, semantic search becomes available in this session. "
            "Use '.' to index the whole project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to index, relative to project root. Use '.' for whole project.",
                },
                "languages": {
                    "type": "string",
                    "description": "Optional comma-separated language filter: py,js,kt,cpp,asm",
                },
            },
            "required": ["path"],
        },
    },
)
def index_code(path: str, languages: str | None = None) -> dict[str, Any]:
    if _config is None:
        return {"error": "Tool not configured. Session setup issue."}

    working_dir = _config.tools.working_dir

    if path.strip() in (".", "", "./"):
        index_root = working_dir
    else:
        validated = _validate_path(path.strip(), working_dir)
        if validated is None:
            return {"error": f"Path '{path}' is outside the project root or invalid."}
        index_root = str(validated)

    lang_list = [l.strip() for l in languages.split(",") if l.strip()] if languages else None

    try:
        from rich.console import Console
        from agent.cli.index import _run_indexing

        console = Console()
        stats = _run_indexing(
            config=_config,
            console=console,
            root=index_root,
            languages=lang_list,
        )
    except Exception as exc:
        logger.exception("index_code: _run_indexing failed")
        return {"error": str(exc)}

    # Rewire live data_provider so this session gains semantic search.
    rewired = False
    if _data_provider is not None:
        try:
            from agent.rag.store import VectorStore
            from agent.rag.embedder import Embedder

            new_store = VectorStore(_config.rag)
            new_embedder = _data_provider._embedder or Embedder(_config.embeddings)
            _data_provider._store = new_store
            _data_provider._embedder = new_embedder
            rewired = True
        except Exception as exc:
            logger.warning("index_code: failed to rewire data_provider: %s", exc)

    rel = Path(index_root).relative_to(working_dir) if index_root != working_dir else Path(".")
    return {
        "indexed": stats["indexed"],
        "skipped": stats["skipped"],
        "chunks": stats["chunks"],
        "path": str(rel),
        "semantic_search_active": rewired,
        "note": (
            "Semantic search now active for this session."
            if rewired
            else "Indexing complete. Semantic search will be available on next session start."
        ),
    }


@register(
    "index_status",
    {
        "description": (
            "Is the code index fresh and consistent? Reports files/chunks, embedding model and "
            "coverage, when it was last indexed, files changed since then, and any consistency "
            "defects. Use this instead of querying .agent/*.db with sqlite — the raw tables "
            "include internal bookkeeping that looks contradictory out of context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "check_files": {
                    "type": "boolean",
                    "description": "Also stat every project file to list ones changed since indexing (default true).",
                },
            },
        },
    },
)
def index_status(check_files: bool = True) -> dict:
    if _config is None:
        return {"error": "index_status not configured"}
    import dataclasses
    from agent.memory.overview import rag_db_path
    from agent.rag import health
    from agent.rag.store import VectorStore
    db = rag_db_path(_config)
    if not db.exists():
        return {"status": "no code index yet — index_code '.' builds it"}
    store = VectorStore(dataclasses.replace(_config.rag, db_path=str(db)))
    try:
        model = getattr(getattr(_config, "embeddings", None), "model", "") or ""
        h = health.check(store, _config.tools.working_dir, embed_model=model,
                         files=bool(check_files), cfg=_config.rag)
        return {"status": health.summary(h), "defects": len(h["defects"]),
                "changed_files": h.get("pending")}
    except Exception as exc:
        logger.warning("index_status failed", exc_info=True)
        return {"error": f"index_status failed: {exc}"}
    finally:
        store.close()
