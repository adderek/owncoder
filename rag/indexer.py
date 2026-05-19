"""Index operations: prune, restore, and index_directory."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .chunker import (
    chunk_file, LANGUAGE_MAP, TREE_SITTER_LANG, CHUNK_NODE_TYPES,
)

if TYPE_CHECKING:
    from agent.config import RAGConfig, EmbeddingsConfig
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder

# Re-exported for callers using `from agent.rag.indexer import LANGUAGE_MAP`
__all__ = [
    "chunk_file", "LANGUAGE_MAP", "TREE_SITTER_LANG", "CHUNK_NODE_TYPES",
    "prune_index", "restore_paths", "index_directory", "pending_files",
]


def prune_index(
    root: str,
    store: "VectorStore",
    archive_store,
    reason: str = "stale",
) -> dict:
    """Detect indexed paths that no longer exist on disk or match .agent.ignore,
    move their rows into the archive, and delete them from the main index.
    Returns {archived, paths}.
    """
    root_path = Path(root).resolve()
    from agent.tools.rules import get_rules
    rules = get_rules()

    indexed = store.list_paths()
    stale: list[str] = []
    for stored in indexed:
        fpath = Path(stored)
        if not fpath.is_absolute():
            fpath = root_path / stored
        try:
            rel = str(fpath.resolve().relative_to(root_path))
        except ValueError:
            rel = stored
        missing = not fpath.exists()
        ignored = rules.ignore.matches(rel) if not rules.ignore.empty else False
        if missing or ignored:
            stale.append(stored)

    if not stale:
        return {"archived": 0, "paths": []}

    rows = store.rows_for_paths(stale)
    archived = archive_store.ingest(rows, reason=reason)
    for rel in stale:
        store.delete_by_path(rel)
    return {"archived": archived, "paths": stale}


def restore_paths(store: "VectorStore", archive_store, paths: list[str]) -> dict:
    """Move rows for the given paths from archive back into the main index."""
    rows = archive_store.pop_paths(paths)
    for r in rows:
        store.insert_raw(r)
    restored_paths = sorted({r["path"] for r in rows})
    return {"restored": len(rows), "paths": restored_paths}


def pending_files(
    root: str,
    store: "VectorStore",
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict:
    """Walk disk and compare against indexed mtimes. Returns counts without embedding."""
    root_path = Path(root).resolve()
    exclude = exclude or []
    default_exclude = {
        ".git", "__pycache__", "node_modules", "build", "dist",
        ".agent", ".venv", "venv", ".env",
    }
    all_exclude = default_exclude | {e.rstrip("/") for e in exclude}

    allowed_exts: set[str] | None = None
    if languages:
        allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in languages}

    from agent.tools.rules import get_rules
    rules = get_rules()

    indexed_mtimes = store.get_indexed_mtimes()
    total = 0
    pending = 0
    stale_paths: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in all_exclude]
        if not rules.ignore.empty:
            filtered = []
            for d in dirnames:
                rel = str((Path(dirpath) / d).relative_to(root_path))
                if not rules.ignore.matches(rel):
                    filtered.append(d)
            dirnames[:] = filtered
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if allowed_exts and fpath.suffix.lower() not in allowed_exts:
                continue
            if fpath.suffix.lower() not in LANGUAGE_MAP:
                continue
            rel = str(fpath.relative_to(root_path))
            if rules.ignore.matches(rel):
                continue
            total += 1
            mtime = fpath.stat().st_mtime
            # Stored paths may be absolute (old indexes) or relative (new indexes)
            stored_mtime = indexed_mtimes.get(rel) or indexed_mtimes.get(str(fpath.resolve()))
            if stored_mtime is None or abs(stored_mtime - mtime) >= 0.001:
                pending += 1
                stale_paths.append(rel)

    return {"total": total, "indexed": total - pending, "pending": pending, "paths": stale_paths}


_BATCH_SIZE = 32


def _chunk_and_embed(
    fpath: Path,
    rel: str,
    mtime: float,
    embedder: "Embedder",
    cfg: "RAGConfig",
    git_hash: str | None,
) -> tuple[str, list[dict], float]:
    """Chunk a file and embed all batches. Safe to call from a worker thread."""
    chunks = chunk_file(str(fpath), cfg)
    if not chunks:
        return rel, [], mtime
    for chunk in chunks:
        chunk["path"] = rel
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i:i + _BATCH_SIZE]
        texts = [c["content"] for c in batch]
        try:
            embeddings = embedder.embed(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
        except Exception:
            pass
        for chunk in batch:
            chunk["mtime"] = mtime
            chunk["git_hash"] = git_hash
    return rel, chunks, mtime


def index_directory(
    root: str,
    store: "VectorStore",
    embedder: "Embedder",
    cfg: "RAGConfig",
    languages: list[str] | None = None,
    exclude: list[str] | None = None,
    force: bool = False,
    git_hash: str | None = None,
    progress_cb=None,
    code_store=None,
) -> dict:
    root_path = Path(root).resolve()
    exclude = exclude or []
    default_exclude = {
        ".git", "__pycache__", "node_modules", "build", "dist",
        ".agent", ".venv", "venv", ".env",
    }
    all_exclude = default_exclude | {e.rstrip("/") for e in exclude}

    allowed_exts: set[str] | None = None
    if languages:
        allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in languages}

    from agent.tools.rules import get_rules
    rules = get_rules()

    files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in all_exclude]
        if not rules.ignore.empty:
            filtered = []
            for d in dirnames:
                rel = str((Path(dirpath) / d).relative_to(root_path))
                if not rules.ignore.matches(rel):
                    filtered.append(d)
            dirnames[:] = filtered
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if allowed_exts and fpath.suffix.lower() not in allowed_exts:
                continue
            if fpath.suffix.lower() not in LANGUAGE_MAP:
                continue
            rel = str(fpath.relative_to(root_path))
            if rules.ignore.matches(rel):
                continue
            files.append(fpath)

    indexed = 0
    skipped = 0
    total_chunks = 0
    embed_workers: int = getattr(getattr(embedder, "_cfg", None), "embed_workers", 1)

    if embed_workers <= 1:
        # Serial path: embed per batch → commit per batch (best crash recovery).
        for fpath in files:
            rel = str(fpath.relative_to(root_path))
            mtime = fpath.stat().st_mtime

            if not force:
                stored_mtime = store.get_mtime(rel)
                if stored_mtime is not None and abs(stored_mtime - mtime) < 0.001:
                    skipped += 1
                    continue

            store.delete_by_path(rel)
            chunks = chunk_file(str(fpath), cfg)
            if not chunks:
                continue

            for chunk in chunks:
                chunk["path"] = rel

            for i in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[i:i + _BATCH_SIZE]
                texts = [c["content"] for c in batch]
                try:
                    embeddings = embedder.embed(texts)
                    for chunk, emb in zip(batch, embeddings):
                        chunk["embedding"] = emb
                        chunk["mtime"] = mtime
                        chunk["git_hash"] = git_hash
                except Exception:
                    for chunk in batch:
                        chunk["mtime"] = mtime
                        chunk["git_hash"] = git_hash
                store.upsert_many(batch, fresh=True)

            total_chunks += len(chunks)
            indexed += 1

            if code_store is not None:
                _enqueue_for_summarization(code_store, chunks, rel, mtime, git_hash, force)
            if progress_cb:
                progress_cb(rel, len(chunks))

    else:
        # Parallel path: embed_workers concurrent embed requests, writes on main thread.
        # Chunking + embedding runs in the thread pool; SQLite writes stay on main thread.
        from concurrent.futures import ThreadPoolExecutor

        work: list[tuple[str, float, object]] = []  # (rel, mtime, future | None=skip)
        executor = ThreadPoolExecutor(max_workers=embed_workers)
        try:
            for fpath in files:
                rel = str(fpath.relative_to(root_path))
                mtime = fpath.stat().st_mtime
                if not force:
                    stored_mtime = store.get_mtime(rel)
                    if stored_mtime is not None and abs(stored_mtime - mtime) < 0.001:
                        skipped += 1
                        work.append((rel, mtime, None))
                        continue
                future = executor.submit(_chunk_and_embed, fpath, rel, mtime, embedder, cfg, git_hash)
                work.append((rel, mtime, future))

            for rel, mtime, future in work:
                if future is None:
                    continue
                _, chunks, _ = future.result()
                if not chunks:
                    continue
                store.delete_by_path(rel)
                for i in range(0, len(chunks), _BATCH_SIZE):
                    store.upsert_many(chunks[i:i + _BATCH_SIZE], fresh=True)
                total_chunks += len(chunks)
                indexed += 1
                if code_store is not None:
                    _enqueue_for_summarization(code_store, chunks, rel, mtime, git_hash, force)
                if progress_cb:
                    progress_cb(rel, len(chunks))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return {"indexed": indexed, "skipped": skipped, "chunks": total_chunks, "files": len(files)}


def _enqueue_for_summarization(code_store, chunks: list[dict], path: str, mtime: float, git_hash, force: bool) -> None:
    """Upsert changed chunks into code_store with status='pending'."""
    import hashlib

    file_checksum = hashlib.sha256(path.encode()).hexdigest()[:16]
    existing_file = code_store.get_file_record(path)

    # Build per-chunk checksums and detect which ones changed
    for chunk in chunks:
        content = chunk.get("content", "")
        obj_cs = hashlib.sha256(content.encode()).hexdigest()[:16]
        chunk_id = chunk["id"]

        if not force:
            existing = code_store.get_unit(chunk_id)
            if existing and existing.get("object_checksum") == obj_cs:
                continue  # content unchanged — keep existing description

        code_store.upsert_unit({
            "id": chunk_id,
            "path": path,
            "language": chunk.get("language"),
            "node_type": chunk.get("node_type"),
            "name": chunk.get("name"),
            "level": 0,
            "start_line": chunk.get("start_line"),
            "end_line": chunk.get("end_line"),
            "object_checksum": obj_cs,
            "parent_id": chunk.get("parent_chunk_id"),
            "status": "pending",
            "mtime": mtime,
            "git_hash": git_hash,
        })

    code_store.set_file_record(path, file_checksum)
