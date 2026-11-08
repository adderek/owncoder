"""Index operations: prune, restore, and index_directory."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from agent.security import path_policy
from typing import TYPE_CHECKING

from .chunker import (
    chunk_file, LANGUAGE_MAP, TREE_SITTER_LANG, CHUNK_NODE_TYPES,
    is_text_candidate, TEXT_MAX_BYTES,
)

if TYPE_CHECKING:
    from agent.config import RAGConfig, EmbeddingsConfig
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder

log = logging.getLogger(__name__)

# Re-exported for callers using `from agent.rag.indexer import LANGUAGE_MAP`
__all__ = [
    "chunk_file", "LANGUAGE_MAP", "TREE_SITTER_LANG", "CHUNK_NODE_TYPES",
    "prune_index", "restore_paths", "index_directory", "pending_files",
]


def _wanted_file(fpath: Path, allowed_exts: set[str] | None, include_text: bool, cfg=None) -> bool:
    """Should the walk index *fpath*?

    Code files (LANGUAGE_MAP) always qualify (subject to the language filter);
    generic text files (.md, .toml, .example, .template, Makefile, ...) qualify
    when rag.index_text_files is on (the default) and *include_text* is true
    (no language filter given, or the filter names "text").
    """
    ext = fpath.suffix.lower()
    if ext in LANGUAGE_MAP:
        return allowed_exts is None or ext in allowed_exts
    if not include_text:
        return False
    if cfg is not None and not getattr(cfg, "index_text_files", True):
        return False
    max_bytes = getattr(cfg, "text_max_bytes", TEXT_MAX_BYTES) if cfg else TEXT_MAX_BYTES
    return is_text_candidate(fpath, max_bytes)


def _fast_file_checksum(path: str) -> str | None:
    """Return sha256[:16] of file bytes, or None on error."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return None


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

    # Chunkless rows in file_mtimes too: walking chunks alone left deleted,
    # ignored and mis-rooted empty files behind forever.
    indexed = store.list_paths() + store.list_mtime_only_paths()
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
    cfg=None,
) -> dict:
    """Walk disk and compare against indexed mtimes. Returns counts without embedding."""
    root_path = Path(root).resolve()
    exclude = exclude or []
    # One shared list of prunable directories (security.path_policy), plus the
    # indexer's own: ".env" is a directory here, not the secrets file.
    default_exclude = set(path_policy.hidden_dir_names()) | {".env"}
    all_exclude = default_exclude | {e.rstrip("/") for e in exclude}

    allowed_exts: set[str] | None = None
    if languages:
        allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in languages}
    include_text = languages is None or "text" in (languages or [])

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
            if not _wanted_file(fpath, allowed_exts, include_text, cfg):
                continue
            rel = str(fpath.relative_to(root_path))
            if rules.ignore.matches(rel):
                continue
            mtime = _mtime_or_none(fpath)
            if mtime is None:
                continue
            total += 1
            # Stored paths may be absolute (old indexes) or relative (new indexes)
            stored_mtime = indexed_mtimes.get(rel) or indexed_mtimes.get(str(fpath.resolve()))
            if stored_mtime is None or abs(stored_mtime - mtime) >= 0.001:
                pending += 1
                stale_paths.append(rel)

    return {"total": total, "indexed": total - pending, "pending": pending, "paths": stale_paths}


def _mtime_or_none(fpath: Path) -> float | None:
    """mtime, or None if the file vanished since the walk listed it.

    The walk and the stat are not atomic: runtime files (presence beacons,
    editor temp files) can be deleted in between.
    """
    try:
        return fpath.stat().st_mtime
    except FileNotFoundError:
        return None


_BATCH_SIZE = 32


def _chunk_and_embed(
    fpath: Path,
    rel: str,
    mtime: float,
    embedder: "Embedder",
    cfg: "RAGConfig",
    git_hash: str | None,
) -> tuple[str, list[dict], float, int]:
    """Chunk a file and embed all batches. Safe to call from a worker thread.

    Returns (rel, chunks, mtime, embedded) — `embedded` counts the chunks that
    got a vector. A failed batch still returns its chunks so they stay
    keyword-searchable; the caller aggregates the shortfall into one warning.
    """
    chunks = chunk_file(str(fpath), cfg)
    if not chunks:
        return rel, [], mtime, 0
    for chunk in chunks:
        chunk["path"] = rel
    embedded = 0
    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i:i + _BATCH_SIZE]
        texts = [c["content"] for c in batch]
        try:
            embeddings = embedder.embed(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk["embedding"] = emb
                if emb:  # [] is the Embedder's per-text failure marker
                    embedded += 1
        except Exception:
            log.debug("embedding batch of %d failed", len(batch), exc_info=True)
        for chunk in batch:
            chunk["mtime"] = mtime
            chunk["git_hash"] = git_hash
    return rel, chunks, mtime, embedded


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

    # Compare against the stamp on the existing index. Two distinct cases,
    # because the remedy differs: a dimension change means stored vectors cannot
    # be compared with query vectors at all, while a same-dimension model/quant
    # swap (bge-m3 q4_k_m vs q8_0) only degrades ranking.
    # Read-only — the stamp is committed at the end, and only if embeddings were
    # actually produced (see below).
    emb_cfg = getattr(embedder, "_cfg", None)
    emb_model = getattr(emb_cfg, "model", "") if emb_cfg is not None else ""
    if emb_model:
        prev_model, prev_dims = store.compare_embedding_model(emb_model)
        if prev_model:
            if prev_dims is not None and prev_dims != emb_cfg.dimensions:
                log.error(
                    "Embedding model changed: index built with %r (%d dims), now "
                    "embedding with %r (%d dims) — incompatible vectors, vector "
                    "search cannot use the old chunks. Re-apply embeddings with "
                    "`agent index --reembed`, or rebuild from source with "
                    "`agent init --force`.",
                    prev_model, prev_dims, emb_model, emb_cfg.dimensions,
                )
            else:
                log.warning(
                    "Embedding model changed: index built with %r, now embedding "
                    "with %r (same dims, different vectors). Mixed vectors degrade "
                    "search — run `agent index --reembed` to replace every vector "
                    "without re-reading the tree.",
                    prev_model, emb_model,
                )
    # An empty vector table carries no other model's vectors, so the stamp can
    # be committed with the first vector written rather than at the end. A pass
    # cut short (agent exit, Ctrl-C) used to leave Qwen3 vectors under a stale
    # 'bge-m3' stamp, which the next startup reported as a model mismatch.
    stamp_pending = [bool(emb_model) and not _has_vectors(store)]

    def _stamp_early() -> None:
        if stamp_pending[0]:
            stamp_pending[0] = False
            store.record_embedding_model(emb_model, getattr(emb_cfg, "base_url", ""))

    # One shared list of prunable directories (security.path_policy), plus the
    # indexer's own: ".env" is a directory here, not the secrets file.
    default_exclude = set(path_policy.hidden_dir_names()) | {".env"}
    all_exclude = default_exclude | {e.rstrip("/") for e in exclude}

    allowed_exts: set[str] | None = None
    if languages:
        allowed_exts = {ext for ext, lang in LANGUAGE_MAP.items() if lang in languages}
    include_text = languages is None or "text" in (languages or [])

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
            if not _wanted_file(fpath, allowed_exts, include_text, cfg):
                continue
            rel = str(fpath.relative_to(root_path))
            if rules.ignore.matches(rel):
                continue
            files.append(fpath)

    indexed = 0
    skipped = 0
    total_chunks = 0
    dedup_same = 0
    dedup_cross = 0
    embed_ok = 0      # chunks that got a vector this run
    embed_failed = 0  # chunks stored without one (keyword-searchable only)
    embed_workers: int = getattr(getattr(embedder, "_cfg", None), "embed_workers", 1)

    # Pre-load in-memory dedup state from code_store to avoid per-chunk DB queries.
    dedup_cache: dict | None = None          # {obj_cs: unit_dict} for O(1) leaf lookup
    described_file_checksums: set | None = None  # content_checksums of fully-described files
    if code_store is not None and not force:
        dedup_cache = code_store.load_described_checksum_map()
        described_file_checksums = code_store.get_described_content_checksums()

    if embed_workers <= 1:
        # Serial path: embed per batch → commit per batch (best crash recovery).
        for fpath in files:
            rel = str(fpath.relative_to(root_path))
            mtime = _mtime_or_none(fpath)
            if mtime is None:
                store.delete_by_path(rel)
                continue
            abs_path = str(fpath)

            if not force:
                stored_mtime = store.get_mtime(rel)
                if stored_mtime is not None and abs(stored_mtime - mtime) < 0.001:
                    skipped += 1
                    continue

            # File-level skip: if content_checksum matches a fully-described file,
            # skip chunking + embedding entirely.
            if described_file_checksums is not None:
                try:
                    file_cs = _fast_file_checksum(abs_path)
                    if file_cs and file_cs in described_file_checksums:
                        store.touch_mtime(rel, mtime)
                        skipped += 1
                        continue
                except OSError:
                    pass

            store.delete_by_path(rel)
            if abs_path != rel:
                store.delete_by_path(abs_path)  # clean up legacy absolute-path entries
            chunks = chunk_file(abs_path, cfg)
            if not chunks:
                store.set_file_mtime(rel, mtime)
                continue

            for chunk in chunks:
                chunk["path"] = rel

            for i in range(0, len(chunks), _BATCH_SIZE):
                batch = chunks[i:i + _BATCH_SIZE]
                texts = [c["content"] for c in batch]
                try:
                    embeddings = embedder.embed(texts)
                    got = 0
                    for chunk, emb in zip(batch, embeddings):
                        chunk["embedding"] = emb
                        chunk["mtime"] = mtime
                        chunk["git_hash"] = git_hash
                        got += 1 if emb else 0
                    embed_ok += got
                    embed_failed += len(batch) - got
                    if got:
                        _stamp_early()
                except Exception:
                    embed_failed += len(batch)
                    log.debug("embedding batch of %d failed", len(batch), exc_info=True)
                    for chunk in batch:
                        chunk["mtime"] = mtime
                        chunk["git_hash"] = git_hash
                store.upsert_many(batch, fresh=True)

            total_chunks += len(chunks)
            indexed += 1

            ds = None
            if code_store is not None:
                ds = _enqueue_for_summarization(
                    code_store, chunks, rel, mtime, git_hash, force,
                    abs_path=abs_path, dedup_cache=dedup_cache,
                )
                dedup_same += ds["dedup_same"]
                dedup_cross += ds["dedup_cross"]
                if dedup_cache is not None:
                    dedup_cache.update(ds.get("new_described", {}))
            if progress_cb:
                progress_cb(rel, len(chunks), dedup=ds)

    else:
        # Parallel path: embed_workers concurrent embed requests, writes on main thread.
        # Chunking + embedding runs in the thread pool; SQLite writes stay on main thread.
        from concurrent.futures import ThreadPoolExecutor

        work: list[tuple[str, str, float, object]] = []  # (rel, abs_path, mtime, future|None)
        executor = ThreadPoolExecutor(max_workers=embed_workers)
        try:
            for fpath in files:
                rel = str(fpath.relative_to(root_path))
                abs_path = str(fpath)
                mtime = _mtime_or_none(fpath)
                if mtime is None:
                    store.delete_by_path(rel)
                    continue
                if not force:
                    stored_mtime = store.get_mtime(rel)
                    if stored_mtime is not None and abs(stored_mtime - mtime) < 0.001:
                        skipped += 1
                        work.append((rel, abs_path, mtime, None))
                        continue
                    # File-level skip via content checksum.
                    if described_file_checksums is not None:
                        try:
                            file_cs = _fast_file_checksum(abs_path)
                            if file_cs and file_cs in described_file_checksums:
                                store.touch_mtime(rel, mtime)
                                skipped += 1
                                work.append((rel, abs_path, mtime, None))
                                continue
                        except OSError:
                            pass
                future = executor.submit(_chunk_and_embed, fpath, rel, mtime, embedder, cfg, git_hash)
                work.append((rel, abs_path, mtime, future))

            for rel, abs_path, mtime, future in work:
                if future is None:
                    continue
                _, chunks, _, emb_n = future.result()
                embed_ok += emb_n
                if emb_n:
                    _stamp_early()
                embed_failed += len(chunks) - emb_n
                if not chunks:
                    store.set_file_mtime(rel, mtime)
                    continue
                store.delete_by_path(rel)
                if abs_path != rel:
                    store.delete_by_path(abs_path)  # clean up legacy absolute-path entries
                for i in range(0, len(chunks), _BATCH_SIZE):
                    store.upsert_many(chunks[i:i + _BATCH_SIZE], fresh=True)
                total_chunks += len(chunks)
                indexed += 1
                ds = None
                if code_store is not None:
                    ds = _enqueue_for_summarization(
                        code_store, chunks, rel, mtime, git_hash, force,
                        abs_path=abs_path, dedup_cache=dedup_cache,
                    )
                    dedup_same += ds["dedup_same"]
                    dedup_cross += ds["dedup_cross"]
                    if dedup_cache is not None:
                        dedup_cache.update(ds.get("new_described", {}))
                if progress_cb:
                    progress_cb(rel, len(chunks), dedup=ds)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    if embed_failed:
        log.warning(
            "%d of %d chunks were stored without vectors — they stay "
            "keyword-searchable only. Check the embeddings endpoint.",
            embed_failed, embed_ok + embed_failed,
        )
    if emb_model and embed_ok:
        # Only now is the stamp truthful: embeddings were actually produced, so
        # the index really is filled with this model's vectors.
        store.record_embedding_model(emb_model, getattr(emb_cfg, "base_url", ""))
    elif emb_model:
        log.warning(
            "no embeddings produced (%d chunks failed) — embedding-model stamp "
            "left unchanged",
            embed_failed,
        )

    # Which root the stored relative paths are relative to — without it a
    # re-rooted index (agent/ vs repo root) cannot tell its stale rows apart.
    store.set_meta("index_root", str(root_path))
    store.set_meta("last_index_at", str(time.time()))

    return {
        "indexed": indexed,
        "skipped": skipped,
        "chunks": total_chunks,
        "files": len(files),
        "dedup_same": dedup_same,
        "dedup_cross": dedup_cross,
    }


def _has_vectors(store) -> bool:
    fn = getattr(store, "has_vectors", None)
    return bool(fn()) if fn else False


def _good(vectors, n: int, dims: int | None = None) -> list[bool]:
    """Per-row usability: the Embedder marks a failed text with [] instead of
    raising, so a 'successful' call can still carry no vector at all."""
    vectors = list(vectors or [])
    vectors += [[]] * (n - len(vectors))
    return [bool(v) and (dims is None or len(v) == dims) for v in vectors[:n]]


def progress_text(done: int, total: int, elapsed: float, failed: int = 0) -> str:
    """'4192/10257 (41%) 3.2/s, ETA 31m' — shared by every re-embed front end."""
    pct = 100.0 * done / total if total else 100.0
    rate = done / elapsed if elapsed > 0 else 0.0
    out = f"{done}/{total} ({pct:.0f}%)"
    if rate > 0:
        left = (total - done) / rate
        if left >= 5400:
            eta = f"{left / 3600:.1f}h"
        elif left >= 90:
            eta = f"{left / 60:.0f}m"
        else:
            eta = f"{left:.0f}s"
        out += f" {rate:.1f}/s, ETA {eta}"
    if failed:
        out += f", {failed} failed"
    return out


def embed_missing(store, embedder, limit: int = 0) -> dict:
    """Embed chunks that have no vector, from their stored text.

    A file indexed while the embedder was down keeps its chunks (keyword
    search works) and its mtime, so nothing ever retried it — those chunks
    stayed invisible to semantic search for good. Stops at the first batch that
    yields nothing: a dead endpoint is not hammered, the next pass resumes.
    Callers check the index's embedding model matches *embedder* first.
    """
    rows = store.missing_vector_texts(limit)
    out = {"missing": len(rows), "embedded": 0, "failed": 0}
    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i:i + _BATCH_SIZE]
        try:
            vectors = embedder.embed([text for _, text in batch])
        except Exception as e:
            log.warning("embed-missing: batch failed (%s) — %d chunks left for the next pass",
                        type(e).__name__, len(rows) - out["embedded"])
            out["failed"] = len(rows) - out["embedded"]
            break
        ok = _good(vectors, len(batch), getattr(store, "_vec_dims", None))
        written = store.write_embeddings(
            [(cid, v) for (cid, _), v, g in zip(batch, vectors, ok) if g])
        out["embedded"] += written
        if not written:
            out["failed"] = len(rows) - out["embedded"]
            break
    return out


def _cos(a, b) -> float:
    import math
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


# Same model + quant on another build/device agrees to ~1.0; a different model
# does not. Measured 2026-11-08: Qwen3-0.6B Q8_0 (CPU) vs the stored vectors
# 1.000 on 60 random chunks; stored bge-m3 vs Qwen3 on the same text ~0.0.
SAME_MODEL_COS = 0.99


def same_space(reference: list[list[float]], candidate: list[list[float]]) -> float:
    """Lowest cosine between paired vectors (0.0 when anything is missing)."""
    if not reference or len(reference) != len(candidate):
        return 0.0
    if not all(reference) or not all(candidate):
        return 0.0
    return min(_cos(a, b) for a, b in zip(reference, candidate))


def _label(embedder) -> str:
    return (getattr(embedder, "endpoint", None)
            or getattr(getattr(embedder, "_cfg", None), "base_url", "?"))


def reembed_all(store, embedder, cfg, progress_cb=None, extra_embedders=()) -> dict:
    """Replace every stored vector with one produced by *embedder*.

    The vector half of the index only: chunk text is read back from the index,
    nothing is re-chunked, and the summarization queue is not touched.
    `agent init --force` re-reads and re-chunks the whole tree and then
    re-summarizes it — orders of magnitude more work for the same vectors, and
    the wrong tool for an embedding-model swap.

    The old vectors are dropped only after a probe batch has proved the embedder
    answers *with vectors*, so a dead endpoint leaves the index intact rather
    than empty. The stamp is committed right after the reset — the table then
    only ever holds this model's vectors — so an interrupted run is finished by
    `embed_missing` (`agent index --update`) instead of starting over.

    *extra_embedders* (e.g. temporary GPU servers) share the work, one batch in
    flight each; an extra whose vectors are not in the primary's space (cosine
    < SAME_MODEL_COS on the probe) is dropped before it writes anything.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    emb_cfg = getattr(embedder, "_cfg", None)
    emb_model = getattr(emb_cfg, "model", "") if emb_cfg is not None else ""
    rows = store.all_chunk_texts()
    result = {"chunks": len(rows), "embedded": 0, "failed": 0, "aborted": False,
              "workers": [], "dropped": [], "elapsed": 0.0}
    if not rows:
        return result

    t0 = time.monotonic()
    probe = rows[:_BATCH_SIZE]
    probe_texts = [text for _, text in probe]
    try:
        vectors = embedder.embed(probe_texts)
    except Exception as e:
        log.error(
            "re-embed aborted: embedder did not answer (%s: %s) — index left "
            "untouched, %d chunks keep their previous vectors",
            type(e).__name__, e, len(rows),
        )
        result["aborted"] = True
        return result
    if not vectors or not all(_good(vectors, len(probe))):
        log.error("re-embed aborted: embedder returned no (or empty) vectors — index untouched")
        result["aborted"] = True
        return result

    dims = len(vectors[0])
    if getattr(emb_cfg, "dimensions", 0) and emb_cfg.dimensions != dims:
        log.warning(
            "embedder returned %d dims but config says %d — sizing the vector "
            "table from the returned width",
            dims, emb_cfg.dimensions,
        )

    workers = [embedder]
    for extra in extra_embedders:
        why = ""
        try:
            got = extra.embed(probe_texts[:4])
        except Exception as e:
            got, why = [], f"{type(e).__name__}: {e}"
        sim = same_space(vectors[:4], got)
        if sim >= SAME_MODEL_COS:
            workers.append(extra)
        else:
            why = why or f"vectors differ from the primary (cos {sim:.3f})"
            log.warning("re-embed: dropping helper %s — %s", _label(extra), why)
            result["dropped"].append(f"{_label(extra)}: {why}")
    result["workers"] = [_label(w) for w in workers]

    # Past this point the old vectors are being replaced, not compared.
    store.reset_vectors(dims)
    if emb_model:
        store.record_embedding_model(emb_model, getattr(emb_cfg, "base_url", ""))
    result["embedded"] = store.write_embeddings(
        [(cid, v) for (cid, _), v in zip(probe, vectors)])

    last_log = [t0]

    def _report() -> None:
        if progress_cb:
            progress_cb(result["embedded"], len(rows))
        now = time.monotonic()
        if now - last_log[0] >= 30 or result["embedded"] + result["failed"] >= len(rows):
            last_log[0] = now
            log.info("re-embed: %s", progress_text(
                result["embedded"], len(rows), now - t0, result["failed"]))

    _report()

    # Work queue of (batch, attempt). Rows that come back without a vector are
    # retried — on whichever worker is free — up to 3 attempts in total.
    pending = [(rows[i:i + _BATCH_SIZE], 0)
               for i in range(_BATCH_SIZE, len(rows), _BATCH_SIZE)]
    pending.reverse()  # pop() from the end keeps index order
    strikes = {id(w): 0 for w in workers}
    idle = list(workers)
    inflight: dict = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        while pending or inflight:
            while pending and idle:
                w = idle.pop(0)
                batch, attempt = pending.pop()
                fut = pool.submit(w.embed, [t for _, t in batch])
                inflight[fut] = (w, batch, attempt)
            if not inflight:
                break  # every worker retired; leftovers stay pending
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                w, batch, attempt = inflight.pop(fut)
                try:
                    vecs = fut.result()
                except Exception:
                    vecs = []
                ok = _good(vecs, len(batch), dims)
                result["embedded"] += store.write_embeddings(
                    [(cid, v) for (cid, _), v, g in zip(batch, vecs, ok) if g])
                bad = [row for row, g in zip(batch, ok) if not g]
                if bad:
                    strikes[id(w)] += 1
                    if attempt + 1 < 3:
                        pending.append((bad, attempt + 1))
                    else:
                        log.warning(
                            "re-embed: %d chunks failed 3 times — they keep no vector "
                            "and stay keyword-searchable only. `agent index --update` "
                            "retries them.", len(bad))
                        result["failed"] += len(bad)
                else:
                    strikes[id(w)] = 0
                # A worker failing 3 batches in a row is retired, unless it is
                # the last one standing (then the attempt cap ends the run).
                others = len(idle) + len(inflight)
                if strikes[id(w)] >= 3 and others:
                    log.warning("re-embed: retiring %s after 3 failed batches", _label(w))
                    result["dropped"].append(f"{_label(w)}: retired after 3 failed batches")
                else:
                    idle.append(w)
                _report()

    result["failed"] += sum(len(b) for b, _ in pending)
    result["elapsed"] = time.monotonic() - t0
    return result


def _file_rollup(code_store, chunks: list[dict], path: str, mtime: float, git_hash) -> None:
    """A level-1 unit over the file's top-level units, left 'stale' for the
    background worker to summarise once its parts are described."""
    import hashlib
    tops = [c for c in chunks if not c.get("parent_chunk_id")]
    if len(tops) < 2:
        return                       # one part: its own description is the file's
    fid = hashlib.sha256(f"{path}:file".encode()).hexdigest()[:16]
    code_store.upsert_unit({
        "id": fid, "path": path, "language": chunks[0].get("language"),
        "node_type": "file", "name": Path(path).name, "level": 1,
        "start_line": min(c.get("start_line") or 1 for c in chunks),
        "end_line": max(c.get("end_line") or 0 for c in chunks),
        "object_checksum": hashlib.sha256(
            "".join(c["id"] for c in tops).encode()).hexdigest()[:16],
        "status": "stale", "mtime": mtime, "git_hash": git_hash,
    })
    ids = [c["id"] for c in sorted(tops, key=lambda c: c.get("start_line") or 0)]
    code_store.set_parent(ids, fid)
    code_store.upsert_children(fid, ids)


def _enqueue_for_summarization(
    code_store, chunks: list[dict], path: str, mtime: float, git_hash, force: bool,
    abs_path: str | None = None,
    dedup_cache: dict | None = None,
) -> dict:
    """Upsert changed chunks into code_store; skip LLM for content-identical chunks.

    Returns {"dedup_same": int, "dedup_cross": int, "pending": int, "new_described": dict}.
    new_described is {obj_cs: unit_dict} for chunks resolved via cross-path dedup this call,
    so the caller can update the in-memory cache.
    """
    import hashlib

    dedup_same = 0
    dedup_cross = 0
    pending = 0
    new_described: dict[str, dict] = {}

    # Snapshot existing leaf units BEFORE delete so we can skip re-summarization
    # for chunks whose content is unchanged.
    prior: dict[str, dict] = {} if force else code_store.get_unit_checksums_for_file(path)

    # Wipe old units to avoid orphan accumulation when chunk boundaries shift.
    code_store.delete_units_for_file(path)

    file_checksum = hashlib.sha256(path.encode()).hexdigest()[:16]

    # Compute actual file content checksum + size (reuse from cache if we already read this file).
    content_checksum: str | None = None
    file_size: int | None = None
    try:
        fpath = abs_path or path
        stat = os.stat(fpath)
        file_size = stat.st_size
        with open(fpath, "rb") as fh:
            content_checksum = hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        pass

    for chunk in chunks:
        content = chunk.get("content", "")
        obj_cs = hashlib.sha256(content.encode()).hexdigest()[:16]
        chunk_id = chunk["id"]

        # 1. Same-path skip: chunk existed with same content and was already described.
        if not force:
            prev = prior.get(chunk_id)
            if prev and prev.get("object_checksum") == obj_cs and prev.get("status") == "described":
                # Re-insert with preserved description to avoid orphaning.
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
                    "node_checksum": prev.get("node_checksum"),
                    "description": prev.get("description"),
                    "inferred_name": prev.get("inferred_name"),
                    "confidence": prev.get("confidence"),
                    # The model's text, when a reviewer's sits in description.
                    "description_model": prev.get("description_model"),
                    "name_model": prev.get("name_model"),
                    "parent_id": chunk.get("parent_chunk_id"),
                    "status": "described",
                    "mtime": mtime,
                    "git_hash": git_hash,
                })
                dedup_same += 1
                continue

            # 2. Cross-path skip: another file with identical content already described.
            donor = code_store.find_described_unit_by_object_checksum(obj_cs, cache=dedup_cache)
            if donor:
                unit_record = {
                    "id": chunk_id,
                    "path": path,
                    "language": chunk.get("language"),
                    "node_type": chunk.get("node_type"),
                    "name": chunk.get("name"),
                    "level": 0,
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                    "object_checksum": obj_cs,
                    "node_checksum": donor.get("node_checksum"),
                    "description": donor.get("description"),
                    "inferred_name": donor.get("inferred_name"),
                    "parent_id": chunk.get("parent_chunk_id"),
                    "status": "described",
                    "mtime": mtime,
                    "git_hash": git_hash,
                }
                code_store.upsert_unit(unit_record)
                # Add to cache so subsequent files in this run see it immediately.
                new_described[obj_cs] = unit_record
                dedup_cross += 1
                continue

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
        pending += 1

    if getattr(code_store, "file_rollups", False):
        _file_rollup(code_store, chunks, path, mtime, git_hash)

    code_store.set_file_record(
        path, file_checksum,
        content_checksum=content_checksum,
        file_size=file_size,
    )

    return {
        "dedup_same": dedup_same,
        "dedup_cross": dedup_cross,
        "pending": pending,
        "new_described": new_described,
    }
