"""Code-index health: one consistency report, one repair, one status line.

Backs `agent index --check [--fix]`, the `index_status` tool and the UI
freshness view, so the model and the user read the same numbers instead of
querying `.agent/index.db` by hand (a session looped on exactly that: the
legacy `file_mtimes` rows contradicted `chunks`).

Cheap by default. `files=True` adds the disk walk that counts files changed
since they were indexed (stat per file, no reads).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.rag.store import VectorStore


def _mtime_orphans(store: "VectorStore", root: Path) -> list[str]:
    """Chunkless rows whose file is gone, ignored, or outside this root."""
    from agent.tools.rules import get_rules
    rules = get_rules()
    out = []
    for p in store.list_mtime_only_paths():
        fp = Path(p) if Path(p).is_absolute() else root / p
        ignored = (not rules.ignore.empty) and rules.ignore.matches(p)
        if ignored or not fp.exists():
            out.append(p)
    return out


def check(store: "VectorStore", root: str, *, embed_model: str = "", files: bool = False,
          cfg=None) -> dict:
    """Defects and freshness numbers. Never writes."""
    root_path = Path(root).resolve()
    st = store.stats()
    cov = store.vector_coverage()
    conn = store._conn()
    dup = conn.execute(
        "SELECT count(*) FROM file_mtimes WHERE path IN (SELECT DISTINCT path FROM chunks)").fetchone()[0]
    stored_root = store.get_meta("index_root") or ""
    last = store.get_meta("last_index_at") or ""
    h: dict = {
        "files": st.get("files", 0),
        "chunks": st.get("chunks", 0),
        "embedding_model": st.get("embedding_model") or "",
        "coverage": cov,
        "fts_drift": store.fts_drift(),
        "mtime_orphans": _mtime_orphans(store, root_path),
        "mtime_duplicates": dup,
        "index_root": stored_root,
        "root_ok": (not stored_root) or Path(stored_root) == root_path,
        "model_ok": (not embed_model) or (not st.get("embedding_model"))
                    or st.get("embedding_model") == embed_model,
        "configured_model": embed_model,
        "last_index_at": float(last) if last else None,
        "last_pass_at": float(store.get_meta("last_pass_at") or 0) or None,
    }
    if files:
        from agent.rag.indexer import pending_files
        p = pending_files(root=str(root_path), store=store, cfg=cfg)
        h["pending"] = p["pending"]
        h["pending_paths"] = p.get("paths", [])[:20]
        h["total_files"] = p["total"]
    h["defects"] = defects(h)
    return h


def defects(h: dict) -> list[str]:
    out = []
    if h["fts_drift"]:
        out.append(f"keyword index out of sync by {h['fts_drift']} rows")
    if h["mtime_orphans"]:
        out.append(f"{len(h['mtime_orphans'])} stale file records (deleted, ignored or other root)")
    if h["mtime_duplicates"]:
        out.append(f"{h['mtime_duplicates']} duplicate file records (shadowed by chunks)")
    cov = h["coverage"]
    if cov.get("missing"):
        out.append(f"{cov['missing']} of {cov['chunks']} chunks have no embedding — invisible to "
                   "semantic search; agent index --update fills them (needs an embedder)")
    if cov.get("orphans"):
        out.append(f"{cov['orphans']} embeddings without a chunk")
    if not h["root_ok"]:
        out.append(f"index built for root {h['index_root']} — paths may not resolve here")
    if not h["model_ok"]:
        out.append(f"index embedded with {h['embedding_model']}, config uses {h['configured_model']} "
                   "(agent index --reembed)")
    return out


def fix(store: "VectorStore", root: str, archive=None) -> dict:
    """Repair what `check` reports, except what needs embedding (missing vectors,
    model mismatch) — that is `agent index --update/--reembed`, which may need
    an embedder. Safe to repeat; a second run changes nothing."""
    from agent.rag.indexer import prune_index
    out = {"fts_rebuilt": 0, "pruned": 0, "duplicates": 0, "orphan_vectors": 0}
    drift = store.fts_drift()
    if drift:
        store.rebuild_fts()
        out["fts_rebuilt"] = drift
    if archive is not None:
        out["pruned"] = len(prune_index(root, store, archive)["paths"])
    else:
        for p in _mtime_orphans(store, Path(root).resolve()):
            store.delete_by_path(p)
            out["pruned"] += 1
    out["duplicates"] = store.dedupe_file_mtimes()
    try:
        out["orphan_vectors"] = store.delete_orphan_vectors()
    except Exception:
        pass
    return out


def _ago(ts: float | None) -> str:
    if not ts:
        return "never"
    s = int(time.time() - ts)
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if s >= n:
            return f"{s // n} {unit} ago"
    return f"{s} s ago"


def summary(h: dict) -> str:
    """Compact text for the model and the terminal."""
    cov = h["coverage"]
    lines = [f"code index: {h['files']} files, {h['chunks']} chunks, "
             f"embeddings {h['embedding_model'] or 'none'}"
             + (f" ({cov['embedded']}/{cov['chunks']} chunks embedded)" if cov.get("embedded") is not None else ""),
             f"last indexed: {_ago(h['last_index_at']) if h['last_index_at'] else ('unknown' if h['chunks'] else 'never')}"
             + (f", last maintenance pass: {_ago(h['last_pass_at'])}" if h.get("last_pass_at") else "")]
    if "pending" in h:
        if h["pending"]:
            lines.append(f"changed since indexed: {h['pending']} of {h['total_files']} files: "
                         + ", ".join(h["pending_paths"][:10])
                         + (" …" if h["pending"] > 10 else ""))
        else:
            lines.append(f"up to date: all {h['total_files']} files match the index")
    lines += [f"defect: {d}" for d in h["defects"]] or ["no consistency defects"]
    return "\n".join(lines)
