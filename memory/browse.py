"""Read-only browsing of the memory tiers, one registry entry per tier.

`overview.py` answers "how much is stored"; this answers "show me". Both
feed the same UI: the overview is the landing page of the memory view, and
each tier here is one section of it.

A tier is a (list, get) pair registered in `_TIERS`, so a new tier — code
units, chunks, archive, rules, skills — is an entry and a function, not a
change to the UI or the endpoints.

Read-only by design: deleting memory is a decision with consequences and
belongs behind an explicit action, not a browse call.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.config.models import Config

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 220


def _memory_db(config: "Config") -> Path:
    return Path(config.tools.working_dir) / config.tools.agent_dir / "memory.db"


def _store(config: "Config"):
    from agent.memory.store import MemoryStore
    return MemoryStore(_memory_db(config))


def _tags(raw) -> list[str]:
    try:
        return json.loads(raw or "[]")
    except Exception:
        return []


def _preview(body: str) -> str:
    body = " ".join((body or "").split())
    return body[:_PREVIEW_CHARS] + ("…" if len(body) > _PREVIEW_CHARS else "")


def _entry_row(e: dict) -> dict:
    return {
        "id": e.get("id"),
        "title": e.get("title") or "(untitled)",
        "tags": _tags(e.get("tags")),
        "preview": _preview(e.get("body") or ""),
        "updated_at": e.get("updated_at") or e.get("created_at"),
        "source": e.get("source") or "",
    }


# ── memory.db scopes ────────────────────────────────────────────────────────
def _scope_list(scope: str):
    def _list(config: "Config", query: str = "", limit: int = 50) -> dict:
        store = _store(config)
        if query.strip():
            rows = store.fts_search(query, scope=scope, top_k=limit)
        else:
            rows = store.list_entries(scope=scope, limit=limit,
                                      order_by="updated_at DESC")
        return {"items": [_entry_row(r) for r in rows]}
    return _list


def _scope_get(scope: str):
    def _get(config: "Config", item_id: str) -> dict:
        e = _store(config).get(item_id)
        if not e or e.get("scope") != scope:
            return {"error": "not found"}
        return {"title": e.get("title") or "(untitled)",
                "body": e.get("body") or "",
                "tags": _tags(e.get("tags")),
                "meta": {"scope": e.get("scope"), "source": e.get("source") or "",
                         "created_at": e.get("created_at"),
                         "updated_at": e.get("updated_at"),
                         "hits": e.get("hit_count")}}
    return _get


# ── LLM-described units ─────────────────────────────────────────────────────
# Code and assembly both get split into units and described by the model, one
# level at a time; the two live in different DBs with near-identical columns,
# so one pair of readers serves both.
def _sqlite_ro(path: Path):
    """Open read-only. Browsing must never create the DB it is looking for —
    an empty file in place of a missing one hides the real answer."""
    import sqlite3
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _units_db(config: "Config", which: str) -> Path:
    from agent.memory.overview import rag_db_path
    if which == "asm_unit":
        return rag_db_path(config)         # asm units live in the chunk index
    db = Path(config.summarization.db_path)
    return db if db.is_absolute() else Path(config.tools.working_dir) / db


def _unit_title(r) -> str:
    name = (r["inferred_name"] if "inferred_name" in r.keys() else None) or ""
    if not name and "name" in r.keys():
        name = r["name"] or ""
    where = f"{r['path']}:{r['start_line']}"
    return f"{name} — {where}" if name else where


def _unit_list(which: str, table: str):
    def _list(config: "Config", query: str = "", limit: int = 50) -> dict:
        conn = _sqlite_ro(_units_db(config, which))
        if conn is None:
            return {"items": [], "note": "no unit database yet — run 'agent init'"}
        try:
            sql = f"SELECT * FROM {table}"
            params: list = []
            if query.strip():
                like = f"%{query.strip()}%"
                sql += " WHERE description LIKE ? OR path LIKE ? OR inferred_name LIKE ?"
                params += [like, like, like]
            # Highest level first: the roll-ups say more per row than leaves.
            sql += " ORDER BY level DESC, path, start_line LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
        except Exception as exc:
            return {"items": [], "note": str(exc)}
        finally:
            conn.close()
        return {"items": [{
            "id": r["id"],
            "title": _unit_title(r),
            "tags": [f"L{r['level']}", r["status"]],
            "preview": _preview(r["description"] or ""),
            "updated_at": r["mtime"],
            "source": r["path"],
        } for r in rows]}
    return _list


def _unit_get(which: str, table: str):
    def _get(config: "Config", item_id: str) -> dict:
        conn = _sqlite_ro(_units_db(config, which))
        if conn is None:
            return {"error": "no unit database yet"}
        try:
            r = conn.execute(f"SELECT * FROM {table} WHERE id=?", (item_id,)).fetchone()
            if r is None:
                return {"error": "not found"}
            keys = r.keys()
            meta = {"path": r["path"], "lines": f"{r['start_line']}–{r['end_line']}",
                    "level": r["level"], "status": r["status"]}
            for extra in ("language", "node_type", "confidence", "analysis_model",
                          "revision"):
                if extra in keys and r[extra]:
                    meta[extra] = r[extra]
            body = r["description"] or "(not described yet)"
            for extra in ("calls", "side_effects", "key_patterns"):
                if extra in keys and r[extra]:
                    body += f"\n\n{extra}: {r[extra]}"
            return {"title": _unit_title(r), "body": body,
                    "tags": [f"L{r['level']}", r["status"]], "meta": meta}
        finally:
            conn.close()
    return _get


def _unit_count(config: "Config", which: str, table: str) -> int | None:
    conn = _sqlite_ro(_units_db(config, which))
    if conn is None:
        return 0
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


# ── RAG chunks ──────────────────────────────────────────────────────────────
def _chunk_title(r) -> str:
    name = r["name"] or ""
    kind = r["node_type"] or ""
    where = f"{r['path']}:{r['start_line']}–{r['end_line']}"
    head = " ".join(x for x in (kind, name) if x)
    return f"{head} — {where}" if head else where


def _chunk_list(config: "Config", query: str = "", limit: int = 50) -> dict:
    from agent.memory.overview import rag_db_path
    conn = _sqlite_ro(rag_db_path(config))
    if conn is None:
        return {"items": [], "note": "no code index yet — run 'agent init'"}
    try:
        if query.strip():
            # The index already carries FTS5 over content/name/path; a LIKE scan
            # over 18k chunks would be the slow way to ask the same question.
            try:
                rows = conn.execute(
                    """SELECT c.* FROM chunks_fts
                       JOIN chunks c ON c.rowid = chunks_fts.rowid
                       WHERE chunks_fts MATCH ?
                       ORDER BY bm25(chunks_fts) LIMIT ?""",
                    (query, limit)).fetchall()
            except Exception:
                # A bare-word query can be invalid FTS syntax; fall back to a
                # literal phrase rather than showing the user a parser error.
                rows = conn.execute(
                    """SELECT c.* FROM chunks_fts
                       JOIN chunks c ON c.rowid = chunks_fts.rowid
                       WHERE chunks_fts MATCH ?
                       ORDER BY bm25(chunks_fts) LIMIT ?""",
                    ('"' + query.replace('"', "") + '"', limit)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY path, start_line LIMIT ?",
                (limit,)).fetchall()
    except Exception as exc:
        return {"items": [], "note": str(exc)}
    finally:
        conn.close()
    return {"items": [{
        "id": r["id"],
        "title": _chunk_title(r),
        "tags": [t for t in (r["language"],) if t],
        "preview": _preview(r["content"] or ""),
        "updated_at": r["mtime"],
        "source": r["path"],
    } for r in rows]}


def _chunk_get(config: "Config", item_id: str) -> dict:
    from agent.memory.overview import rag_db_path
    conn = _sqlite_ro(rag_db_path(config))
    if conn is None:
        return {"error": "no code index yet"}
    try:
        r = conn.execute("SELECT * FROM chunks WHERE id=?", (item_id,)).fetchone()
        if r is None:
            return {"error": "not found"}
        # vec_chunks is a vec0 virtual table and this connection has no
        # extensions loaded, so ask its shadow table instead: `id` there is the
        # chunk id the vector was stored under.
        try:
            embedded = bool(conn.execute(
                "SELECT 1 FROM vec_chunks_rowids WHERE id=? LIMIT 1",
                (item_id,)).fetchone())
        except Exception:
            embedded = False
        meta = {"path": r["path"], "lines": f"{r['start_line']}–{r['end_line']}",
                "language": r["language"] or "", "node_type": r["node_type"] or "",
                "git_hash": (r["git_hash"] or "")[:12],
                "embedded": "yes" if embedded else "no"}
        return {"title": _chunk_title(r), "body": r["content"] or "",
                "tags": [t for t in (r["language"],) if t], "meta": meta}
    finally:
        conn.close()


def _chunk_count(config: "Config") -> int:
    from agent.memory.overview import rag_db_path
    conn = _sqlite_ro(rag_db_path(config))
    if conn is None:
        return 0
    try:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


# ── knowledge base ──────────────────────────────────────────────────────────
def _kb_corpus(config: "Config"):
    path = getattr(config.kb, "corpus_path", "")
    if not getattr(config.kb, "enabled", False) or not path:
        return None
    from kb.api import Corpus
    return Corpus.open(path)


def _kb_name(node) -> str:
    return (node.inferred_name_override or node.inferred_name_base
            or node.id)


def _kb_description(node) -> str:
    return node.description_override or node.description_base or ""


def _kb_list(config: "Config", query: str = "", limit: int = 50) -> dict:
    corpus = _kb_corpus(config)
    if corpus is None:
        return {"items": [], "note": "kb.enabled is false — no corpus configured"}
    try:
        if query.strip():
            nodes = corpus.search(query, limit=limit)
        else:
            # No query: the corpus has no "recent" order, so show the nodes a
            # reader wants first — described ones, highest priority.
            rows = corpus.conn.execute(
                "SELECT id FROM nodes ORDER BY priority DESC, id LIMIT ?",
                (limit,)).fetchall()
            nodes = [n for n in (corpus.get(r["id"]) for r in rows) if n]
        return {"items": [{
            "id": n.id,
            "title": _kb_name(n),
            "tags": [v for v in (n.dims or {}).values() if v],
            "preview": _preview(_kb_description(n)),
            "updated_at": n.analysis_date,
            "source": n.dims.get("scope", "") if n.dims else "",
        } for n in nodes]}
    finally:
        corpus.close()


def _kb_get(config: "Config", item_id: str) -> dict:
    corpus = _kb_corpus(config)
    if corpus is None:
        return {"error": "kb.enabled is false"}
    try:
        node = corpus.get(item_id)
        if node is None:
            return {"error": "not found"}
        locs = [f"{l.path}:{l.start_line}" if getattr(l, "start_line", None)
                else getattr(l, "path", str(l)) for l in (node.locators or [])]
        body = _kb_description(node) or "(no description)"
        if locs:
            body += "\n\nLocations:\n" + "\n".join("  " + s for s in locs)
        return {"title": _kb_name(node), "body": body,
                "tags": [v for v in (node.dims or {}).values() if v],
                "meta": {"id": node.id, "completeness": node.completeness,
                         "data_grade": node.data_grade,
                         "stale": node.stale,
                         "analysis_model": node.analysis_model or "",
                         "children_described":
                             f"{node.children_described_n}/{node.children_described_m}"}}
    finally:
        corpus.close()


# ── registry ────────────────────────────────────────────────────────────────
# label, list fn, get fn. Order is the order the UI shows them in.
_TIERS: dict[str, tuple[str, Callable, Callable]] = {
    "note": ("notes", _scope_list("note"), _scope_get("note")),
    "session_summary": ("session summaries", _scope_list("session_summary"),
                        _scope_get("session_summary")),
    "facts_round": ("facts rounds", _scope_list("facts_round"),
                    _scope_get("facts_round")),
    "behavioral_rule": ("behavioral rules", _scope_list("behavioral_rule"),
                        _scope_get("behavioral_rule")),
    "context_chunk": ("context chunks", _scope_list("context_chunk"),
                      _scope_get("context_chunk")),
    "unit": ("described units", _unit_list("unit", "units"),
             _unit_get("unit", "units")),
    "asm_unit": ("described asm units", _unit_list("asm_unit", "asm_units"),
                 _unit_get("asm_unit", "asm_units")),
    "chunk": ("code index chunks", _chunk_list, _chunk_get),
    "kb": ("kb / wiki", _kb_list, _kb_get),
}

# Tiers whose counts do not come from memory.db scopes.
_COUNTERS: dict[str, "Callable[[Config], int | None]"] = {
    "unit": lambda c: _unit_count(c, "unit", "units"),
    "asm_unit": lambda c: _unit_count(c, "asm_unit", "asm_units"),
    "chunk": _chunk_count,
}


def tiers(config: "Config") -> list[dict]:
    """Browsable tiers with their counts — the memory view's left rail.

    A tier with nothing in it is still listed: "0 notes" is an answer, and a
    rail that changes shape as data arrives is harder to navigate than one
    that does not.
    """
    counts: dict[str, int] = {}
    try:
        counts = _store(config).counts_by_scope()
    except Exception:
        logger.debug("browse: scope counts failed", exc_info=True)
    kb_count = None
    try:
        corpus = _kb_corpus(config)
        if corpus is not None:
            try:
                kb_count = corpus.conn.execute(
                    "SELECT COUNT(*) FROM nodes").fetchone()[0]
            finally:
                corpus.close()
    except Exception:
        logger.debug("browse: kb count failed", exc_info=True)

    out = []
    for key, (label, _l, _g) in _TIERS.items():
        if key == "kb":
            count = kb_count
        elif key in _COUNTERS:
            try:
                count = _COUNTERS[key](config)
            except Exception:
                logger.debug("browse: count for %s failed", key, exc_info=True)
                count = 0
        else:
            count = counts.get(key, 0)
        out.append({"key": key, "label": label, "count": count,
                    "available": count is not None})
    return out


def browse(config: "Config", tier: str, query: str = "", limit: int = 50) -> dict:
    entry = _TIERS.get(tier)
    if entry is None:
        return {"error": f"unknown tier '{tier}'", "items": []}
    try:
        out = entry[1](config, query, limit)
    except Exception as exc:
        logger.debug("browse: tier %s failed", tier, exc_info=True)
        return {"error": str(exc), "items": []}
    out.setdefault("items", [])
    out["tier"] = tier
    out["label"] = entry[0]
    return out


def item(config: "Config", tier: str, item_id: str) -> dict:
    entry = _TIERS.get(tier)
    if entry is None:
        return {"error": f"unknown tier '{tier}'"}
    try:
        return entry[2](config, item_id)
    except Exception as exc:
        logger.debug("browse: item %s/%s failed", tier, item_id, exc_info=True)
        return {"error": str(exc)}
