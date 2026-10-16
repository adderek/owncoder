"""Reviewer actions on described units: correct, judge, re-describe, search.

The write side of the unit explorer (units_explorer.py reads). Corrections
go through rag/unit_overrides.py so they survive re-analysis; a
re-description uses the session's current model — usually stronger than the
bulk describer — and is told what the reviewer said about the old text.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from agent.memory.browse import _units_db
from agent.memory.units_explorer import _abs, _child_ids, _file_edges, _tables
from agent.rag import unit_overrides

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


class ReviewError(Exception):
    pass


def _open_rw(config: "Config", which: str) -> sqlite3.Connection:
    table, _ = _tables(which)
    db = _units_db(config, which)
    if not db.exists():
        raise ReviewError("no unit database")
    from agent.core.sqlite_util import open_threadlocal_conn
    conn = open_threadlocal_conn(str(db), load_vec=True)
    unit_overrides.ensure_schema(conn, table)
    return conn


def _unit(conn, table: str, unit_id: str) -> dict:
    r = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (unit_id,)).fetchone()
    if r is None:
        raise ReviewError("unit not found")
    return dict(r)


def _embed(embedder, text: str) -> list[float] | None:
    if embedder is None or not text:
        return None
    try:
        return embedder.embed_one(text)
    except Exception:
        logger.warning("units review: embedding failed", exc_info=True)
        return None


def _write_vector(conn, which: str, unit_id: str, emb: list[float] | None) -> None:
    """Point the unit's vector at `emb` — when the table's dims allow."""
    if not emb:
        return
    vec = "vec_units" if which == "unit" else "vec_asm_units"
    meta, key = (("_units_meta", "vec_units_dims") if which == "unit"
                 else ("_asm_meta", "vec_asm_dims"))
    try:
        row = conn.execute(f"SELECT value FROM {meta} WHERE key = ?", (key,)).fetchone()
        if row is None or int(row[0]) != len(emb):
            return
        import sqlite_vec
        conn.execute(f"DELETE FROM {vec} WHERE unit_id = ?", (unit_id,))
        conn.execute(f"INSERT INTO {vec}(unit_id, embedding) VALUES (?, ?)",
                     (unit_id, sqlite_vec.serialize_float32(emb)))
    except Exception:
        logger.warning("units review: vector update failed", exc_info=True)


def edit(config: "Config", which: str, unit_id: str, *, name: str | None,
         description: str | None, embedder=None) -> dict:
    """Reviewer's name/description; empty strings return to the model's text."""
    table, _ = _tables(which)
    conn = _open_rw(config, which)
    try:
        unit = _unit(conn, table, unit_id)
        emb = _embed(embedder, (description or "").strip())
        blob = None
        if emb:
            import sqlite_vec
            blob = sqlite_vec.serialize_float32(emb)
        unit_overrides.set_text(conn, table, unit, name=name, description=description,
                                embedding=blob)
        after = _unit(conn, table, unit_id)
        if not (description or "").strip():
            # Back to the model's words: its vector, if we can make one.
            _write_vector(conn, which, unit_id, _embed(embedder, after["description"] or ""))
        else:
            _write_vector(conn, which, unit_id, emb)
        _mark_parents_pending(conn, table, unit)
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def _mark_parents_pending(conn, table: str, unit: dict) -> None:
    """A roll-up was written from its parts' text; a corrected part makes it
    due for re-summarising on the next analysis run."""
    pid, seen = unit.get("parent_id"), set()
    while pid and pid not in seen:
        seen.add(pid)
        conn.execute(f"UPDATE {table} SET status = 'pending' WHERE id = ? "
                     "AND status IN ('described', 'grouped')", (pid,))
        r = conn.execute(f"SELECT parent_id FROM {table} WHERE id = ?", (pid,)).fetchone()
        pid = r[0] if r else None


def review(config: "Config", which: str, unit_id: str, verdict: str | None,
           note: str = "") -> dict:
    if verdict not in (None, "", "ok", "wrong"):
        raise ReviewError("verdict must be ok, wrong or empty")
    table, _ = _tables(which)
    conn = _open_rw(config, which)
    try:
        unit_overrides.set_review(conn, _unit(conn, table, unit_id), verdict or None, note)
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def _hint(unit: dict, ov: dict | None) -> str:
    old = unit.get("description_model") or unit.get("description") or ""
    if not ov or ov.get("review") != "wrong":
        return ""
    note = ov.get("review_note") or "(no note)"
    return (f'A reviewer rejected an earlier description of this code: "{old}". '
            f"Reviewer's note: {note}. Write a corrected description.")


def redescribe(config: "Config", which: str, unit_id: str, *, embedder=None,
               fetch_pseudo=None) -> dict:
    """Describe the unit again with the session's model; returns new fields.

    fetch_pseudo(path, addr) -> str supplies pseudocode for a binary unit
    stored before pseudocode was kept.
    """
    from openai import OpenAI
    table, _ = _tables(which)
    conn = _open_rw(config, which)
    try:
        unit = _unit(conn, table, unit_id)
        ov = unit_overrides.get(conn, unit_id)
        hint = _hint(unit, ov)
        llm = config.llm
        client = OpenAI(base_url=llm.base_url, api_key=llm.api_key)
        if which == "asm_unit":
            fields = _redescribe_asm(config, conn, client, unit, hint, fetch_pseudo)
        else:
            fields = _redescribe_code(config, conn, client, unit, hint)
        fields["status"] = "grouped" if unit["level"] else "described"
        updated = {**unit, **fields}
        # Through the store's own write, so an override still wins.
        _store_write(config, which, updated, _embed(embedder, fields.get("description", "")))
        # The verdict was about the old text.
        if ov and ov.get("review"):
            unit_overrides.set_review(conn, unit, None, None)
            conn.commit()
        return {"ok": True, "model": llm.model, **{k: fields.get(k) for k in
                ("description", "inferred_name", "confidence")}}
    finally:
        conn.close()


def _store_write(config: "Config", which: str, unit: dict, emb) -> None:
    unit = {k: v for k, v in unit.items() if k not in ("description_model", "name_model")}
    if emb:
        unit["embedding"] = emb
    if which == "asm_unit":
        from agent.rag.asm_store import AsmStore
        store = AsmStore(dataclasses.replace(config.rag, db_path=str(_units_db(config, which))))
    else:
        from agent.rag.code_store import CodeStore
        store = CodeStore(str(_units_db(config, which)))
    try:
        store.upsert_unit(unit)
    finally:
        store.close()


def _children(conn, which: str, table: str, unit: dict) -> list[dict]:
    ids = _child_ids(conn, which, unit["id"])
    rows = {r["id"]: dict(r) for r in conn.execute(
        f"SELECT * FROM {table} WHERE id IN ({','.join('?' * len(ids))})", ids)} if ids else {}
    return [rows[i] for i in ids if i in rows]


def _redescribe_asm(config, conn, client, unit, hint, fetch_pseudo) -> dict:
    from agent.rag.asm_describer import AsmDescriber
    cfg = dataclasses.replace(config.asm, describer_model=config.llm.model)
    d = AsmDescriber(client, cfg, config.llm, token_limits=config.token_limits)
    if unit["level"]:
        kids = _children(conn, "asm_unit", "asm_units", unit)
        if not kids:
            raise ReviewError("roll-up has no parts to summarise")
        return _clean(d.summarize_group(kids, hint=hint))
    edges, _src = _file_edges(config, conn, "asm_unit", unit["path"])
    names = {r["id"]: (r["inferred_name"] or "", r["description"] or "") for r in conn.execute(
        "SELECT id, inferred_name, description FROM asm_units WHERE path = ? AND level = 0",
        (unit["path"],))}
    callees = []
    for src, dst, _k in edges:
        if src == unit["id"]:
            callees.append(names.get(dst) or (dst.split(":", 1)[-1], "outside this file"))
    if unit.get("addr"):
        ev = json.loads(unit.get("evidence") or "{}")
        pseudo = ev.get("pseudocode") or (fetch_pseudo(unit["path"], unit["addr"])
                                          if fetch_pseudo else "")
        if not pseudo:
            raise ReviewError("no pseudocode stored and REA is not reachable")
        out = d.describe_function(
            {"addr": unit["addr"], "name": ev.get("ghidra_name") or unit["inferred_name"],
             "signature": ev.get("signature"), "pseudocode": pseudo},
            callees, [], hint=hint)
        if out.get("failed"):
            raise ReviewError("the model did not return a usable description")
        out.pop("unverified_apis", None)
        return _clean(out)
    lines = _lines(config, unit["path"])
    chunk = {**unit, "content": "".join(lines[unit["start_line"] - 1:unit["end_line"]])}
    return _clean(d.describe_chunk(chunk, None, None, callees=callees, hint=hint))


def _redescribe_code(config, conn, client, unit, hint) -> dict:
    from agent.rag.describer import Describer
    s = config.summarization
    d = Describer(client, config.llm.model, ctx_tokens=s.ctx_tokens,
                  max_output_tokens=max(s.max_output_tokens, 512))
    if unit["level"]:
        kids = _children(conn, "unit", "units", unit)
        if not kids:
            raise ReviewError("roll-up has no parts to summarise")
        out = d.summarize_group(kids, unit.get("language") or "code",
                                unit.get("node_type") or "chunk", hint=hint)
    else:
        lines = _lines(config, unit["path"])
        chunk = {**unit, "content": "".join(lines[unit["start_line"] - 1:unit["end_line"]])}
        out = d.describe_chunk(chunk, hint=hint)
    import time
    out["analysis_model"] = config.llm.model
    out["analysis_date"] = time.time()
    return _clean(out)


def _lines(config, path: str) -> list[str]:
    try:
        return _abs(config, path).read_text(encoding="utf-8", errors="replace").splitlines(True)
    except OSError as exc:
        raise ReviewError(f"source not readable: {exc.strerror or exc}") from None


_TEXT_FIELDS = ("description", "inferred_name", "side_effects", "key_patterns", "confidence")


def _clean(fields: dict) -> dict:
    """Models return lists where text was asked; the columns hold text."""
    from agent.rag.asm_describer import _text
    return {k: (_text(v) if k in _TEXT_FIELDS else v) for k, v in fields.items()}


def semantic(config: "Config", which: str, embedding: list[float], limit: int = 50) -> dict:
    """Units nearest to the query vector, best first."""
    from agent.memory.units_explorer import _row
    table, _ = _tables(which)
    vec = "vec_units" if which == "unit" else "vec_asm_units"
    db = _units_db(config, which)
    if not db.exists():
        return {"items": [], "total": 0, "note": "no unit database"}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (vec,)).fetchone():
            return {"items": [], "total": 0, "note": "no vectors in this store — "
                    "semantic search needs units indexed with embeddings on"}
        rows = conn.execute(f"""
            SELECT u.*, v.distance FROM {vec} v JOIN {table} u ON u.id = v.unit_id
            WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance""",
            (sqlite_vec.serialize_float32(embedding), limit)).fetchall()
    except sqlite3.OperationalError as exc:
        # Typically a dimension mismatch: the embedder changed since indexing.
        return {"items": [], "total": 0, "note": f"vector search failed: {exc}"}
    finally:
        conn.close()
    items = []
    for r in rows:
        it = _row(r)
        it["score"] = round(1.0 - r["distance"] / 2.0, 3)
        items.append(it)
    return {"items": items, "total": len(items), "offset": 0}
