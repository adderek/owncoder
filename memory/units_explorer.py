"""Navigable view of described code units — any language, asm and binaries.

The memory browser lists units flat; this module answers the questions a
reader asks next: which files have units, what a unit is made of (children),
what it belongs to (parent chain), what it calls and what calls it, and the
code the description is about.

Two stores share one shape — `units`/`unit_children` in summaries.db (code,
any language) and `asm_units`/`asm_children` in the chunk index (text asm and
binaries imported through REA, which also fill `asm_edges`) — so every
function takes `which` ("unit" | "asm_unit") and works on either.

Read-only: connections are opened `mode=ro`, nothing here writes.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from agent.memory.browse import _preview, _sqlite_ro, _unit_title, _units_db

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_TABLES = {"unit": ("units", "unit_children"),
           "asm_unit": ("asm_units", "asm_children")}
_SOURCE_MAX_LINES = 400
_LINK_MAX = 200


def _tables(which: str) -> tuple[str, str]:
    if which not in _TABLES:
        raise ValueError(f"unknown unit tier '{which}'")
    return _TABLES[which]


def _has_table(conn, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _row(r) -> dict:
    """Compact form used for every list and link."""
    keys = r.keys()
    return {
        "id": r["id"],
        "title": _unit_title(r),
        "path": r["path"],
        "level": r["level"],
        "status": r["status"],
        "lines": [r["start_line"], r["end_line"]],
        "addr": r["addr"] if "addr" in keys else None,
        "kind": (r["node_type"] if "node_type" in keys else None) or "",
        "preview": _preview(r["description"] or ""),
    }


def _open(config: "Config", which: str):
    table, _ = _tables(which)
    conn = _sqlite_ro(_units_db(config, which))
    if conn is None:
        return None
    if not _has_table(conn, table):
        conn.close()
        return None
    return conn


_NO_DB = "no units yet — run 'agent init' (code) or analyze_binary / asm indexing"


def files(config: "Config", which: str, query: str = "", limit: int = 300) -> dict:
    """Paths that carry units, with how many are described."""
    table, _ = _tables(which)
    conn = _open(config, which)
    if conn is None:
        return {"files": [], "total": 0, "note": _NO_DB}
    try:
        where, params = "", []
        if query.strip():
            where, params = "WHERE path LIKE ?", [f"%{query.strip()}%"]
        total = conn.execute(
            f"SELECT COUNT(DISTINCT path) FROM {table} {where}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT path, COUNT(*) AS n,
                       SUM(status = 'described') AS described,
                       MAX(level) AS top_level
                FROM {table} {where}
                GROUP BY path ORDER BY path LIMIT ?""", params + [limit]).fetchall()
    finally:
        conn.close()
    return {"files": [{"path": r["path"], "n": r["n"], "described": r["described"] or 0,
                       "top_level": r["top_level"]} for r in rows],
            "total": total}


def units(config: "Config", which: str, *, path: str = "", parent: str = "",
          query: str = "", level: str = "", status: str = "",
          offset: int = 0, limit: int = 100) -> dict:
    """One page of units.

    `parent` lists that unit's children in order; `path` alone lists the
    file's roots (units without a parent). Without either, everything that
    matches the filters — that is the search mode.
    """
    if parent:
        return _children_page(config, which, parent, offset, limit)
    table, _ = _tables(which)
    conn = _open(config, which)
    if conn is None:
        return {"items": [], "total": 0, "note": _NO_DB}
    clauses, params = [], []
    if path:
        clauses.append("path = ?")
        params.append(path)
        if not query.strip():
            # A parent id that points at nothing (a pruned roll-up) still
            # leaves its children reachable as roots.
            clauses.append(f"(parent_id IS NULL OR parent_id = '' OR parent_id NOT IN"
                           f" (SELECT id FROM {table}))")
    if query.strip():
        like = f"%{query.strip()}%"
        clauses.append("(description LIKE ? OR inferred_name LIKE ? OR path LIKE ?"
                       + (" OR name LIKE ?" if which == "unit" else "") + ")")
        params += [like] * (4 if which == "unit" else 3)
    if level.strip().lstrip("-").isdigit():
        clauses.append("level = ?")
        params.append(int(level))
    if status.strip():
        clauses.append("status = ?")
        params.append(status.strip())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    # Inside one file reading order is position; across files a roll-up says
    # more per row than a leaf, so higher levels come first.
    order = "start_line, level DESC" if path else "level DESC, path, start_line"
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM {table} {where}", params).fetchone()[0]
        rows = conn.execute(f"SELECT * FROM {table} {where} ORDER BY {order} "
                            "LIMIT ? OFFSET ?", params + [limit, offset]).fetchall()
        items = [_row(r) for r in rows]
        _mark_children(conn, which, items)
    finally:
        conn.close()
    return {"items": items, "total": total, "offset": offset}


def _child_ids(conn, which: str, parent: str) -> list[str]:
    table, children = _tables(which)
    ids = []
    if _has_table(conn, children):
        ids = [r[0] for r in conn.execute(
            f"SELECT child_id FROM {children} WHERE parent_id=? ORDER BY child_order",
            (parent,))]
    # parent_id is the other half of the same relation; code units set only
    # that one, asm sets both — union, explicit order first.
    seen = set(ids)
    for r in conn.execute(f"SELECT id FROM {table} WHERE parent_id=? ORDER BY start_line",
                          (parent,)):
        if r[0] not in seen:
            ids.append(r[0])
            seen.add(r[0])
    return ids


def _fetch(conn, table: str, ids: list[str]) -> list:
    if not ids:
        return []
    out = {}
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        q = ",".join("?" * len(part))
        for r in conn.execute(f"SELECT * FROM {table} WHERE id IN ({q})", part):
            out[r["id"]] = r
    return [out[i] for i in ids if i in out]


def _mark_children(conn, which: str, items: list[dict]) -> None:
    """Flag rows that can be expanded, without a query per row."""
    if not items:
        return
    table, children = _tables(which)
    ids = [it["id"] for it in items]
    q = ",".join("?" * len(ids))
    has = {r[0] for r in conn.execute(
        f"SELECT DISTINCT parent_id FROM {table} WHERE parent_id IN ({q})", ids)}
    if _has_table(conn, children):
        has |= {r[0] for r in conn.execute(
            f"SELECT DISTINCT parent_id FROM {children} WHERE parent_id IN ({q})", ids)}
    for it in items:
        it["has_children"] = it["id"] in has


def _children_page(config, which, parent, offset, limit) -> dict:
    table, _ = _tables(which)
    conn = _open(config, which)
    if conn is None:
        return {"items": [], "total": 0, "note": _NO_DB}
    try:
        ids = _child_ids(conn, which, parent)
        items = [_row(r) for r in _fetch(conn, table, ids[offset:offset + limit])]
        _mark_children(conn, which, items)
    finally:
        conn.close()
    return {"items": items, "total": len(ids), "offset": offset}


def _json_list(raw) -> list[str]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return [s.strip() for s in str(raw).split(",") if s.strip()]
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


_ADDR_LINE_RE = re.compile(r"\s*([0-9A-Fa-f]{3,8})h?:", re.I)
_ADDR_REF_RE = re.compile(r"(?:0x)?([0-9A-Fa-f]{2,8})h?", re.I)
_addr_cache: dict[tuple[str, float], dict[int, int]] = {}


def _addr_lines(config: "Config", path: str) -> dict[int, int]:
    """address → line for listings that prefix each line with it (`0C68: INC`)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(config.tools.working_dir) / p
    try:
        key = (str(p), p.stat().st_mtime)
    except OSError:
        return {}
    if key not in _addr_cache:
        out: dict[int, int] = {}
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for n, line in enumerate(f, 1):
                    m = _ADDR_LINE_RE.match(line)
                    if m:
                        out.setdefault(int(m.group(1), 16), n)
        except OSError:
            pass
        _addr_cache.clear()          # one listing at a time is what a reader opens
        _addr_cache[key] = out
    return _addr_cache[key]


def _resolve_calls(config: "Config", conn, which: str, r, names: list[str]) -> list[dict]:
    """Match `calls` labels to units of the same file by name or address.

    Text asm records callees as labels or addresses the LLM read off the
    code; an address resolves through the listing's line prefixes to the
    innermost unit holding that line. Best-effort: an unmatched label stays
    plain text.
    """
    table, _ = _tables(which)
    out = []
    cols = ["inferred_name"] + (["name"] if which == "unit" else ["addr"])
    lines = None
    for name in names[:_LINK_MAX]:
        cond = " OR ".join(f"{c} = ?" for c in cols)
        hit = conn.execute(f"SELECT * FROM {table} WHERE path = ? AND ({cond}) "
                           "ORDER BY level LIMIT 1",
                           [r["path"]] + [name] * len(cols)).fetchone()
        m = _ADDR_REF_RE.fullmatch(name.strip())
        if hit is None and which == "asm_unit" and m:
            if lines is None:
                lines = _addr_lines(config, r["path"])
            line = lines.get(int(m.group(1), 16))
            if line:
                hit = conn.execute(
                    f"SELECT * FROM {table} WHERE path = ? AND start_line <= ? "
                    "AND end_line >= ? ORDER BY level, start_line DESC LIMIT 1",
                    (r["path"], line, line)).fetchone()
                if hit is not None:
                    out.append({"label": name, "line": line, **_row(hit)})
                    continue
        out.append({"label": name, **(_row(hit) if hit else {})})
    return out


def _edges(conn, which: str, r) -> tuple[list[dict] | None, list[dict]]:
    """Callees/callers from asm_edges (REA binaries); None when not recorded."""
    table, _ = _tables(which)
    if which != "asm_unit" or not _has_table(conn, "asm_edges"):
        return None, []
    out_ids = [x[0] for x in conn.execute(
        "SELECT dst FROM asm_edges WHERE src=? ORDER BY dst", (r["id"],))]
    in_ids = [x[0] for x in conn.execute(
        "SELECT src FROM asm_edges WHERE dst=? ORDER BY src", (r["id"],))]
    if not out_ids and not in_ids:
        return None, []
    internal = {x["id"]: x for x in _fetch(conn, table, out_ids + in_ids)}

    def link(i: str) -> dict:
        if i in internal:
            return {"label": internal[i]["inferred_name"] or i, **_row(internal[i])}
        return {"label": i.removeprefix("import:"), "external": True}
    return ([link(i) for i in out_ids[:_LINK_MAX]],
            [link(i) for i in in_ids[:_LINK_MAX]])


def _callers_by_name(config: "Config", conn, which: str, r) -> list[dict]:
    """Units of the same file whose `calls` names this one (text asm).

    A label matches the inferred name; an address matches when the listing
    puts it inside this unit's lines. Only leaves from outside the unit
    count, so a group is not "called" by its own parts.
    """
    table, _ = _tables(which)
    if "calls" not in r.keys():
        return []
    lo, hi, name = r["start_line"], r["end_line"], r["inferred_name"]
    lines = None
    out = []
    for c in conn.execute(f"SELECT * FROM {table} WHERE path = ? AND level = 0 "
                          "AND calls IS NOT NULL AND calls != '' "
                          "AND (end_line < ? OR start_line > ?)",
                          (r["path"], lo, hi)):
        for label in _json_list(c["calls"]):
            hit = bool(name) and label == name
            m = None if hit else _ADDR_REF_RE.fullmatch(label.strip())
            if m:
                if lines is None:
                    lines = _addr_lines(config, r["path"])
                line = lines.get(int(m.group(1), 16))
                hit = line is not None and lo <= line <= hi
            if hit:
                out.append({"label": c["inferred_name"] or c["id"], **_row(c)})
                break
        if len(out) >= _LINK_MAX:
            break
    return out


def _source(config: "Config", r, binary: bool) -> dict:
    if binary:
        return {"note": "binary — no source text; range is an address span"}
    p = Path(r["path"])
    if not p.is_absolute():
        p = Path(config.tools.working_dir) / p
    start, end = r["start_line"] or 1, r["end_line"] or r["start_line"] or 1
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return {"note": f"source not readable: {exc.strerror or exc}"}
    total = end - start + 1
    cut = total > _SOURCE_MAX_LINES
    text = "".join(lines[start - 1:start - 1 + min(total, _SOURCE_MAX_LINES)])
    return {"text": text, "start": start, "end": end, "truncated": cut,
            "stale": end > len(lines)}


def node(config: "Config", which: str, unit_id: str) -> dict:
    """Everything known about one unit, with its neighbours as links."""
    table, _ = _tables(which)
    conn = _open(config, which)
    if conn is None:
        return {"error": _NO_DB}
    try:
        r = conn.execute(f"SELECT * FROM {table} WHERE id=?", (unit_id,)).fetchone()
        if r is None:
            return {"error": "not found"}
        keys = r.keys()
        binary = bool("addr" in keys and r["addr"])
        crumbs, seen, pid = [], {r["id"]}, r["parent_id"]
        while pid and pid not in seen and len(crumbs) < 32:
            seen.add(pid)
            p = conn.execute(f"SELECT * FROM {table} WHERE id=?", (pid,)).fetchone()
            if p is None:
                break
            crumbs.insert(0, _row(p))
            pid = p["parent_id"]
        child_ids = _child_ids(conn, which, r["id"])
        children = [_row(c) for c in _fetch(conn, table, child_ids[:_LINK_MAX])]
        _mark_children(conn, which, children)
        siblings = {k: (_row(s) if s else None) for k, s in (
            (k, conn.execute(f"SELECT * FROM {table} WHERE id=?", (r[k],)).fetchone()
             if r[k] else None) for k in ("prev_id", "next_id"))}
        callees, callers = _edges(conn, which, r)
        if callees is None:
            calls = _json_list(r["calls"]) if "calls" in keys else []
            callees = _resolve_calls(config, conn, which, r, calls)
            callers = _callers_by_name(config, conn, which, r)
        meta = {k: r[k] for k in ("language", "node_type", "name", "confidence",
                                  "analysis_model", "revision", "addr", "git_hash")
                if k in keys and r[k] not in (None, "")}
        extra = {}
        for k in ("side_effects", "key_patterns", "evidence", "extra_json"):
            if k in keys and r[k]:
                try:
                    extra[k] = json.loads(r[k])
                except (TypeError, ValueError):
                    extra[k] = r[k]
        return {
            "unit": {**_row(r), "description": r["description"] or "",
                     "inferred_name": r["inferred_name"] or ""},
            "meta": meta, "extra": extra,
            "breadcrumb": crumbs,
            "children": children, "children_total": len(child_ids),
            "prev": siblings["prev_id"], "next": siblings["next_id"],
            "callees": callees, "callers": callers,
            "source": _source(config, r, binary),
        }
    finally:
        conn.close()


def summary(config: "Config") -> dict:
    """Which unit stores exist and how big — picks the explorer's start tier."""
    out = []
    for which, label in (("unit", "code units"), ("asm_unit", "asm / binary units")):
        table, _ = _tables(which)
        conn = _open(config, which)
        n = d = 0
        if conn is not None:
            try:
                n, d = conn.execute(f"SELECT COUNT(*), SUM(status='described') "
                                    f"FROM {table}").fetchone()
            finally:
                conn.close()
        out.append({"key": which, "label": label, "count": n or 0, "described": d or 0})
    return {"tiers": out}
