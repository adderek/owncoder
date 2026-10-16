"""Human corrections to unit descriptions — shared by `units` and `asm_units`.

A reviewer's name/description wins over the model's and must survive the
next analysis run. The code indexer drops and re-inserts a file's units on
every change, so corrections live in their own table, `unit_overrides`, in
the same database, and are laid over each unit as it is written:
`description`/`inferred_name` stay the *effective* text everything else
reads (RAG, callers' context, asm_tree), the model's own output is kept in
`description_model`/`name_model` while an override is set, and the stored
embedding is the reviewer's text's.

A unit whose id changed (lines moved) is re-matched by path + content
checksum; one whose content changed keeps its override, flagged stale
(`checksum` here ≠ the unit's) so the reader knows to check it.
"""
from __future__ import annotations

import sqlite3
import time

DDL = """
CREATE TABLE IF NOT EXISTS unit_overrides (
    unit_id           TEXT PRIMARY KEY,
    path              TEXT NOT NULL,
    checksum          TEXT,
    human_name        TEXT,
    human_description TEXT,
    human_at          REAL,
    review            TEXT,
    review_note       TEXT,
    review_at         REAL,
    embedding         BLOB
);
CREATE INDEX IF NOT EXISTS unit_overrides_path_cs ON unit_overrides(path, checksum);
"""

MODEL_COLS = ("description_model", "name_model")


def ensure_schema(conn: sqlite3.Connection, table: str) -> None:
    conn.executescript(DDL)
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    for col in MODEL_COLS:
        if col not in have:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def checksum_of(unit: dict) -> str | None:
    """Content checksum: asm units carry `checksum`, code units `object_checksum`."""
    return unit.get("checksum") or unit.get("object_checksum")


def lay_over(conn: sqlite3.Connection, table: str, unit: dict) -> dict:
    """Effective text for `unit` about to be written; `unit` is not changed.

    Returns {description, inferred_name, description_model, name_model,
    embedding}; embedding is the reviewer text's blob (written instead of
    the model-text vector) or None.

    The model's text is whatever `unit` carries unless that *is* the
    reviewer's text — a dict read back from the table and re-written by a
    pipeline step — in which case the stored model copy is used.
    """
    uid = unit["id"]
    row = conn.execute("SELECT * FROM unit_overrides WHERE unit_id = ?", (uid,)).fetchone()
    cs = checksum_of(unit)
    if row is None and cs:
        # Same code under a new id: an override whose unit is gone moves here.
        row = conn.execute(
            f"SELECT * FROM unit_overrides WHERE path = ? AND checksum = ? "
            f"AND unit_id NOT IN (SELECT id FROM {table}) LIMIT 1",
            (unit["path"], cs)).fetchone()
        if row is not None:
            conn.execute("UPDATE unit_overrides SET unit_id = ? WHERE unit_id = ?",
                         (uid, row["unit_id"]))
    out = {"description": unit.get("description"), "inferred_name": unit.get("inferred_name"),
           "description_model": None, "name_model": None, "embedding": None}
    if row is None:
        return out
    for field, model_col, human_col in (("description", "description_model", "human_description"),
                                        ("inferred_name", "name_model", "human_name")):
        human = row[human_col]
        if human:
            mine = unit.get(field)
            out[model_col] = (unit.get(model_col) or mine) if mine == human else mine
            out[field] = human
    if row["human_description"]:
        out["embedding"] = row["embedding"]
    return out


def vector_blob(model_emb, human_blob: bytes | None,
                current_dims: int | None) -> tuple[bytes | None, int]:
    """(blob, dims) to store: the reviewer text's vector when there is one
    and it fits the table, else the model text's."""
    if human_blob and (current_dims is None or len(human_blob) // 4 == current_dims):
        return human_blob, len(human_blob) // 4
    if model_emb:
        import sqlite_vec
        return sqlite_vec.serialize_float32(model_emb), len(model_emb)
    return None, 0


def _row(conn, unit_id: str):
    return conn.execute("SELECT * FROM unit_overrides WHERE unit_id = ?", (unit_id,)).fetchone()


def _ensure_row(conn, unit: dict) -> None:
    if _row(conn, unit["id"]) is None:
        conn.execute("INSERT INTO unit_overrides(unit_id, path, checksum) VALUES (?, ?, ?)",
                     (unit["id"], unit["path"], checksum_of(unit)))


def set_text(conn: sqlite3.Connection, table: str, unit: dict, *,
             name: str | None, description: str | None,
             embedding: bytes | None = None) -> None:
    """Write the reviewer's name/description (None or "" clears that field)."""
    _ensure_row(conn, unit)
    name = (name or "").strip() or None
    description = (description or "").strip() or None
    conn.execute("""UPDATE unit_overrides SET human_name = ?, human_description = ?,
                    human_at = ?, checksum = ?, embedding = ? WHERE unit_id = ?""",
                 (name, description, time.time(), checksum_of(unit),
                  embedding if description else None, unit["id"]))
    model_desc = unit.get("description_model") or unit.get("description")
    model_name = unit.get("name_model") or unit.get("inferred_name")
    conn.execute(f"""UPDATE {table} SET
                     description = ?, description_model = ?,
                     inferred_name = ?, name_model = ? WHERE id = ?""",
                 (description or model_desc, model_desc if description else None,
                  name or model_name, model_name if name else None, unit["id"]))
    _prune(conn, unit["id"])


def set_review(conn: sqlite3.Connection, unit: dict, review: str | None,
               note: str | None) -> None:
    _ensure_row(conn, unit)
    conn.execute("UPDATE unit_overrides SET review = ?, review_note = ?, review_at = ? "
                 "WHERE unit_id = ?",
                 (review or None, (note or "").strip() or None,
                  time.time() if review else None, unit["id"]))
    _prune(conn, unit["id"])


def _prune(conn, unit_id: str) -> None:
    conn.execute("""DELETE FROM unit_overrides WHERE unit_id = ? AND human_name IS NULL
                    AND human_description IS NULL AND review IS NULL""", (unit_id,))


def get(conn: sqlite3.Connection, unit_id: str) -> dict | None:
    try:
        r = _row(conn, unit_id)
    except sqlite3.OperationalError:          # older DB, no table yet
        return None
    if r is None:
        return None
    d = dict(r)
    d.pop("embedding", None)
    return d
