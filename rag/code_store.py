"""Generalized unit store for hierarchical code summarization.

Schema mirrors kb/src/kb/schema.sql but inlined here for zero cross-repo
dependency. One SQLite DB per project (.agent/summaries.db by default).
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path


def _checksum(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


_DDL = """
CREATE TABLE IF NOT EXISTS units (
    id                TEXT PRIMARY KEY,
    path              TEXT NOT NULL,
    language          TEXT,
    node_type         TEXT,
    name              TEXT,
    level             INTEGER NOT NULL DEFAULT 0,
    start_line        INTEGER,
    end_line          INTEGER,
    description       TEXT,
    inferred_name     TEXT,
    object_checksum   TEXT,
    node_checksum     TEXT,
    edge_set_checksum TEXT,
    revision          INTEGER NOT NULL DEFAULT 1,
    parent_id         TEXT,
    prev_id           TEXT,
    next_id           TEXT,
    status            TEXT NOT NULL DEFAULT 'pending',
    confidence        TEXT,
    mtime             REAL,
    git_hash          TEXT,
    analysis_date     REAL,
    analysis_model    TEXT,
    extra_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_units_path_level ON units(path, level, start_line);
CREATE INDEX IF NOT EXISTS idx_units_status     ON units(status);
CREATE INDEX IF NOT EXISTS idx_units_parent     ON units(parent_id);

CREATE TABLE IF NOT EXISTS unit_children (
    parent_id   TEXT NOT NULL,
    child_id    TEXT NOT NULL,
    child_order INTEGER NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_children_parent ON unit_children(parent_id, child_order);

CREATE TABLE IF NOT EXISTS indexed_files (
    path        TEXT PRIMARY KEY,
    checksum    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'indexed',
    indexed_at  REAL
);

CREATE TABLE IF NOT EXISTS judge_cache (
    key        TEXT PRIMARY KEY,
    changed    INTEGER NOT NULL,
    reason     TEXT,
    cached_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS _units_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class CodeStore:
    def __init__(self, db_path: str) -> None:
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(p.resolve())
        self._local = threading.local()
        self._setup()
        self._vec_dims: int | None = self._read_vec_dims()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            self._run_ddl(conn)
        return self._local.conn

    def _run_ddl(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL)
        conn.commit()

    def _setup(self) -> None:
        self._run_ddl(self._conn)

    def _read_vec_dims(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM _units_meta WHERE key = 'vec_units_dims'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _ensure_vec_table(self, dims: int) -> None:
        conn = self._conn
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception as e:
            conn.enable_load_extension(False)
            raise RuntimeError(f"sqlite-vec unavailable: {e}") from e
        conn.enable_load_extension(False)
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_units USING vec0(
                unit_id   TEXT PRIMARY KEY,
                embedding float[{dims}] distance_metric=cosine
            )
        """)
        if self._vec_dims != dims:
            conn.execute(
                "INSERT OR REPLACE INTO _units_meta(key, value) VALUES ('vec_units_dims', ?)",
                (str(dims),),
            )
            self._vec_dims = dims
        conn.commit()

    # ── write ────────────────────────────────────────────────────────────────

    def upsert_unit(self, unit: dict) -> None:
        conn = self._conn
        conn.execute("""
            INSERT OR REPLACE INTO units
            (id, path, language, node_type, name, level,
             start_line, end_line, description, inferred_name,
             object_checksum, node_checksum, edge_set_checksum,
             revision, parent_id, prev_id, next_id,
             status, confidence, mtime, git_hash,
             analysis_date, analysis_model, extra_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            unit["id"], unit["path"],
            unit.get("language"), unit.get("node_type"), unit.get("name"),
            unit.get("level", 0),
            unit.get("start_line"), unit.get("end_line"),
            unit.get("description"), unit.get("inferred_name"),
            unit.get("object_checksum"), unit.get("node_checksum"),
            unit.get("edge_set_checksum"),
            unit.get("revision", 1),
            unit.get("parent_id"), unit.get("prev_id"), unit.get("next_id"),
            unit.get("status", "pending"),
            unit.get("confidence"),
            unit.get("mtime"), unit.get("git_hash"),
            unit.get("analysis_date"), unit.get("analysis_model"),
            unit.get("extra_json"),
        ))
        emb = unit.get("embedding")
        if emb:
            import sqlite_vec
            dims = len(emb)
            self._ensure_vec_table(dims)
            conn.execute("DELETE FROM vec_units WHERE unit_id = ?", (unit["id"],))
            conn.execute(
                "INSERT INTO vec_units(unit_id, embedding) VALUES (?, ?)",
                (unit["id"], sqlite_vec.serialize_float32(emb)),
            )
        conn.commit()

    def upsert_children(self, parent_id: str, child_ids: list[str]) -> None:
        conn = self._conn
        conn.execute("DELETE FROM unit_children WHERE parent_id = ?", (parent_id,))
        conn.executemany(
            "INSERT INTO unit_children(parent_id, child_id, child_order) VALUES (?,?,?)",
            [(parent_id, cid, i) for i, cid in enumerate(child_ids)],
        )
        conn.commit()

    def set_file_record(self, path: str, checksum: str, status: str = "indexed") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO indexed_files(path, checksum, status, indexed_at) VALUES (?,?,?,?)",
            (path, checksum, status, time.time()),
        )
        self._conn.commit()

    def delete_units_for_file(self, path: str) -> None:
        ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM units WHERE path = ?", (path,)
        ).fetchall()]
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        conn = self._conn
        conn.execute(
            f"DELETE FROM unit_children WHERE parent_id IN ({ph}) OR child_id IN ({ph})",
            ids + ids,
        )
        if self._vec_dims is not None:
            try:
                conn.execute(f"DELETE FROM vec_units WHERE unit_id IN ({ph})", ids)
            except Exception:
                pass
        conn.execute(f"DELETE FROM units WHERE id IN ({ph})", ids)
        conn.commit()

    def mark_parent_stale(self, unit_id: str) -> None:
        row = self._conn.execute(
            "SELECT parent_id FROM units WHERE id = ?", (unit_id,)
        ).fetchone()
        if row and row["parent_id"]:
            self._conn.execute(
                "UPDATE units SET status = 'stale' WHERE id = ? AND status = 'described'",
                (row["parent_id"],),
            )
            self._conn.commit()

    # ── read ─────────────────────────────────────────────────────────────────

    def get_unit(self, unit_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM units WHERE id = ?", (unit_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_units_for_file(self, path: str, level: int | None = None) -> list[dict]:
        if level is None:
            rows = self._conn.execute(
                "SELECT * FROM units WHERE path = ? ORDER BY level, start_line", (path,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM units WHERE path = ? AND level = ? ORDER BY start_line",
                (path, level),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_units(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM units WHERE status = 'pending' ORDER BY level, path, start_line LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stale_units(self, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM units WHERE status = 'stale' ORDER BY level, path, start_line LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_children(self, parent_id: str) -> list[dict]:
        rows = self._conn.execute("""
            SELECT u.* FROM units u
            JOIN unit_children c ON c.child_id = u.id
            WHERE c.parent_id = ?
            ORDER BY c.child_order
        """, (parent_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_file_record(self, path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM indexed_files WHERE path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    # ── judge cache ───────────────────────────────────────────────────────────

    def judge_cache_get(self, old_desc: str, new_desc: str) -> bool | None:
        key = _checksum(old_desc + "\x00" + new_desc)
        row = self._conn.execute(
            "SELECT changed FROM judge_cache WHERE key = ?", (key,)
        ).fetchone()
        return bool(row["changed"]) if row else None

    def judge_cache_set(self, old_desc: str, new_desc: str, changed: bool, reason: str = "") -> None:
        key = _checksum(old_desc + "\x00" + new_desc)
        self._conn.execute(
            "INSERT OR REPLACE INTO judge_cache(key, changed, reason, cached_at) VALUES (?,?,?,?)",
            (key, int(changed), reason, time.time()),
        )
        self._conn.commit()

    # ── search ────────────────────────────────────────────────────────────────

    def semantic_search(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        if self._vec_dims is None or self._vec_dims != len(embedding):
            return []
        try:
            import sqlite_vec
            query_blob = sqlite_vec.serialize_float32(embedding)
        except Exception:
            return []
        rows = self._conn.execute("""
            SELECT v.unit_id, v.distance,
                   u.path, u.language, u.node_type, u.name,
                   u.start_line, u.end_line, u.description, u.inferred_name, u.level
            FROM vec_units v
            JOIN units u ON u.id = v.unit_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (query_blob, top_k)).fetchall()
        return [{"score": 1.0 - r["distance"] / 2.0, **dict(r)} for r in rows]

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = self._conn
        total = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
        by_status = dict(conn.execute(
            "SELECT status, COUNT(*) FROM units GROUP BY status"
        ).fetchall())
        files = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
        return {"total": total, "by_status": by_status, "files": files}

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
