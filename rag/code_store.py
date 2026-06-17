"""Generalized unit store for hierarchical code summarization.

Schema mirrors kb/src/kb/schema.sql but inlined here for zero cross-repo
dependency. One SQLite DB per project (.agent/summaries.db by default).
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


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

CREATE INDEX IF NOT EXISTS idx_units_path_level      ON units(path, level, start_line);
CREATE INDEX IF NOT EXISTS idx_units_status          ON units(status);
CREATE INDEX IF NOT EXISTS idx_units_parent          ON units(parent_id);
CREATE INDEX IF NOT EXISTS idx_units_object_checksum ON units(object_checksum);
CREATE INDEX IF NOT EXISTS idx_units_edge_checksum   ON units(edge_set_checksum);

CREATE TABLE IF NOT EXISTS unit_children (
    parent_id   TEXT NOT NULL,
    child_id    TEXT NOT NULL,
    child_order INTEGER NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE INDEX IF NOT EXISTS idx_children_parent ON unit_children(parent_id, child_order);

CREATE TABLE IF NOT EXISTS indexed_files (
    path             TEXT PRIMARY KEY,
    checksum         TEXT NOT NULL,
    content_checksum TEXT,
    file_size        INTEGER,
    status           TEXT NOT NULL DEFAULT 'indexed',
    indexed_at       REAL
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
            from agent.core.sqlite_util import open_threadlocal_conn
            conn = open_threadlocal_conn(self._db_path, load_vec=True, foreign_keys=True)
            self._local.conn = conn
            self._run_ddl(conn)
        return self._local.conn

    def _run_ddl(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_DDL)
        # Migrations for existing DBs
        for stmt in (
            "ALTER TABLE indexed_files ADD COLUMN content_checksum TEXT",
            "ALTER TABLE indexed_files ADD COLUMN file_size INTEGER",
        ):
            try:
                conn.execute(stmt)
            except Exception:
                pass
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

    def set_file_record(
        self,
        path: str,
        checksum: str,
        status: str = "indexed",
        content_checksum: str | None = None,
        file_size: int | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO indexed_files"
            "(path, checksum, content_checksum, file_size, status, indexed_at)"
            " VALUES (?,?,?,?,?,?)",
            (path, checksum, content_checksum, file_size, status, time.time()),
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
                logger.warning("vec_units delete failed — search index may be inconsistent", exc_info=True)
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

    def get_unit_checksums_for_file(self, path: str) -> dict[str, dict]:
        """Return {unit_id: {object_checksum, status, description}} for a file (level=0 only)."""
        rows = self._conn.execute(
            "SELECT id, object_checksum, status, description FROM units WHERE path = ? AND level = 0",
            (path,),
        ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def find_described_unit_by_object_checksum(self, obj_cs: str, cache: dict | None = None) -> dict | None:
        """Find any described unit with matching object_checksum (cross-path dedup).

        If `cache` is provided (built via load_described_checksum_map), the lookup is O(1) in-memory.
        """
        if cache is not None:
            return cache.get(obj_cs)
        row = self._conn.execute(
            "SELECT * FROM units WHERE object_checksum = ? AND status = 'described' AND description IS NOT NULL LIMIT 1",
            (obj_cs,),
        ).fetchone()
        return dict(row) if row else None

    def find_described_unit_by_edge_set_checksum(self, edge_cs: str) -> dict | None:
        """Find any described rollup unit with matching edge_set_checksum (subtree dedup)."""
        row = self._conn.execute(
            "SELECT * FROM units WHERE edge_set_checksum = ? AND status = 'described' AND description IS NOT NULL LIMIT 1",
            (edge_cs,),
        ).fetchone()
        return dict(row) if row else None

    def load_described_checksum_map(self) -> dict[str, dict]:
        """Return {object_checksum: unit_dict} for all described leaf units.

        Used as a fast in-memory dedup cache for an indexing run. The caller is
        responsible for updating it when new units are described.
        """
        rows = self._conn.execute(
            """SELECT object_checksum, description, node_checksum, inferred_name, analysis_model
               FROM units
               WHERE status = 'described' AND description IS NOT NULL AND object_checksum IS NOT NULL
                 AND level = 0"""
        ).fetchall()
        return {r["object_checksum"]: dict(r) for r in rows}

    def get_described_content_checksums(self) -> set[str]:
        """Return set of content_checksums for files that are fully described."""
        rows = self._conn.execute(
            """SELECT f.content_checksum
               FROM indexed_files f
               WHERE f.content_checksum IS NOT NULL
                 AND NOT EXISTS (
                     SELECT 1 FROM units u
                     WHERE u.path = f.path AND u.status != 'described'
                 )
                 AND EXISTS (
                     SELECT 1 FROM units u WHERE u.path = f.path AND u.status = 'described'
                 )"""
        ).fetchall()
        return {r["content_checksum"] for r in rows}

    def bulk_dedup_pending(self, analysis_date: float) -> int:
        """Resolve all pending units whose object_checksum already has a described match.

        Runs as a single SQL pass — no LLM involved. Returns count of units resolved.
        """
        conn = self._conn
        # Collect pending units that have a described counterpart with the same checksum.
        rows = conn.execute("""
            SELECT u.id,
                   d.description, d.node_checksum, d.inferred_name, d.analysis_model
            FROM units u
            JOIN units d ON d.object_checksum = u.object_checksum
                         AND d.status = 'described'
                         AND d.description IS NOT NULL
                         AND d.id != u.id
            WHERE u.status = 'pending'
            LIMIT 10000
        """).fetchall()
        if not rows:
            return 0
        conn.executemany("""
            UPDATE units
               SET status         = 'described',
                   description    = ?,
                   node_checksum  = ?,
                   inferred_name  = ?,
                   analysis_model = ?,
                   analysis_date  = ?
             WHERE id = ?
        """, [
            (r["description"], r["node_checksum"], r["inferred_name"], r["analysis_model"], analysis_date, r["id"])
            for r in rows
        ])
        conn.commit()
        # Mark parents stale for anything that just got described.
        ids = [r["id"] for r in rows]
        ph = ",".join("?" * len(ids))
        conn.execute(f"""
            UPDATE units SET status = 'stale'
            WHERE status = 'described'
              AND id IN (
                  SELECT DISTINCT parent_id FROM units WHERE id IN ({ph}) AND parent_id IS NOT NULL
              )
        """, ids)
        conn.commit()
        return len(rows)

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

    def stats_by_level(self) -> dict[int, dict[str, int]]:
        """Return {level: {status: count}} for all levels."""
        rows = self._conn.execute(
            "SELECT level, status, COUNT(*) FROM units GROUP BY level, status"
        ).fetchall()
        result: dict[int, dict[str, int]] = {}
        for level, status, count in rows:
            result.setdefault(level, {})[status] = count
        return result

    def max_level(self) -> int:
        row = self._conn.execute("SELECT MAX(level) FROM units").fetchone()
        return row[0] or 0

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
