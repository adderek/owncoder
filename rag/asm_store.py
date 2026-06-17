from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import RAGConfig


def _content_checksum(lines: list[str]) -> str:
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


class AsmStore:
    def __init__(self, cfg: "RAGConfig") -> None:
        db_path = Path(cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path.resolve())
        self._local = threading.local()
        # Initialise tables using the main-thread connection, then read vec dims.
        self._setup()
        self._vec_dims: int | None = self._read_vec_dims()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating one if needed.

        sqlite-vec extensions are registered per-connection and SQLite virtual
        table cursors have thread affinity even when check_same_thread=False is
        set, so we give each thread its own connection instead of sharing one.
        """
        if not hasattr(self._local, "conn"):
            from agent.core.sqlite_util import open_threadlocal_conn
            # load_vec on every conn: the vec0 module must be registered for BOTH
            # the write path (_ensure_vec_table / upsert) and the search path,
            # not just at query time on one connection.
            conn = open_threadlocal_conn(self._db_path, load_vec=True, foreign_keys=True)
            self._local.conn = conn
            # Re-run idempotent DDL so the new connection knows about all tables.
            self._setup_conn(conn)
        return self._local.conn

    def _setup_conn(self, conn: sqlite3.Connection) -> None:
        """Load extension + run DDL on a single connection (idempotent)."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS asm_units (
                id          TEXT PRIMARY KEY,
                path        TEXT NOT NULL,
                level       INTEGER NOT NULL,
                start_line  INTEGER NOT NULL,
                end_line    INTEGER NOT NULL,
                description TEXT,
                inferred_name TEXT,
                checksum    TEXT NOT NULL,
                revision    INTEGER NOT NULL DEFAULT 1,
                parent_id   TEXT,
                prev_id     TEXT,
                next_id     TEXT,
                status      TEXT NOT NULL DEFAULT 'pending',
                confidence  TEXT,
                mtime       REAL,
                git_hash    TEXT,
                calls       TEXT,
                side_effects TEXT,
                key_patterns TEXT
            );

            CREATE INDEX IF NOT EXISTS asm_units_path_level
                ON asm_units(path, level, start_line);

            CREATE TABLE IF NOT EXISTS asm_children (
                parent_id   TEXT NOT NULL,
                child_id    TEXT NOT NULL,
                child_order INTEGER NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            );

            CREATE INDEX IF NOT EXISTS asm_children_parent
                ON asm_children(parent_id, child_order);

            CREATE TABLE IF NOT EXISTS _asm_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS asm_files (
                path        TEXT PRIMARY KEY,
                checksum    TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'split',
                analyzed_at REAL
            );

            CREATE TABLE IF NOT EXISTS asm_split_progress (
                path         TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                start_line   INTEGER NOT NULL,
                end_line     INTEGER NOT NULL,
                boundaries   TEXT NOT NULL,
                PRIMARY KEY (path, window_index)
            );
        """)
        conn.commit()

    def _setup(self) -> None:
        """Initialise the main-thread connection and run DDL."""
        self._setup_conn(self._conn)

    def _read_vec_dims(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM _asm_meta WHERE key = 'vec_asm_dims'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _ensure_vec_table(self, dims: int) -> None:
        conn = self._conn  # thread-local connection
        try:
            import sqlite_vec
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec: {e}") from e
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_asm_units USING vec0(
                unit_id TEXT PRIMARY KEY,
                embedding float[{dims}] distance_metric=cosine
            )
        """)
        if self._vec_dims != dims:
            conn.execute(
                "INSERT OR REPLACE INTO _asm_meta(key, value) VALUES ('vec_asm_dims', ?)",
                (str(dims),),
            )
            self._vec_dims = dims
        conn.commit()

    def upsert_unit(self, unit: dict) -> None:
        conn = self._conn
        conn.execute("""
            INSERT OR REPLACE INTO asm_units
            (id, path, level, start_line, end_line, description, inferred_name,
             checksum, revision, parent_id, prev_id, next_id, status, confidence,
             mtime, git_hash, calls, side_effects, key_patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            unit["id"], unit["path"], unit["level"],
            unit["start_line"], unit["end_line"],
            unit.get("description"), unit.get("inferred_name"),
            unit["checksum"],
            unit.get("revision", 1),
            unit.get("parent_id"), unit.get("prev_id"), unit.get("next_id"),
            unit.get("status", "pending"), unit.get("confidence"),
            unit.get("mtime"), unit.get("git_hash"),
            unit.get("calls"), unit.get("side_effects"), unit.get("key_patterns"),
        ))

        if unit.get("embedding"):
            import sqlite_vec
            emb = unit["embedding"]
            dims = len(emb)
            self._ensure_vec_table(dims)
            conn.execute("DELETE FROM vec_asm_units WHERE unit_id = ?", (unit["id"],))
            conn.execute(
                "INSERT INTO vec_asm_units(unit_id, embedding) VALUES (?, ?)",
                (unit["id"], sqlite_vec.serialize_float32(emb)),
            )

        conn.commit()

    def get_unit(self, unit_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM asm_units WHERE id = ?", (unit_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_units_for_file(self, path: str, level: int | None = None) -> list[dict]:
        if level is None:
            rows = self._conn.execute(
                "SELECT * FROM asm_units WHERE path = ? ORDER BY level, start_line",
                (path,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM asm_units WHERE path = ? AND level = ? ORDER BY start_line",
                (path, level),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_units(self, level: int | None = None) -> list[dict]:
        if level is None:
            rows = self._conn.execute(
                "SELECT * FROM asm_units ORDER BY path, level, start_line"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM asm_units WHERE level = ? ORDER BY path, start_line",
                (level,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_children(self, parent_id: str) -> list[dict]:
        rows = self._conn.execute("""
            SELECT u.* FROM asm_units u
            JOIN asm_children c ON c.child_id = u.id
            WHERE c.parent_id = ?
            ORDER BY c.child_order
        """, (parent_id,)).fetchall()
        return [dict(r) for r in rows]

    def upsert_children(self, parent_id: str, child_ids: list[str]) -> None:
        self._conn.execute("DELETE FROM asm_children WHERE parent_id = ?", (parent_id,))
        self._conn.executemany(
            "INSERT INTO asm_children(parent_id, child_id, child_order) VALUES (?, ?, ?)",
            [(parent_id, cid, i) for i, cid in enumerate(child_ids)],
        )
        self._conn.commit()

    def mark_pending_above(self, unit_id: str) -> None:
        """Walk up parent chain and set status='pending' on each ancestor."""
        current_id = unit_id
        visited: set[str] = set()
        while True:
            row = self._conn.execute(
                "SELECT parent_id FROM asm_units WHERE id = ?", (current_id,)
            ).fetchone()
            if not row or not row["parent_id"] or row["parent_id"] in visited:
                break
            parent_id = row["parent_id"]
            visited.add(parent_id)
            self._conn.execute(
                "UPDATE asm_units SET status='pending' WHERE id = ?", (parent_id,)
            )
            current_id = parent_id
        self._conn.commit()

    def delete_units_for_file(self, path: str) -> None:
        ids = [r["id"] for r in self._conn.execute(
            "SELECT id FROM asm_units WHERE path = ?", (path,)
        ).fetchall()]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM asm_children WHERE parent_id IN ({placeholders}) "
            f"OR child_id IN ({placeholders})",
            ids + ids,
        )
        if self._vec_dims is not None:
            self._conn.execute(
                f"DELETE FROM vec_asm_units WHERE unit_id IN ({placeholders})", ids
            )
        self._conn.execute(f"DELETE FROM asm_units WHERE id IN ({placeholders})", ids)
        self._conn.commit()

    def get_pending_units(self, path: str, level: int) -> list[dict]:
        rows = self._conn.execute("""
            SELECT * FROM asm_units
            WHERE path = ? AND level = ? AND status = 'pending'
            ORDER BY start_line
        """, (path, level)).fetchall()
        return [dict(r) for r in rows]

    def semantic_search(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        if self._vec_dims is None or self._vec_dims != len(embedding):
            return []
        conn = self._conn  # thread-local connection — vec0 already loaded by _conn
        try:
            import sqlite_vec
            query_blob = sqlite_vec.serialize_float32(embedding)
        except Exception:
            return []
        rows = conn.execute("""
            SELECT v.unit_id, v.distance,
                   u.id, u.path, u.level, u.start_line, u.end_line,
                   u.description, u.inferred_name, u.status, u.confidence
            FROM vec_asm_units v
            JOIN asm_units u ON u.id = v.unit_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (query_blob, top_k)).fetchall()
        return [{"score": 1.0 - row["distance"] / 2.0, **dict(row)} for row in rows]

    def save_split_window(
        self, path: str, window_index: int, start_line: int, end_line: int, boundaries: list[int]
    ) -> None:
        import json
        self._conn.execute(
            "INSERT OR REPLACE INTO asm_split_progress"
            "(path, window_index, start_line, end_line, boundaries) VALUES (?, ?, ?, ?, ?)",
            (path, window_index, start_line, end_line, json.dumps(boundaries)),
        )
        self._conn.commit()

    def load_split_windows(self, path: str) -> dict[int, dict]:
        import json
        rows = self._conn.execute(
            "SELECT window_index, start_line, end_line, boundaries"
            " FROM asm_split_progress WHERE path = ? ORDER BY window_index",
            (path,),
        ).fetchall()
        return {
            r["window_index"]: {
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "boundaries": json.loads(r["boundaries"]),
            }
            for r in rows
        }

    def clear_split_windows(self, path: str) -> None:
        self._conn.execute("DELETE FROM asm_split_progress WHERE path = ?", (path,))
        self._conn.commit()

    def get_file_record(self, path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT checksum, status FROM asm_files WHERE path = ?", (path,)
        ).fetchone()
        return dict(row) if row else None

    def set_file_record(self, path: str, checksum: str, status: str = "split") -> None:
        import time
        self._conn.execute(
            "INSERT OR REPLACE INTO asm_files(path, checksum, status, analyzed_at) VALUES (?, ?, ?, ?)",
            (path, checksum, status, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        # Close only the calling thread's connection (typically the main thread).
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
