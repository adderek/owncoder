"""General-purpose text memory store: FTS5 + optional sqlite-vec embeddings.

Backs all memory tiers:
  scope='facts_round'      — per-session compaction rounds (semantic recall)
  scope='note'             — user/agent-saved cross-session notes
  scope='session_summary'  — session summaries for cross-session search
  scope='context_chunk'    — pre-chunked context file sections

DB lives at a caller-specified path:
  per-session: <session_dir>/memory.db
  project:     <agent_dir>/memory.db
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _has_all_tags(tags_json: str | None, required: list[str]) -> bool:
    if not required:
        return True
    try:
        tags = json.loads(tags_json or "[]")
    except Exception:
        return False
    tag_set = set(tags)
    return all(t in tag_set for t in required)


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(Path(db_path).resolve())
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._vec_dims: int | None = None
        self._setup()
        self._vec_dims = self._read_vec_dims()

    # ── connection ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            from agent.core.sqlite_util import apply_concurrency_pragmas
            apply_concurrency_pragmas(conn)
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception:
                pass  # vec search degraded to FTS only
            conn.enable_load_extension(False)
            self._local.conn = conn
        return self._local.conn

    def _setup(self) -> None:
        self._conn().executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id          TEXT PRIMARY KEY,
                scope       TEXT NOT NULL,
                source      TEXT,
                title       TEXT,
                body        TEXT NOT NULL,
                tags        TEXT,
                created_at  REAL,
                updated_at  REAL
            );
            CREATE INDEX IF NOT EXISTS idx_entries_scope ON entries(scope);
            CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
                body, title, tags,
                content=entries,
                content_rowid=rowid
            );
            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self._conn().commit()
        # Migration: add hit_count if missing (SQLite doesn't support IF NOT EXISTS for columns).
        try:
            self._conn().execute("ALTER TABLE entries ADD COLUMN hit_count INTEGER DEFAULT 0")
            self._conn().commit()
        except Exception:
            pass

    # ── vec table ───────────────────────────────────────────────────────────

    def _read_vec_dims(self) -> int | None:
        row = self._conn().execute(
            "SELECT value FROM _meta WHERE key='embedding_dims'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _ensure_vec(self, dims: int) -> bool:
        if self._vec_dims == dims:
            return True
        try:
            self._conn().execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_entries USING vec0(
                    entry_id TEXT PRIMARY KEY,
                    embedding float[{dims}] distance_metric=cosine
                )
            """)
            self._conn().execute(
                "INSERT OR REPLACE INTO _meta(key,value) VALUES('embedding_dims',?)",
                (str(dims),),
            )
            self._conn().commit()
            self._vec_dims = dims
            return True
        except Exception:
            return False

    # ── write ────────────────────────────────────────────────────────────────

    def add(
        self,
        scope: str,
        body: str,
        *,
        source: str = "",
        title: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        entry_id: str | None = None,
    ) -> str:
        eid = entry_id or str(uuid.uuid4())
        now = time.time()
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        conn = self._conn()

        existing = conn.execute(
            "SELECT rowid FROM entries WHERE id=?", (eid,)
        ).fetchone()
        if existing:
            conn.execute(
                "DELETE FROM entries_fts WHERE rowid=?", (existing["rowid"],)
            )

        cur = conn.execute(
            """INSERT OR REPLACE INTO entries
               (id, scope, source, title, body, tags, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (eid, scope, source, title, body, tags_json, now, now),
        )
        conn.execute(
            "INSERT INTO entries_fts(rowid,body,title,tags) VALUES(?,?,?,?)",
            (cur.lastrowid, body, title or "", tags_json),
        )

        if embedding:
            dims = len(embedding)
            if self._ensure_vec(dims):
                try:
                    import sqlite_vec
                    conn.execute(
                        "DELETE FROM vec_entries WHERE entry_id=?", (eid,)
                    )
                    conn.execute(
                        "INSERT INTO vec_entries(entry_id,embedding) VALUES(?,?)",
                        (eid, sqlite_vec.serialize_float32(embedding)),
                    )
                except Exception:
                    pass

        conn.commit()
        return eid

    def delete(self, entry_id: str) -> None:
        conn = self._conn()
        row = conn.execute(
            "SELECT rowid FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM entries_fts WHERE rowid=?", (row["rowid"],))
        if self._vec_dims is not None:
            conn.execute("DELETE FROM vec_entries WHERE entry_id=?", (entry_id,))
        conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        conn.commit()

    # ── read ─────────────────────────────────────────────────────────────────

    def get(self, entry_id: str) -> dict | None:
        row = self._conn().execute(
            "SELECT * FROM entries WHERE id=?", (entry_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_entries(
        self,
        scope: str | None = None,
        limit: int = 100,
        order_by: str = "created_at DESC",
    ) -> list[dict]:
        conn = self._conn()
        if scope:
            rows = conn.execute(
                f"SELECT * FROM entries WHERE scope=? ORDER BY {order_by} LIMIT ?",
                (scope, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM entries ORDER BY {order_by} LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── search ────────────────────────────────────────────────────────────────

    def update_source_tags(self, scope: str, source: str, tags: list[str]) -> int:
        """Update tags for all entries matching scope+source. Returns count updated."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT rowid, id, body, title FROM entries WHERE scope=? AND source=?",
            (scope, source),
        ).fetchall()
        if not rows:
            return 0
        tags_json = json.dumps(tags, ensure_ascii=False)
        now = time.time()
        for row in rows:
            # Delete FTS entry before changing content table to avoid inconsistency.
            conn.execute("DELETE FROM entries_fts WHERE rowid=?", (row["rowid"],))
            conn.execute(
                "UPDATE entries SET tags=?, updated_at=? WHERE rowid=?",
                (tags_json, now, row["rowid"]),
            )
            conn.execute(
                "INSERT INTO entries_fts(rowid,body,title,tags) VALUES(?,?,?,?)",
                (row["rowid"], row["body"] or "", row["title"] or "", tags_json),
            )
        conn.commit()
        return len(rows)

    def _tag_filter_sql(self, tags_filter: list[str] | None) -> tuple[str, list]:
        """Return (sql_fragment, params) for filtering entries by tags."""
        if not tags_filter:
            return "", []
        clauses = [
            "EXISTS (SELECT 1 FROM json_each(e.tags) WHERE json_each.value=?)"
            for _ in tags_filter
        ]
        return " AND " + " AND ".join(clauses), list(tags_filter)

    def fts_search(
        self,
        query: str,
        scope: str | None = None,
        top_k: int = 10,
        tags_filter: list[str] | None = None,
    ) -> list[dict]:
        if not query.strip():
            return []
        conn = self._conn()
        tag_sql, tag_params = self._tag_filter_sql(tags_filter)
        try:
            if scope:
                rows = conn.execute(
                    f"""SELECT e.*, bm25(entries_fts) AS score
                       FROM entries_fts
                       JOIN entries e ON e.rowid = entries_fts.rowid
                       WHERE entries_fts MATCH ? AND e.scope=?{tag_sql}
                       ORDER BY score LIMIT ?""",
                    (query, scope, *tag_params, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""SELECT e.*, bm25(entries_fts) AS score
                       FROM entries_fts
                       JOIN entries e ON e.rowid = entries_fts.rowid
                       WHERE entries_fts MATCH ?{tag_sql}
                       ORDER BY score LIMIT ?""",
                    (query, *tag_params, top_k),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    def vector_search(
        self,
        embedding: list[float],
        scope: str | None = None,
        top_k: int = 10,
        tags_filter: list[str] | None = None,
    ) -> list[dict]:
        if self._vec_dims is None or self._vec_dims != len(embedding):
            return []
        tag_sql, tag_params = self._tag_filter_sql(tags_filter)
        try:
            import sqlite_vec
            blob = sqlite_vec.serialize_float32(embedding)
            conn = self._conn()
            if scope:
                rows = conn.execute(
                    f"""SELECT ve.entry_id, ve.distance, e.*
                       FROM vec_entries ve
                       JOIN entries e ON e.id = ve.entry_id
                       WHERE ve.embedding MATCH ? AND k=? AND e.scope=?{tag_sql}
                       ORDER BY ve.distance""",
                    (blob, top_k * 3, scope, *tag_params),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""SELECT ve.entry_id, ve.distance, e.*
                       FROM vec_entries ve
                       JOIN entries e ON e.id = ve.entry_id
                       WHERE ve.embedding MATCH ? AND k=?{tag_sql}
                       ORDER BY ve.distance""",
                    (blob, top_k * 3, *tag_params),
                ).fetchall()
            result = [
                {"score": 1.0 - r["distance"] / 2.0, **dict(r)} for r in rows
            ]
            if scope:
                result = [r for r in result if r.get("scope") == scope]
            if tags_filter:
                result = [r for r in result if _has_all_tags(r.get("tags"), tags_filter)]
            return result[:top_k]
        except Exception:
            return []

    def hybrid_search(
        self,
        query: str,
        embedding: list[float] | None = None,
        scope: str | None = None,
        top_k: int = 10,
        tags_filter: list[str] | None = None,
    ) -> list[dict]:
        vec_results = (
            self.vector_search(embedding, scope=scope, top_k=top_k * 2, tags_filter=tags_filter)
            if embedding
            else []
        )
        fts_results = self.fts_search(query, scope=scope, top_k=top_k * 2, tags_filter=tags_filter)

        def _norm(id_score: list[tuple[str, float]]) -> dict[str, float]:
            if not id_score:
                return {}
            scores = [s for _, s in id_score]
            mn, mx = min(scores), max(scores)
            if mx == mn:
                return {i: 1.0 for i, _ in id_score}
            return {i: (s - mn) / (mx - mn) for i, s in id_score}

        vec_norm = _norm([(r["id"], r["score"]) for r in vec_results])
        fts_norm = _norm([(r["id"], -r["score"]) for r in fts_results])

        all_entries: dict[str, dict] = {}
        for r in vec_results:
            all_entries[r["id"]] = r
        for r in fts_results:
            all_entries.setdefault(r["id"], r)

        combined = []
        for eid, entry in all_entries.items():
            v = vec_norm.get(eid, 0.0)
            b = fts_norm.get(eid, 0.0)
            combined.append((0.6 * v + 0.4 * b, entry))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [{"combined_score": s, **d} for s, d in combined[:top_k]]

    def increment_hit_count(self, entry_id: str) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE entries SET hit_count = COALESCE(hit_count, 0) + 1, updated_at = ? WHERE id = ?",
            (time.time(), entry_id),
        )
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
