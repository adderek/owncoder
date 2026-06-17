from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path


class ArchiveStore:
    """Cold storage for chunks that were removed from the main index.

    Separate sqlite file so archived rows cannot leak into normal search.
    Embeddings are kept as raw float32 blobs in a regular column — no vec0
    index (KNN is not used on the cold path; FTS is enough). When a path is
    restored, the blob is re-inserted into the main store's vec0 table.
    """

    def __init__(self, db_path: str) -> None:
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(p.resolve())
        self._local = threading.local()
        self._setup()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            from agent.core.sqlite_util import open_threadlocal_conn
            self._local.conn = open_threadlocal_conn(self._db_path)
        return self._local.conn

    def _setup(self) -> None:
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                language TEXT,
                node_type TEXT,
                name TEXT,
                start_line INTEGER,
                end_line INTEGER,
                content TEXT NOT NULL,
                mtime REAL,
                git_hash TEXT,
                embedding BLOB,
                embedding_dims INTEGER,
                archived_at REAL NOT NULL,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE INDEX IF NOT EXISTS idx_chunks_archived_at ON chunks(archived_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                name,
                path,
                content=chunks,
                content_rowid=rowid
            );
        """)
        conn.commit()

    def ingest(self, rows: list[dict], reason: str) -> int:
        """Insert rows (as produced by VectorStore.rows_for_paths) into the
        archive. Idempotent on `id` — re-archiving the same chunk refreshes
        archived_at and reason."""
        if not rows:
            return 0
        conn = self._conn()
        now = time.time()
        count = 0
        for r in rows:
            existing = conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (r["id"],)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))
            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (id, path, language, node_type, name, start_line, end_line,
                 content, mtime, git_hash, embedding, embedding_dims,
                 archived_at, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r["id"], r["path"], r.get("language"), r.get("node_type"),
                r.get("name"), r.get("start_line"), r.get("end_line"),
                r["content"], r.get("mtime"), r.get("git_hash"),
                r.get("embedding_blob"), r.get("embedding_dims"),
                now, reason,
            ))
            new_row = conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (r["id"],)
            ).fetchone()
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
                (new_row["rowid"], r["content"], r.get("name") or "", r["path"]),
            )
            count += 1
        conn.commit()
        return count

    def pop_paths(self, paths: list[str]) -> list[dict]:
        """Return rows for the given paths (in the shape expected by
        VectorStore.insert_raw) and delete them from the archive. Matches
        exact path OR any archived path whose stored value ends with the
        given string (so users can pass 'src/foo.py' against an absolute
        indexed path)."""
        if not paths:
            return []
        conn = self._conn()
        matched_ids: list[int] = []
        rows: list = []
        for p in paths:
            hits = conn.execute(
                "SELECT rowid, * FROM chunks WHERE path = ? OR path LIKE ?",
                (p, f"%/{p}"),
            ).fetchall()
            for h in hits:
                if h["rowid"] not in matched_ids:
                    matched_ids.append(h["rowid"])
                    rows.append(h)
        if not rows:
            return []
        result = []
        for r in rows:
            d = dict(r)
            d["embedding_blob"] = bytes(d["embedding"]) if d.get("embedding") else None
            result.append(d)
        ids = [r["id"] for r in rows]
        rowids = [r["rowid"] for r in rows]
        id_ph = ",".join("?" * len(ids))
        rid_ph = ",".join("?" * len(rowids))
        conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({rid_ph})", rowids)
        conn.execute(f"DELETE FROM chunks WHERE id IN ({id_ph})", ids)
        conn.commit()
        return result

    def purge_expired(self, ttl_days: int) -> int:
        """Permanently delete rows older than ttl_days. ttl_days <= 0 disables
        expiration. Returns number of rows removed."""
        if ttl_days <= 0:
            return 0
        cutoff = time.time() - ttl_days * 86400
        conn = self._conn()
        rows = conn.execute(
            "SELECT rowid FROM chunks WHERE archived_at < ?", (cutoff,)
        ).fetchall()
        if not rows:
            return 0
        rowids = [r["rowid"] for r in rows]
        ph = ",".join("?" * len(rowids))
        conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({ph})", rowids)
        conn.execute(f"DELETE FROM chunks WHERE rowid IN ({ph})", rowids)
        conn.commit()
        conn.execute("VACUUM")
        return len(rowids)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        try:
            rows = self._conn().execute("""
                SELECT c.id, c.path, c.language, c.node_type, c.name,
                       c.start_line, c.end_line, c.content,
                       c.archived_at, c.reason,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (query, top_k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        conn = self._conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        paths = conn.execute("SELECT COUNT(DISTINCT path) as cnt FROM chunks").fetchone()
        oldest = conn.execute("SELECT MIN(archived_at) as m FROM chunks").fetchone()
        return {
            "chunks": row["cnt"],
            "files": paths["cnt"],
            "oldest_archived_at": oldest["m"],
        }

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
