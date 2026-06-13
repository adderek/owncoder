from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent.config import RAGConfig


class VectorStore:
    def __init__(self, cfg: "RAGConfig") -> None:
        self._cfg = cfg
        db_path = Path(cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path.resolve())
        self._local = threading.local()
        self._setup()
        # Dimension of the vec0 KNN table; set on first embedding insert, read back from _meta.
        self._vec_dims: int | None = self._read_vec_dims()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            from agent.core.sqlite_util import apply_concurrency_pragmas
            apply_concurrency_pragmas(conn)
            conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(conn)
            except Exception as e:
                raise RuntimeError(f"Failed to load sqlite-vec: {e}") from e
            conn.enable_load_extension(False)
            self._local.conn = conn
        return self._local.conn

    def _setup(self) -> None:
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
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
                git_hash TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                name,
                path,
                content=chunks,
                content_rowid=rowid
            );

            CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_mtimes (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL
            );
        """)
        conn.commit()

    def _read_vec_dims(self) -> int | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM _meta WHERE key = 'embedding_dims'"
        ).fetchone()
        if not row:
            return None
        meta_dims = int(row["value"])
        # Verify the actual table schema matches _meta — they can diverge when the
        # embedding model changes (table created at old dims, meta updated separately).
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_chunks'"
        ).fetchone()
        if ddl is None:
            # Table missing; reset so _ensure_vec_table recreates it.
            conn.execute("DELETE FROM _meta WHERE key='embedding_dims'")
            conn.commit()
            return None
        if f"float[{meta_dims}]" not in (ddl["sql"] or ""):
            # Schema mismatch: drop stale table and clear meta so it gets rebuilt.
            actual = (ddl["sql"] or "").split("float[")[-1].split("]")[0] if "float[" in (ddl["sql"] or "") else "?"
            logger.warning(
                "Embedding dimension mismatch: index.db has float[%s] but config expects %d dims. "
                "Dropping stale vec_chunks table. Run `agent init --force` to rebuild all embeddings; "
                "`agent index --update` will only re-embed files changed since last index.",
                actual, meta_dims,
            )
            conn.executescript("DROP TABLE IF EXISTS vec_chunks;")
            conn.execute("DELETE FROM _meta WHERE key='embedding_dims'")
            conn.commit()
            return None
        return meta_dims

    def _ensure_vec_table(self, dims: int) -> None:
        """Create the vec0 KNN table if it doesn't exist for the given dimensions."""
        if self._vec_dims == dims:
            return
        conn = self._get_conn()
        # Drop any stale table with the wrong dims before creating the new one.
        if self._vec_dims is not None:
            logger.warning(
                "Embedding dimensions changed (%d → %d); dropping vec_chunks. "
                "Run `agent init --force` to rebuild all embeddings.",
                self._vec_dims, dims,
            )
            conn.executescript("DROP TABLE IF EXISTS vec_chunks;")
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding float[{dims}] distance_metric=cosine
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO _meta(key, value) VALUES ('embedding_dims', ?)",
            (str(dims),),
        )
        conn.commit()
        self._vec_dims = dims

    def upsert(self, chunk: dict) -> None:
        conn = self._get_conn()
        # Remove stale FTS entry (INSERT OR REPLACE on chunks changes the rowid).
        existing = conn.execute(
            "SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))

        cur = conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["path"], chunk.get("language"), chunk.get("node_type"),
            chunk.get("name"), chunk.get("start_line"), chunk.get("end_line"),
            chunk["content"], chunk.get("mtime"), chunk.get("git_hash"),
        ))
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
            (cur.lastrowid, chunk["content"], chunk.get("name") or "", chunk["path"]),
        )

        if chunk.get("embedding"):
            import sqlite_vec
            emb = chunk["embedding"]
            dims = len(emb)
            self._ensure_vec_table(dims)
            conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk["id"],))
            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (chunk["id"], sqlite_vec.serialize_float32(emb)),
            )

        conn.commit()

    def upsert_many(self, chunks: list[dict], fresh: bool = False) -> None:
        """Insert chunks. Set fresh=True when delete_by_path was already called for
        these chunks' path — skips the per-chunk rowid lookup and stale FTS/vec cleanup."""
        conn = self._get_conn()
        for chunk in chunks:
            if not fresh:
                existing = conn.execute(
                    "SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)
                ).fetchone()
                if existing:
                    conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))

            cur = conn.execute("""
                INSERT OR REPLACE INTO chunks
                (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["id"], chunk["path"], chunk.get("language"), chunk.get("node_type"),
                chunk.get("name"), chunk.get("start_line"), chunk.get("end_line"),
                chunk["content"], chunk.get("mtime"), chunk.get("git_hash"),
            ))
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
                (cur.lastrowid, chunk["content"], chunk.get("name") or "", chunk["path"]),
            )

            if chunk.get("embedding"):
                import sqlite_vec
                emb = chunk["embedding"]
                dims = len(emb)
                self._ensure_vec_table(dims)
                # Always delete before insert — cheap primary-key lookup, prevents
                # UNIQUE violations when legacy absolute-path chunks share the same id.
                conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk["id"],))
                conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (chunk["id"], sqlite_vec.serialize_float32(emb)),
                )

        conn.commit()

    def delete_by_path(self, path: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM file_mtimes WHERE path = ?", (path,))
        rows = conn.execute("SELECT id, rowid FROM chunks WHERE path = ?", (path,)).fetchall()
        if not rows:
            conn.commit()
            return
        ids = [r["id"] for r in rows]
        rowids = [r["rowid"] for r in rows]
        placeholders = ",".join("?" * len(ids))
        rowid_placeholders = ",".join("?" * len(rowids))
        conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({rowid_placeholders})", rowids)
        if self._vec_dims is not None:
            conn.execute(f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        conn.commit()

    def list_paths(self) -> list[str]:
        rows = self._get_conn().execute(
            "SELECT DISTINCT path FROM chunks ORDER BY path"
        ).fetchall()
        return [r["path"] for r in rows]

    def rows_for_paths(self, paths: list[str]) -> list[dict]:
        """Return chunk rows for the given paths, including raw embedding blobs
        from vec_chunks (as bytes) when present. Used by the archive pipeline."""
        if not paths:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(paths))
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE path IN ({placeholders})", paths
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            emb_blob = None
            if self._vec_dims is not None:
                emb_row = conn.execute(
                    "SELECT embedding FROM vec_chunks WHERE chunk_id = ?",
                    (d["id"],),
                ).fetchone()
                if emb_row is not None:
                    emb_blob = bytes(emb_row["embedding"])
            d["embedding_blob"] = emb_blob
            d["embedding_dims"] = self._vec_dims if emb_blob else None
            result.append(d)
        return result

    def insert_raw(self, row: dict) -> None:
        """Insert a row (as returned by rows_for_paths) back into the main store,
        preserving its embedding blob. Used for archive restore."""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT rowid FROM chunks WHERE id = ?", (row["id"],)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))
        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["id"], row["path"], row.get("language"), row.get("node_type"),
            row.get("name"), row.get("start_line"), row.get("end_line"),
            row["content"], row.get("mtime"), row.get("git_hash"),
        ))
        new_row = conn.execute("SELECT rowid FROM chunks WHERE id = ?", (row["id"],)).fetchone()
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
            (new_row["rowid"], row["content"], row.get("name") or "", row["path"]),
        )
        emb_blob = row.get("embedding_blob")
        dims = row.get("embedding_dims")
        if emb_blob and dims:
            self._ensure_vec_table(dims)
            conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (row["id"],))
            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (row["id"], emb_blob),
            )
        conn.commit()

    def get_mtime(self, path: str) -> float | None:
        row = self._get_conn().execute(
            "SELECT mtime FROM chunks WHERE path = ? LIMIT 1", (path,)
        ).fetchone()
        if row:
            return row["mtime"]
        row = self._get_conn().execute(
            "SELECT mtime FROM file_mtimes WHERE path = ? LIMIT 1", (path,)
        ).fetchone()
        return row["mtime"] if row else None

    def set_file_mtime(self, path: str, mtime: float) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO file_mtimes(path, mtime) VALUES (?, ?)",
            (path, mtime),
        )
        conn.commit()

    def vector_search(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        import sqlite_vec
        if self._vec_dims is None or self._vec_dims != len(embedding):
            return []
        query_blob = sqlite_vec.serialize_float32(embedding)
        rows = self._get_conn().execute("""
            SELECT vc.chunk_id, vc.distance,
                   c.id, c.path, c.language, c.node_type, c.name,
                   c.start_line, c.end_line, c.content
            FROM vec_chunks vc
            JOIN chunks c ON c.id = vc.chunk_id
            WHERE vc.embedding MATCH ? AND k = ?
            ORDER BY vc.distance
        """, (query_blob, top_k)).fetchall()
        # vec0 cosine distance: 0 = identical, 2 = opposite. Convert to similarity score.
        return [{"score": 1.0 - row["distance"] / 2.0, **dict(row)} for row in rows]

    def fts_search(self, query: str, top_k: int = 20) -> list[dict]:
        try:
            rows = self._get_conn().execute("""
                SELECT c.id, c.path, c.language, c.node_type, c.name,
                       c.start_line, c.end_line, c.content,
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

    def hybrid_search(self, query: str, embedding: list[float], top_k: int = 8) -> list[dict]:
        vec_results = self.vector_search(embedding, top_k=20)
        fts_results = self.fts_search(query, top_k=20)

        # Normalize a list of (id, score) where higher is better to [0, 1].
        def normalize(id_score: list[tuple[str, float]]) -> dict[str, float]:
            if not id_score:
                return {}
            scores = [s for _, s in id_score]
            mn, mx = min(scores), max(scores)
            if mx == mn:
                return {i: 1.0 for i, _ in id_score}
            return {i: (s - mn) / (mx - mn) for i, s in id_score}

        # vec score is already similarity (higher = better)
        vec_norm = normalize([(r["id"], r["score"]) for r in vec_results])
        # bm25 returns negative values — negate so higher is better
        fts_norm = normalize([(r["id"], -r["score"]) for r in fts_results])

        all_chunks: dict[str, dict] = {}
        for r in vec_results:
            all_chunks[r["id"]] = r
        for r in fts_results:
            all_chunks.setdefault(r["id"], r)

        combined = []
        for cid, chunk in all_chunks.items():
            v = vec_norm.get(cid, 0.0)
            b = fts_norm.get(cid, 0.0)
            combined.append((0.6 * v + 0.4 * b, chunk))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [{"combined_score": s, **d} for s, d in combined[:top_k]]

    def stats(self) -> dict:
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        paths = conn.execute("SELECT COUNT(DISTINCT path) as cnt FROM chunks").fetchone()
        return {"chunks": row["cnt"], "files": paths["cnt"]}

    def get_indexed_mtimes(self) -> dict[str, float]:
        """Return {path: mtime} for all indexed files (including chunk-less visited files)."""
        conn = self._get_conn()
        result: dict[str, float] = {}
        for r in conn.execute("SELECT path, mtime FROM file_mtimes").fetchall():
            result[r["path"]] = r["mtime"]
        # chunks entries override file_mtimes (chunks are authoritative when present)
        for r in conn.execute("SELECT path, mtime FROM chunks GROUP BY path").fetchall():
            result[r["path"]] = r["mtime"]
        return result

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn
