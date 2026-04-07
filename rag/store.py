from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import RAGConfig


class VectorStore:
    def __init__(self, cfg: "RAGConfig") -> None:
        self._cfg = cfg
        db_path = Path(cfg.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        self._setup()
        self._conn.enable_load_extension(False)
        # Dimension of the vec0 KNN table; set on first embedding insert, read back from _meta.
        self._vec_dims: int | None = self._read_vec_dims()

    def _setup(self) -> None:
        conn = self._conn
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec: {e}") from e

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
        """)
        conn.commit()

    def _read_vec_dims(self) -> int | None:
        row = self._conn.execute(
            "SELECT value FROM _meta WHERE key = 'embedding_dims'"
        ).fetchone()
        return int(row["value"]) if row else None

    def _ensure_vec_table(self, dims: int) -> None:
        """Create the vec0 KNN table if it doesn't exist for the given dimensions."""
        if self._vec_dims == dims:
            return
        conn = self._conn
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
        conn = self._conn
        # Remove stale FTS entry (INSERT OR REPLACE on chunks changes the rowid).
        existing = conn.execute(
            "SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)
        ).fetchone()
        if existing:
            conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))

        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["path"], chunk.get("language"), chunk.get("node_type"),
            chunk.get("name"), chunk.get("start_line"), chunk.get("end_line"),
            chunk["content"], chunk.get("mtime"), chunk.get("git_hash"),
        ))

        new_row = conn.execute("SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)).fetchone()
        conn.execute(
            "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
            (new_row["rowid"], chunk["content"], chunk.get("name") or "", chunk["path"]),
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

    def upsert_many(self, chunks: list[dict]) -> None:
        import sqlite_vec
        conn = self._conn
        for chunk in chunks:
            existing = conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)
            ).fetchone()
            if existing:
                conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (existing["rowid"],))

            conn.execute("""
                INSERT OR REPLACE INTO chunks
                (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["id"], chunk["path"], chunk.get("language"), chunk.get("node_type"),
                chunk.get("name"), chunk.get("start_line"), chunk.get("end_line"),
                chunk["content"], chunk.get("mtime"), chunk.get("git_hash"),
            ))

            new_row = conn.execute("SELECT rowid FROM chunks WHERE id = ?", (chunk["id"],)).fetchone()
            conn.execute(
                "INSERT INTO chunks_fts(rowid, content, name, path) VALUES (?, ?, ?, ?)",
                (new_row["rowid"], chunk["content"], chunk.get("name") or "", chunk["path"]),
            )

            if chunk.get("embedding"):
                emb = chunk["embedding"]
                dims = len(emb)
                self._ensure_vec_table(dims)
                conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk["id"],))
                conn.execute(
                    "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (chunk["id"], sqlite_vec.serialize_float32(emb)),
                )

        conn.commit()

    def delete_by_path(self, path: str) -> None:
        conn = self._conn
        rows = conn.execute("SELECT id, rowid FROM chunks WHERE path = ?", (path,)).fetchall()
        if not rows:
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

    def get_mtime(self, path: str) -> float | None:
        row = self._conn.execute(
            "SELECT mtime FROM chunks WHERE path = ? LIMIT 1", (path,)
        ).fetchone()
        return row["mtime"] if row else None

    def vector_search(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        import sqlite_vec
        if self._vec_dims is None or self._vec_dims != len(embedding):
            return []
        query_blob = sqlite_vec.serialize_float32(embedding)
        rows = self._conn.execute("""
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
            rows = self._conn.execute("""
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
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        paths = self._conn.execute("SELECT COUNT(DISTINCT path) as cnt FROM chunks").fetchone()
        return {"chunks": row["cnt"], "files": paths["cnt"]}

    def close(self) -> None:
        self._conn.close()
