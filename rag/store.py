from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import RAGConfig


def _pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


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
        """)

        # Create vec0 table — dimensions must match config
        dims = self._cfg.__class__.__name__  # just used as a marker
        # We store embeddings in a separate table with JSON since vec0 requires
        # compile-time fixed dimensions; use a portable approach
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()

    def upsert(self, chunk: dict) -> None:
        conn = self._conn
        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, path, language, node_type, name, start_line, end_line, content, mtime, git_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk["id"], chunk["path"], chunk.get("language"), chunk.get("node_type"),
            chunk.get("name"), chunk.get("start_line"), chunk.get("end_line"),
            chunk["content"], chunk.get("mtime"), chunk.get("git_hash"),
        ))

        if chunk.get("embedding"):
            packed = _pack_embedding(chunk["embedding"])
            conn.execute("""
                INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding)
                VALUES (?, ?)
            """, (chunk["id"], packed))

        conn.commit()

    def upsert_many(self, chunks: list[dict]) -> None:
        for chunk in chunks:
            self.upsert(chunk)

    def delete_by_path(self, path: str) -> None:
        conn = self._conn
        rows = conn.execute("SELECT id FROM chunks WHERE path = ?", (path,)).fetchall()
        ids = [r["id"] for r in rows]
        if ids:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            conn.commit()

    def get_mtime(self, path: str) -> float | None:
        row = self._conn.execute(
            "SELECT mtime FROM chunks WHERE path = ? LIMIT 1", (path,)
        ).fetchone()
        return row["mtime"] if row else None

    def vector_search(self, embedding: list[float], top_k: int = 20) -> list[dict]:
        packed = _pack_embedding(embedding)
        dims = len(embedding)
        rows = self._conn.execute("""
            SELECT c.id, c.path, c.language, c.node_type, c.name,
                   c.start_line, c.end_line, c.content,
                   ce.embedding
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
        """).fetchall()

        scored = []
        query_vec = embedding
        for row in rows:
            stored = struct.unpack(f"{dims}f", row["embedding"][:dims * 4])
            # cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, stored))
            norm_q = sum(a * a for a in query_vec) ** 0.5
            norm_s = sum(b * b for b in stored) ** 0.5
            if norm_q > 0 and norm_s > 0:
                score = dot / (norm_q * norm_s)
            else:
                score = 0.0
            scored.append((score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **d} for s, d in scored[:top_k]]

    def fts_search(self, query: str, top_k: int = 20) -> list[dict]:
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
        return [dict(r) for r in rows]

    def hybrid_search(self, query: str, embedding: list[float], top_k: int = 8) -> list[dict]:
        vec_results = self.vector_search(embedding, top_k=20)
        fts_results = self.fts_search(query, top_k=20)

        # Normalize scores
        def normalize(results: list[dict], key: str) -> dict[str, float]:
            if not results:
                return {}
            scores = [r[key] for r in results]
            # FTS5 bm25 returns negative values — negate so higher is better
            if key == "score" and scores and scores[0] < 0:
                scores = [-s for s in scores]
            mn, mx = min(scores), max(scores)
            if mx == mn:
                return {r["id"]: 1.0 for r in results}
            return {r["id"]: (s - mn) / (mx - mn) for r, s in zip(results, scores)}

        vec_norm = normalize(vec_results, "score")
        fts_norm: dict[str, float] = {}
        if fts_results:
            raw_fts = [{"id": r["id"], "score": -r["score"]} for r in fts_results]
            fts_norm = normalize(raw_fts, "score")

        all_ids: dict[str, dict] = {}
        for r in vec_results:
            all_ids[r["id"]] = r
        for r in fts_results:
            all_ids.setdefault(r["id"], r)

        combined = []
        for cid, chunk in all_ids.items():
            v = vec_norm.get(cid, 0.0)
            b = fts_norm.get(cid, 0.0)
            combined_score = 0.6 * v + 0.4 * b
            combined.append((combined_score, chunk))

        combined.sort(key=lambda x: x[0], reverse=True)
        return [{"combined_score": s, **d} for s, d in combined[:top_k]]

    def stats(self) -> dict:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM chunks").fetchone()
        paths = self._conn.execute("SELECT COUNT(DISTINCT path) as cnt FROM chunks").fetchone()
        return {"chunks": row["cnt"], "files": paths["cnt"]}

    def close(self) -> None:
        self._conn.close()
