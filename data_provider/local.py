"""LocalDataProvider — in-process DataProvider backed by VectorStore + Embedder."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.rag.store import VectorStore
    from agent.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class LocalDataProvider:
    """Wraps raw RAG objects; satisfies DataProviderProtocol.

    `store`, `embedder`, `asm_store` may be None (index not built).
    """

    def __init__(
        self,
        store: "VectorStore | None" = None,
        embedder: "Embedder | None" = None,
        asm_store=None,
        config=None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._asm_store = asm_store
        self._config = config

    # ── high-level API ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Embed query and search; falls back to FTS if no embedder."""
        if self._store is None:
            return []
        try:
            cfg = self._config
            hybrid = cfg.rag.hybrid if cfg else True
            embedding = None
            if self._embedder:
                try:
                    embedding = self._embedder.embed_one(query)
                except Exception:
                    logger.debug("embed_one failed for query %r", query, exc_info=True)
            if embedding and hybrid:
                return self._store.hybrid_search(query, embedding, top_k=top_k)
            elif embedding:
                return self._store.vector_search(embedding, top_k=top_k)
            else:
                return self._store.fts_search(query, top_k=top_k)
        except Exception:
            logger.warning("DataProvider.search failed", exc_info=True)
            return []

    def stats(self) -> dict[str, Any]:
        if self._store is None:
            return {"files": 0, "chunks": 0}
        try:
            return self._store.stats()
        except Exception:
            return {"files": 0, "chunks": 0}

    # ── escape hatches ───────────────────────────────────────────────────────

    def get_store(self) -> Any:
        return self._store

    def get_embedder(self) -> Any:
        return self._embedder

    def get_asm_store(self) -> Any:
        return self._asm_store
