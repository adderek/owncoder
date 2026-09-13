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
        # Checked once here, not per query. A dimension mismatch is not something
        # to "try anyway": cosine distance is undefined across float[N] and
        # float[M], so the vec0 MATCH can only return noise, and mixing a query
        # vector into a table built by another model silently mis-ranks hits.
        self._mismatch = ""
        if store is not None and config is not None:
            try:
                self._mismatch = store.embedding_mismatch(
                    config.embeddings.model, config.embeddings.dimensions
                )
            except Exception:
                logger.debug("embedding mismatch check failed", exc_info=True)
        if self._mismatch == "dims":
            logger.error(
                "Vector search disabled: the index holds %s-dim vectors but "
                "config asks for %s (%s). Falling back to keyword search — run "
                "`agent index --reembed`.",
                store._vec_dims, config.embeddings.dimensions,
                config.embeddings.model,
            )
        elif self._mismatch == "model":
            logger.warning(
                "Index vectors were produced by %r, not %r — same dimensions, so "
                "search runs, but ranking is degraded. Fix with "
                "`agent index --reembed`.",
                store.get_meta("embedding_model"), config.embeddings.model,
            )

    # ── high-level API ───────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._store is not None

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
                    embedding = self._embedder.embed_query(query)
                except Exception:
                    logger.debug("embed_one failed for query %r", query, exc_info=True)
            if embedding and self._mismatch == "dims":
                # The vectors cannot be compared with the query; blending them in
                # would bury the FTS hits under noise.
                return self._store.fts_search(query, top_k=top_k)
            if embedding and hybrid:
                return self._store.hybrid_search(query, embedding, top_k=top_k)
            elif embedding:
                return self._store.vector_search(embedding, top_k=top_k)
            else:
                return self._store.fts_search(query, top_k=top_k)
        except Exception:
            logger.warning("DataProvider.search failed", exc_info=True)
            return []

    def asm_search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Semantic search over indexed ASM units."""
        if self._asm_store is None or self._embedder is None:
            return []
        if self._mismatch == "dims":
            # Same vectors, same incompatibility — description text is still
            # reachable through the keyword path, this one cannot be blended.
            return []
        try:
            embedding = self._embedder.embed_query(query)
            return self._asm_store.semantic_search(embedding, top_k=top_k)
        except Exception:
            logger.debug("DataProvider.asm_search failed", exc_info=True)
            return []

    def embedding_mismatch(self) -> str:
        """"" | "dims" | "model" — see VectorStore.embedding_mismatch."""
        return self._mismatch

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
