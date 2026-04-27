"""DataProviderProtocol — interface for RAG/data access.

Phase 1: LocalDataProvider wraps VectorStore + Embedder in-process.
Later: transport-backed DataProvider enables remote RAG, data versioning, sharding.

The `get_*` methods are Phase-1 escape hatches exposing raw objects to tools
that haven't been migrated to the high-level search() API yet.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataProviderProtocol(Protocol):
    """Data access interface between controller and storage backends."""

    def is_available(self) -> bool:
        """True when an index exists and queries will return results."""
        ...

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Semantic + keyword search over source code index. Handles embedding internally."""
        ...

    def asm_search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Semantic search over indexed ASM units. Returns [] when no ASM index."""
        ...

    def stats(self) -> dict[str, Any]:
        """Index statistics (file count, chunk count, etc.)."""
        ...

    # ── Phase-1 escape hatches ───────────────────────────────────────────────
    # Expose raw objects for consumers that need direct access.
    # get_store:     Agent.__init__ bootstrap; CLI close(); agent.store UI display.
    # get_embedder:  analyze_asm analysis pipeline (write path, not query path).
    # get_asm_store: analyze_asm analysis pipeline (write path, not query path).

    def get_store(self) -> Any:
        """Returns underlying VectorStore or None."""
        ...

    def get_embedder(self) -> Any:
        """Returns underlying Embedder or None. Used by analyze_asm pipeline."""
        ...

    def get_asm_store(self) -> Any:
        """Returns underlying AsmStore or None. Used by analyze_asm pipeline."""
        ...
