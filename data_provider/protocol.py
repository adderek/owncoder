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

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Semantic + keyword search. Handles embedding internally."""
        ...

    def stats(self) -> dict[str, Any]:
        """Index statistics (file count, chunk count, etc.)."""
        ...

    # ── Phase-1 escape hatches ───────────────────────────────────────────────
    # Return raw objects for tools not yet migrated to the high-level API.
    # These will be removed once all consumers use search() / purpose-built methods.

    def get_store(self) -> Any:
        """Returns underlying VectorStore or None."""
        ...

    def get_embedder(self) -> Any:
        """Returns underlying Embedder or None."""
        ...

    def get_asm_store(self) -> Any:
        """Returns underlying AsmStore or None."""
        ...
