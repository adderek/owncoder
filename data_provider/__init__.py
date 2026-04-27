"""agent.data_provider — RAG/data access abstraction layer.

Phase 1: LocalDataProvider wraps VectorStore + Embedder in-process.
Later: transport-backed DataProvider for remote RAG, versioning, parallelism.
"""
from .protocol import DataProviderProtocol
from .local import LocalDataProvider

__all__ = ["DataProviderProtocol", "LocalDataProvider"]
