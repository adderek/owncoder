"""Startup guard for an index built by a different embedding model.

A mismatch must never be resolved silently: the background index-update thread
would otherwise write the new model's vectors into the old table and blend two
vector spaces in one index.
"""
from __future__ import annotations

import builtins
import io

import pytest
from rich.console import Console

from agent.config.models import EmbeddingsConfig, RAGConfig
from agent.rag.indexer import index_directory
from agent.rag.mismatch import (
    ABORTED,
    FROZEN,
    OK,
    REEMBEDDED,
    backup_index,
    handle_embedding_mismatch,
)


class FakeEmbedder:
    def __init__(self, model: str, dims: int, fail: bool = False, base_url: str = "http://localhost:8080/v1"):
        self._cfg = EmbeddingsConfig(model=model, base_url=base_url, dimensions=dims)
        self._fail = fail

    def embed(self, texts):
        if self._fail:
            raise RuntimeError("connection refused")
        return [[0.1] * self._cfg.dimensions for _ in texts]


class _Config:
    def __init__(self, tmp_path, model, dims):
        self.embeddings = EmbeddingsConfig(
            model=model, base_url="http://localhost:8080/v1", dimensions=dims
        )
        self.rag = RAGConfig(db_path=str(tmp_path / "index.db"), chunk_min_tokens=1, chunk_max_tokens=100)


def _console():
    return Console(file=io.StringIO(), width=200)


def _seed(tmp_path, store, model="Qwen3-Embedding-0.6B-Q8_0", dims=4):
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    (root / "a.md").write_text("# hello\n\nsome searchable prose here\n")
    index_directory(
        str(root), store, FakeEmbedder(model, dims),
        RAGConfig(chunk_min_tokens=1, chunk_max_tokens=100), force=True,
    )


def _store(tmp_path):
    from agent.rag.store import VectorStore
    return VectorStore(RAGConfig(db_path=str(tmp_path / "index.db")))


def test_no_mismatch_is_a_noop(tmp_path):
    store = _store(tmp_path)
    config = _Config(tmp_path, "Qwen3-Embedding-0.6B-Q8_0", 4)
    _seed(tmp_path, store)
    out, action = handle_embedding_mismatch(store, config, _console(), interactive=False)
    assert action == OK
    assert out is store
    assert store.embedding_mismatch(config.embeddings.model, config.embeddings.dimensions) == ""


def test_noninteractive_runs_frozen(tmp_path):
    """A prompt that nobody can answer must not re-embed or blend anything."""
    store = _store(tmp_path)
    _seed(tmp_path, store)  # stamped Qwen3, 4 dims
    config = _Config(tmp_path, "bge-m3", 8)
    before = store.all_chunk_texts()
    out, action = handle_embedding_mismatch(store, config, _console(), interactive=False)
    assert action == FROZEN
    assert out is store
    assert store.embedding_mismatch("bge-m3", 8) == "dims"
    assert store.all_chunk_texts() == before


@pytest.mark.parametrize("answer", ["1", "stop", "exit"])
def test_choice_stop_closes_the_store(tmp_path, monkeypatch, answer):
    store = _store(tmp_path)
    _seed(tmp_path, store)
    config = _Config(tmp_path, "bge-m3", 8)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: answer)
    _, action = handle_embedding_mismatch(store, config, _console(), interactive=True)
    assert action == ABORTED


@pytest.mark.parametrize("answer", ["2", ""])
def test_choice_frozen_leaves_vectors_intact(tmp_path, monkeypatch, answer):
    store = _store(tmp_path)
    _seed(tmp_path, store)
    config = _Config(tmp_path, "bge-m3", 8)
    before = store.all_chunk_texts()
    monkeypatch.setattr(builtins, "input", lambda *a, **k: answer)
    _, action = handle_embedding_mismatch(store, config, _console(), interactive=True)
    assert action == FROZEN
    assert store.all_chunk_texts() == before


def test_choice_reembed_backs_up_and_restamps(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _seed(tmp_path, store)
    config = _Config(tmp_path, "bge-m3", 8)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "3")
    monkeypatch.setattr("agent.rag.embedder.Embedder", lambda cfg: FakeEmbedder(cfg.model, cfg.dimensions))

    out, action = handle_embedding_mismatch(store, config, _console(), interactive=True)

    assert action == REEMBEDDED
    assert out.embedding_mismatch("bge-m3", 8) == ""
    backups = list(tmp_path.glob("index.db.bak-*"))
    assert backups, "a re-embed must leave a backup of the old index"


def test_reembed_that_cannot_reach_embedder_keeps_index_and_backup(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _seed(tmp_path, store)
    config = _Config(tmp_path, "bge-m3", 8)
    before = store.all_chunk_texts()
    monkeypatch.setattr(builtins, "input", lambda *a, **k: "3")
    monkeypatch.setattr("agent.rag.embedder.Embedder", lambda cfg: FakeEmbedder(cfg.model, cfg.dimensions, fail=True))

    out, action = handle_embedding_mismatch(store, config, _console(), interactive=True)

    assert action == FROZEN
    assert out.all_chunk_texts() == before
    assert out.vector_search([0.1] * 4, top_k=5), "a dead embedder must not empty the index"
    assert list(tmp_path.glob("index.db.bak-*"))


def test_backup_index_copies_wal_sidecar(tmp_path):
    db = tmp_path / "index.db"
    db.write_text("main")
    (tmp_path / "index.db-wal").write_text("wal")
    bak = backup_index(str(db))
    assert bak is not None
    assert bak.read_text() == "main"
    assert (bak.parent / (bak.name + "-wal")).read_text() == "wal"


def test_backup_index_missing_file_returns_none(tmp_path):
    assert backup_index(str(tmp_path / "nope.db")) is None
