"""Re-embed failure modes found on 2026-11-08.

* The Embedder signals a failed text with an empty vector instead of raising.
  A probe of empty vectors sized the table to float[0] and dropped every stored
  vector (2026-09-21, the whole bge-m3 index).
* The stamp was written only at the very end of a pass, so an interrupted pass
  left the new model's vectors under the old model's name; the next startup
  then offered a full re-embed of an index that needed none.
* Helpers (temporary GPU servers) must share the work but never write vectors
  from a different vector space.
"""
from __future__ import annotations

import io
import os
import stat

import pytest
from rich.console import Console

from agent.config.models import EmbeddingsConfig, RAGConfig
from agent.rag.indexer import index_directory, progress_text, reembed_all

MODEL = "Qwen3-Embedding-0.6B-Q8_0"


class Fake:
    """Deterministic embedder: vector depends on text, so 'same model' agrees."""

    def __init__(self, model=MODEL, dims=4, *, empty_calls=(), flip=False, base_url="http://a/v1"):
        self._cfg = EmbeddingsConfig(model=model, base_url=base_url, dimensions=dims)
        self.endpoint = base_url
        self.calls = 0
        self.texts = 0
        self._empty_calls = set(empty_calls)
        self._flip = flip

    def _vec(self, text):
        h = sum(map(ord, text)) % 97 + 1
        v = [float(h), 1.0, 2.0, 3.0][: self._cfg.dimensions]
        return [-x for x in v] if self._flip else v

    def embed(self, texts):
        self.calls += 1
        if self.calls in self._empty_calls:
            return [[] for _ in texts]
        self.texts += len(texts)
        return [self._vec(t) for t in texts]


def _store(tmp_path):
    from agent.rag.store import VectorStore
    return VectorStore(RAGConfig(db_path=str(tmp_path / "index.db")))


def _seed(tmp_path, store, n_files=1, embedder=None):
    root = tmp_path / "src"
    root.mkdir(exist_ok=True)
    for i in range(80 * n_files):  # one chunk per file → several 32-chunk batches
        (root / f"f{i}.md").write_text(f"# title {i}\n\nsearchable prose number {i} here\n")
    index_directory(str(root), store, embedder or Fake(),
                    RAGConfig(chunk_min_tokens=1, chunk_max_tokens=20), force=True)


def _cfg():
    return RAGConfig(chunk_min_tokens=1, chunk_max_tokens=20)


class TestWriteEmbeddings:
    def test_skips_empty_vectors(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store)
        cid = store.all_chunk_texts()[0][0]
        assert store.write_embeddings([(cid, [])]) == 0
        assert store.has_vectors()

    def test_width_mismatch_raises_instead_of_dropping(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store)
        cid = store.all_chunk_texts()[0][0]
        with pytest.raises(ValueError):
            store.write_embeddings([(cid, [0.1] * 8)])
        assert store.vector_coverage()["embedded"] > 0


class TestReembedEmptyVectors:
    def test_empty_probe_aborts_and_keeps_vectors(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store)
        before = store.vector_coverage()["embedded"]
        result = reembed_all(store, Fake("other", empty_calls={1}), _cfg())
        assert result["aborted"] is True
        assert store.vector_coverage()["embedded"] == before
        assert store.get_meta("embedding_model") == MODEL

    def test_empty_batch_mid_run_is_retried_not_destructive(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store)
        total = store.stats()["chunks"]
        assert total > 64, "needs several batches"
        result = reembed_all(store, Fake("new", empty_calls={2}), _cfg())
        assert result["aborted"] is False
        assert result["failed"] == 0
        assert store.vector_coverage()["embedded"] == total

    def test_stamp_committed_before_the_bulk(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store)
        seen = []

        class Spy(Fake):
            def embed(self, texts):
                if self.calls >= 1:
                    seen.append(store.get_meta("embedding_model"))
                return super().embed(texts)

        reembed_all(store, Spy("new"), _cfg())
        assert seen and set(seen) == {"new"}, \
            "an interrupted run must already carry the new stamp"


class TestHelpers:
    def test_helper_shares_the_work(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store, n_files=3)
        prim, helper = Fake("new"), Fake("new", base_url="http://b/v1")
        result = reembed_all(store, prim, _cfg(), extra_embedders=[helper])
        assert result["embedded"] == store.stats()["chunks"]
        assert helper.texts > 4, "helper must take batches, not just the probe"
        assert len(result["workers"]) == 2

    def test_helper_from_another_space_is_dropped(self, tmp_path):
        store = _store(tmp_path)
        _seed(tmp_path, store, n_files=2)
        helper = Fake("new", flip=True, base_url="http://b/v1")
        result = reembed_all(store, Fake("new"), _cfg(), extra_embedders=[helper])
        assert result["dropped"] and "differ" in result["dropped"][0]
        assert helper.calls == 1  # the probe only
        assert result["embedded"] == store.stats()["chunks"]


class TestIndexDirectoryStamp:
    def test_empty_marker_vectors_do_not_stamp(self, tmp_path):
        store = _store(tmp_path)
        store.set_meta("embedding_model", "old")
        _seed(tmp_path, store, embedder=Fake("new", empty_calls=set(range(1, 100))))
        assert store.get_meta("embedding_model") == "old"
        assert not store.has_vectors()

    def test_first_vector_into_empty_table_stamps_immediately(self, tmp_path):
        store = _store(tmp_path)
        store.set_meta("embedding_model", "stale")
        seen = []

        class Spy(Fake):
            def embed(self, texts):
                if self.calls >= 1:
                    seen.append(store.get_meta("embedding_model"))
                return super().embed(texts)

        _seed(tmp_path, store, embedder=Spy(MODEL))
        assert seen and seen[-1] == MODEL


class TestStaleStamp:
    def test_matching_vectors_correct_the_stamp(self, tmp_path, monkeypatch):
        from agent.rag.mismatch import OK, handle_embedding_mismatch

        store = _store(tmp_path)
        _seed(tmp_path, store)
        store.set_meta("embedding_model", "bge-m3-q8_0.gguf")

        class _Config:
            embeddings = EmbeddingsConfig(model=MODEL, base_url="http://a/v1", dimensions=4)
            rag = RAGConfig(db_path=str(tmp_path / "index.db"))

        monkeypatch.setattr("agent.rag.embedder.Embedder", lambda cfg: Fake(cfg.model))
        out, action = handle_embedding_mismatch(
            store, _Config(), Console(file=io.StringIO()), interactive=False)
        assert action == OK
        assert out.get_meta("embedding_model") == MODEL

    def test_different_vectors_still_report_mismatch(self, tmp_path, monkeypatch):
        from agent.rag.mismatch import FROZEN, handle_embedding_mismatch

        store = _store(tmp_path)
        _seed(tmp_path, store)

        class _Config:
            embeddings = EmbeddingsConfig(model="bge-m3", base_url="http://a/v1", dimensions=4)
            rag = RAGConfig(db_path=str(tmp_path / "index.db"))

        monkeypatch.setattr("agent.rag.embedder.Embedder",
                            lambda cfg: Fake(cfg.model, flip=True))
        _, action = handle_embedding_mismatch(
            store, _Config(), Console(file=io.StringIO()), interactive=False)
        assert action == FROZEN
        assert store.get_meta("embedding_model") == MODEL


def test_progress_text():
    assert progress_text(4192, 10257, 1310.0) == "4192/10257 (41%) 3.2/s, ETA 32m"
    assert progress_text(0, 10, 0.0) == "0/10 (0%)"
    assert progress_text(10, 10, 5.0, failed=2).endswith(", 2 failed")


def test_parse_helpers():
    from agent.rag.embed_server import parse_helpers
    assert parse_helpers("gpu0,gpu1 cpu") == ["gpu0", "gpu1", "cpu"]
    assert parse_helpers("") == []
    with pytest.raises(ValueError):
        parse_helpers("tpu")


def test_start_helpers_env_contract(tmp_path, monkeypatch):
    """Launcher gets the device word plus EMB_PORT, and EMB_GPU for gpuN."""
    from agent.rag import embed_server

    monkeypatch.setattr(embed_server, "_CACHE_DIR", tmp_path)
    out = tmp_path / "env.txt"
    script = tmp_path / "emb.sh"
    script.write_text(f'#!/bin/sh\necho "$1 $EMB_PORT ${{EMB_GPU:-none}}" >> {out}\nexec sleep 30\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    class C:
        rag = RAGConfig(embed_server_command=str(script))

    probed = []

    def probe(url):
        probed.append(url)
        return out.exists() and len(out.read_text().splitlines()) >= 2

    hs = embed_server.start_helpers(C(), ["gpu1", "cpu"], probe=probe, wait_s=10)
    try:
        assert len(hs) == 2
        lines = sorted(out.read_text().split("\n")[:2])
        assert lines[0].startswith("cpu ") and lines[0].endswith(" none")
        assert lines[1].startswith("gpu ") and lines[1].endswith(" 1")
        ports = {h.base_url.rsplit(":", 1)[1].split("/")[0] for h in hs}
        assert len(ports) == 2
    finally:
        embed_server.stop_helpers(hs)
    for h in hs:
        assert h.proc.poll() is not None
