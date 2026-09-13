"""Text-file indexing: chunker text fallback + indexer walk coverage.

Non-source files (.example, .template, Makefile, unknown extensions that sniff
as text) must be chunked and picked up by the index walk; binaries, oversized
files, dotfiles and lockfiles must not.
"""
from __future__ import annotations

import pytest

from agent.config.models import EmbeddingsConfig, RAGConfig
from agent.rag.chunker import chunk_file, is_text_candidate, BINARY_EXTENSIONS
from agent.rag.indexer import _wanted_file, index_directory
from pathlib import Path


@pytest.fixture
def cfg():
    return RAGConfig(chunk_min_tokens=1, chunk_max_tokens=100)


class TestIsTextCandidate:
    def test_example_and_template(self, tmp_path):
        f = tmp_path / "agent.toml.example"
        f.write_text("network = 'off'\n")
        assert is_text_candidate(f)
        t = tmp_path / "page.template"
        t.write_text("<html>{{x}}</html>\n")
        assert is_text_candidate(t)

    def test_no_extension_makefile(self, tmp_path):
        f = tmp_path / "Makefile"
        f.write_text("all:\n\techo hi\n")
        assert is_text_candidate(f)

    def test_code_ext_not_text(self, tmp_path):
        f = tmp_path / "x.py"
        f.write_text("print(1)\n")
        assert not is_text_candidate(f)  # handled by the code path instead

    def test_binary_ext_rejected(self, tmp_path):
        f = tmp_path / "x.png"
        f.write_bytes(b"\x89PNG")
        assert not is_text_candidate(f)

    def test_nul_sniff_rejects_binary(self, tmp_path):
        f = tmp_path / "blob.weird"
        f.write_bytes(b"text\x00binary")
        assert not is_text_candidate(f)

    def test_oversized_rejected(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100)
        assert not is_text_candidate(f, max_bytes=10)

    def test_empty_rejected(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert not is_text_candidate(f)

    def test_dotfile_rejected(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("KEY=1\n")
        assert not is_text_candidate(f)

    def test_lockfile_rejected(self, tmp_path):
        assert ".lock" in BINARY_EXTENSIONS


class TestChunkFileText:
    def test_example_file_chunked(self, tmp_path, cfg):
        f = tmp_path / "conf.example"
        f.write_text("some configuration example content here\n" * 5)
        chunks = chunk_file(str(f), cfg)
        assert chunks
        assert chunks[0]["language"] == "text"
        assert "configuration example" in chunks[0]["content"]

    def test_text_indexing_disabled(self, tmp_path, cfg):
        cfg.index_text_files = False
        f = tmp_path / "conf.example"
        f.write_text("content\n" * 5)
        assert chunk_file(str(f), cfg) == []

    def test_python_still_python(self, tmp_path, cfg):
        f = tmp_path / "m.py"
        f.write_text("def f():\n    return 1\n")
        chunks = chunk_file(str(f), cfg)
        assert chunks
        assert chunks[0]["language"] == "python"


class TestWantedFile:
    def test_code_and_text_wanted_by_default(self, tmp_path, cfg):
        py = tmp_path / "a.py"; py.write_text("x=1\n")
        ex = tmp_path / "b.example"; ex.write_text("y=2\n")
        assert _wanted_file(py, None, True, cfg)
        assert _wanted_file(ex, None, True, cfg)

    def test_text_excluded_by_language_filter(self, tmp_path, cfg):
        ex = tmp_path / "b.example"; ex.write_text("y=2\n")
        assert not _wanted_file(ex, {".py"}, False, cfg)

    def test_text_included_when_filter_names_text(self, tmp_path, cfg):
        ex = tmp_path / "b.example"; ex.write_text("y=2\n")
        assert _wanted_file(ex, {".py"}, True, cfg)

    def test_config_flag_off_excludes_text(self, tmp_path, cfg):
        cfg.index_text_files = False
        ex = tmp_path / "b.example"; ex.write_text("y=2\n")
        assert not _wanted_file(ex, None, True, cfg)

    def test_binary_never_wanted(self, tmp_path, cfg):
        png = tmp_path / "i.png"; png.write_bytes(b"\x89PNG")
        assert not _wanted_file(png, None, True, cfg)


class TestMissingGrammarFallback:
    """Optional grammars (the `lang` extra) must degrade loudly, not silently."""

    @pytest.fixture(autouse=True)
    def _clean_cache(self):
        from agent.rag import chunker
        chunker._parser_cache.clear()
        chunker._warned_missing_grammar.clear()
        yield
        chunker._parser_cache.clear()
        chunker._warned_missing_grammar.clear()

    def test_missing_optional_grammar_warns_once_with_extra_hint(self, monkeypatch, caplog):
        import importlib
        from agent.rag import chunker

        def boom(name, *a, **k):
            raise ImportError(name)

        monkeypatch.setattr(importlib, "import_module", boom)
        with caplog.at_level("WARNING", logger="agent.rag.chunker"):
            assert chunker._get_parser("kotlin") is None
            assert chunker._get_parser("kotlin") is None
        msgs = [r.getMessage() for r in caplog.records]
        assert len(msgs) == 1
        assert "tree_sitter_kotlin" in msgs[0]
        assert "[lang]" in msgs[0]


class TestEmbeddingModelStamp:
    """The stamp must describe vectors that actually exist.

    Regression: index_directory stamped the configured model unconditionally, so
    a session whose embedder endpoint was down relabelled an index it never
    re-embedded — leaving _meta claiming one model over another model's vectors,
    and the "same dims" warning text asserting something never checked.
    """

    class _Embedder:
        def __init__(self, model, dims, fail=False):
            self._cfg = EmbeddingsConfig(
                model=model, base_url="http://localhost:8080/v1", dimensions=dims
            )
            self._fail = fail

        def embed(self, texts):
            if self._fail:
                raise RuntimeError("connection refused")
            return [[0.1] * self._cfg.dimensions for _ in texts]

    def _store(self, tmp_path):
        from agent.rag.store import VectorStore
        return VectorStore(RAGConfig(db_path=str(tmp_path / "index.db")))

    def _tree(self, tmp_path, name="src"):
        root = tmp_path / name
        root.mkdir()
        (root / "a.md").write_text("# hello\n\nsome searchable prose here\n")
        return root

    def _cfg(self):
        return RAGConfig(chunk_min_tokens=1, chunk_max_tokens=100)

    def test_stamp_committed_when_embeddings_produced(self, tmp_path):
        store = self._store(tmp_path)
        index_directory(
            str(self._tree(tmp_path)), store,
            self._Embedder("Qwen3-Embedding-0.6B-Q8_0", 4), self._cfg(), force=True,
        )
        assert store.get_meta("embedding_model") == "Qwen3-Embedding-0.6B-Q8_0"

    def test_stamp_left_alone_when_embedder_down(self, tmp_path, caplog):
        store = self._store(tmp_path)
        store.set_meta("embedding_model", "Qwen3-Embedding-0.6B-Q8_0")
        with caplog.at_level("WARNING", logger="agent.rag.indexer"):
            index_directory(
                str(self._tree(tmp_path)), store,
                self._Embedder("nomic-embed-text", 768, fail=True), self._cfg(), force=True,
            )
        assert store.get_meta("embedding_model") == "Qwen3-Embedding-0.6B-Q8_0"
        assert any("without vectors" in r.getMessage() for r in caplog.records)

    def test_dim_change_is_reported_as_incompatible(self, tmp_path, caplog):
        store = self._store(tmp_path)
        index_directory(
            str(self._tree(tmp_path, "one")), store,
            self._Embedder("Qwen3-Embedding-0.6B-Q8_0", 4), self._cfg(), force=True,
        )
        with caplog.at_level("ERROR", logger="agent.rag.indexer"):
            index_directory(
                str(self._tree(tmp_path, "two")), store,
                self._Embedder("bge-m3", 8), self._cfg(), force=True,
            )
        errs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert errs, "a dimension change must not be reported at warning level"
        assert "incompatible" in errs[0]


class TestReembedAll:
    """`agent index --reembed` — vectors only.

    Swapping the embedding model used to mean `agent init --force`: re-read the
    tree, re-chunk it, re-run the summarization ladder — all to produce vectors
    for text that had not changed. reembed_all takes the chunks already in the
    index as its input.
    """

    class _Embedder:
        def __init__(self, model, dims, fail=False):
            self._cfg = EmbeddingsConfig(
                model=model, base_url="http://localhost:8080/v1", dimensions=dims
            )
            self._fail = fail

        def embed(self, texts):
            if self._fail:
                raise RuntimeError("connection refused")
            return [[0.1] * self._cfg.dimensions for _ in texts]

    def _store(self, tmp_path):
        from agent.rag.store import VectorStore
        return VectorStore(RAGConfig(db_path=str(tmp_path / "index.db")))

    def _cfg(self):
        return RAGConfig(chunk_min_tokens=1, chunk_max_tokens=100)

    def _seed(self, tmp_path, store, model="Qwen3-Embedding-0.6B-Q8_0", dims=4):
        root = tmp_path / "src"
        root.mkdir(exist_ok=True)
        (root / "a.md").write_text("# hello\n\nsome searchable prose here\n")
        index_directory(
            str(root), store, self._Embedder(model, dims), self._cfg(), force=True,
        )

    def test_replaces_vectors_and_restamps(self, tmp_path):
        from agent.rag.indexer import reembed_all
        store = self._store(tmp_path)
        self._seed(tmp_path, store)
        before = store.all_chunk_texts()
        assert before

        result = reembed_all(store, self._Embedder("bge-m3", 4), self._cfg())

        assert result["aborted"] is False
        assert result["embedded"] == len(before)
        assert store.get_meta("embedding_model") == "bge-m3"
        assert store.stats()["chunks"] == len(before)

    def test_leaves_chunk_text_alone(self, tmp_path):
        from agent.rag.indexer import reembed_all
        store = self._store(tmp_path)
        self._seed(tmp_path, store)
        before = store.all_chunk_texts()

        reembed_all(store, self._Embedder("bge-m3", 4), self._cfg())

        assert store.all_chunk_texts() == before

    def test_aborts_and_keeps_index_when_embedder_down(self, tmp_path):
        from agent.rag.indexer import reembed_all
        store = self._store(tmp_path)
        self._seed(tmp_path, store)
        before = store.all_chunk_texts()

        result = reembed_all(
            store, self._Embedder("nomic-embed-text", 4, fail=True), self._cfg(),
        )

        assert result["aborted"] is True
        assert result["embedded"] == 0
        assert store.get_meta("embedding_model") == "Qwen3-Embedding-0.6B-Q8_0"
        assert store.vector_search([0.1] * 4, top_k=5), \
            "a dead embedder must not empty a working index"
        assert store.all_chunk_texts() == before

    def test_is_the_remedy_for_a_dimension_change(self, tmp_path):
        from agent.rag.indexer import reembed_all
        store = self._store(tmp_path)
        self._seed(tmp_path, store)
        assert store.embedding_mismatch("bge-m3", 8) == "dims"

        reembed_all(store, self._Embedder("bge-m3", 8), self._cfg())

        assert store.embedding_mismatch("bge-m3", 8) == ""
        assert store.vector_search([0.1] * 8, top_k=5)

    def test_mismatch_reports_a_model_swap_at_the_same_width(self, tmp_path):
        store = self._store(tmp_path)
        self._seed(tmp_path, store)
        assert store.embedding_mismatch("bge-m3", 4) == "model"
        assert store.embedding_mismatch("Qwen3-Embedding-0.6B-Q8_0", 4) == ""

    def test_mismatch_is_silent_on_an_empty_index(self, tmp_path):
        store = self._store(tmp_path)
        assert store.embedding_mismatch("bge-m3", 8) == ""
        assert store.embedding_mismatch("", 0) == ""


def test_reembed_flag_is_registered():
    """`agent index --reembed` must reach the dispatch in cli/main.py."""
    from agent.cli.main import build_parser
    args = build_parser().parse_args(["index", "--reembed"])
    assert args.reembed is True
    assert args.update is False

