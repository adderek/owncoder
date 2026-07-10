"""Text-file indexing: chunker text fallback + indexer walk coverage.

Non-source files (.example, .template, Makefile, unknown extensions that sniff
as text) must be chunked and picked up by the index walk; binaries, oversized
files, dotfiles and lockfiles must not.
"""
from __future__ import annotations

import pytest

from agent.config.models import RAGConfig
from agent.rag.chunker import chunk_file, is_text_candidate, BINARY_EXTENSIONS
from agent.rag.indexer import _wanted_file
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
