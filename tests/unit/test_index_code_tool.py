"""Unit tests for tools/index_code."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.tools.index_code.main import index_code, setup, _validate_path


class TestValidatePath:
    def test_dot_returns_none_working_dir(self, tmp_path):
        # dot treated separately in index_code(); _validate_path with dot is valid
        result = _validate_path(".", str(tmp_path))
        assert result == tmp_path.resolve()

    def test_subdir_inside(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        result = _validate_path("src", str(tmp_path))
        assert result == sub.resolve()

    def test_traversal_rejected(self, tmp_path):
        result = _validate_path("../outside", str(tmp_path))
        assert result is None

    def test_abs_path_inside(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        result = _validate_path(str(sub), str(tmp_path))
        assert result == sub.resolve()


class TestIndexCode:
    def _make_config(self, tmp_path):
        cfg = MagicMock()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = ".agent"
        cfg.rag.db_path = str(tmp_path / ".agent" / "index.db")
        return cfg

    def test_not_configured(self):
        import agent.tools.index_code.main as m
        orig = m._config
        m._config = None
        result = index_code(".")
        m._config = orig
        assert "error" in result

    def test_traversal_rejected(self, tmp_path):
        cfg = self._make_config(tmp_path)
        setup(cfg)
        result = index_code("../../etc")
        assert "error" in result
        assert "outside" in result["error"]

    def test_successful_index_whole(self, tmp_path):
        cfg = self._make_config(tmp_path)
        dp = MagicMock()
        dp._embedder = None
        setup(cfg, dp)

        fake_stats = {"indexed": 3, "skipped": 0, "chunks": 10}
        fake_store = MagicMock()

        with patch("agent.cli.index._run_indexing", return_value=fake_stats) as mock_run, \
             patch("agent.rag.store.VectorStore", return_value=fake_store), \
             patch("agent.rag.embedder.Embedder", return_value=MagicMock()):
            result = index_code(".")

        mock_run.assert_called_once()
        assert result["indexed"] == 3
        assert result["chunks"] == 10
        assert result["semantic_search_active"] is True
        assert dp._store is fake_store

    def test_subpath_index(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        cfg = self._make_config(tmp_path)
        dp = MagicMock()
        dp._embedder = MagicMock()
        setup(cfg, dp)

        fake_stats = {"indexed": 1, "skipped": 0, "chunks": 5}
        with patch("agent.cli.index._run_indexing", return_value=fake_stats) as mock_run, \
             patch("agent.rag.store.VectorStore", return_value=MagicMock()):
            result = index_code("src")

        call_kwargs = mock_run.call_args
        assert "src" in str(call_kwargs)
        assert result["path"] == "src"
