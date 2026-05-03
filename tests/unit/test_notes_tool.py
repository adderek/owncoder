"""Unit tests for agent/tools/notes/ — save_note + load_notes_context."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agent.tools.notes.notes import save_note, load_notes_context, setup


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    import agent.tools.notes.notes as mod
    monkeypatch.setattr(mod, "_config", None)
    monkeypatch.setattr(mod, "_embedder", None)
    monkeypatch.setattr(mod, "_notes_store", None)
    yield


def _make_config(tmp_path: Path):
    cfg = MagicMock()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    return cfg


class TestSaveNote:
    def test_save_requires_setup(self):
        result = save_note("t", "b")
        assert "error" in result

    def test_save_and_retrieve(self, tmp_path):
        cfg = _make_config(tmp_path)
        setup(cfg)
        result = save_note("Decision", "Use postgres not sqlite for prod", tags=["db"])
        assert result.get("saved") is True
        assert "id" in result

    def test_empty_title_rejected(self, tmp_path):
        setup(_make_config(tmp_path))
        assert "error" in save_note("", "body")

    def test_empty_body_rejected(self, tmp_path):
        setup(_make_config(tmp_path))
        assert "error" in save_note("title", "")

    def test_with_embedder(self, tmp_path):
        cfg = _make_config(tmp_path)
        embedder = MagicMock()
        embedder.embed_one.return_value = [0.1, 0.2, 0.3]
        setup(cfg, embedder=embedder)
        result = save_note("Note", "some body text")
        assert result.get("saved") is True
        embedder.embed_one.assert_called_once()


class TestLoadNotesContext:
    def test_returns_none_when_no_db(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert load_notes_context(cfg) is None

    def test_returns_none_when_empty(self, tmp_path):
        cfg = _make_config(tmp_path)
        setup(cfg)
        # DB created but no notes
        assert load_notes_context(cfg) is None

    def test_returns_formatted_notes(self, tmp_path):
        cfg = _make_config(tmp_path)
        setup(cfg)
        save_note("Alpha decision", "We chose X over Y because Z", tags=["arch"])
        save_note("Beta preference", "Always use black for formatting")
        ctx = load_notes_context(cfg)
        assert ctx is not None
        assert "Alpha decision" in ctx
        assert "arch" in ctx
        assert "Beta preference" in ctx
        assert "Always use black" in ctx

    def test_notes_in_system_context_heading(self, tmp_path):
        cfg = _make_config(tmp_path)
        setup(cfg)
        save_note("Key fact", "body here")
        ctx = load_notes_context(cfg)
        assert "Saved notes" in ctx
