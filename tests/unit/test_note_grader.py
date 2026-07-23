"""Tests for injected-note relevance grading (agent.memory.note_grader)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.memory.note_grader import grade_notes


@pytest.fixture
def cfg():
    c = Config()
    c.model_entries = {"default": ModelEntry(base_url="http://x/v1", model="m")}
    return c


def _fake_response(content):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _notes():
    return [{"id": "n1", "title": "A", "body": "uses postgres"},
            {"id": "n2", "title": "B", "body": "uses redis"}]


@pytest.mark.asyncio
async def test_returns_ids_the_model_names(monkeypatch, cfg):
    reply = _fake_response('["n1"]')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=reply)
    client.close = AsyncMock()
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)

    ids = await grade_notes("q", _notes(), "answer used postgres", cfg)
    assert ids == ["n1"]


@pytest.mark.asyncio
async def test_falls_back_to_heuristic_on_error(monkeypatch, cfg):
    def _boom(c, base_url="", api_key=""):
        raise RuntimeError("down")
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)

    ids = await grade_notes("q", _notes(), "answer mentions postgres explicitly", cfg)
    # Heuristic fallback never raises; result is whatever overlap it finds.
    assert isinstance(ids, list)


@pytest.mark.asyncio
async def test_empty_notes_short_circuits():
    assert await grade_notes("q", [], "answer", Config()) == []
