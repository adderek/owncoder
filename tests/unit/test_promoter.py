"""Tests for session-end fact promotion (agent.memory.promoter)."""
from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.memory.facts_store import FactsStore
from agent.memory.promoter import promote_session_to_notes


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.model_entries = {"default": ModelEntry(base_url="http://x/v1", model="m")}
    return c


def _seed_round(tmp_path, draft="### Intent\nUse Postgres for storage, never MySQL, per team decision."):
    store = FactsStore("sess-1", base_dir=tmp_path)
    store.new_round(from_turn=0, to_turn=5, knowledge_draft=draft,
                     summary="s", q_view="q", facts={})
    return store


def _fake_response(content):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_promotes_notes_on_success(tmp_path, cfg, monkeypatch):
    facts_store = _seed_round(tmp_path)
    reply = _fake_response(
        '{"notes": [{"title": "DB choice", "body": "Use Postgres.", "tags": ["db"]}]}')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=reply)
    client.close = AsyncMock()
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)

    n = promote_session_to_notes("session-abc", cfg, facts_store=facts_store)
    assert n == 1


def test_never_raises_when_endpoint_down(tmp_path, cfg, monkeypatch):
    facts_store = _seed_round(tmp_path)

    def _boom(c, base_url="", api_key=""):
        raise RuntimeError("down")
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)

    assert promote_session_to_notes("session-abc", cfg, facts_store=facts_store) == 0


def test_no_facts_store_short_circuits(cfg):
    assert promote_session_to_notes("session-abc", cfg, facts_store=None) == 0


def test_incognito_skips_saving(tmp_path, cfg, monkeypatch):
    facts_store = _seed_round(tmp_path)
    reply = _fake_response('{"notes": [{"title": "t", "body": "b", "tags": []}]}')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=reply)
    client.close = AsyncMock()
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)

    n = promote_session_to_notes("session-abc", cfg, facts_store=facts_store,
                                 session_mode="incognito")
    assert n == 0
