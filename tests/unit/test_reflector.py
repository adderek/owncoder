"""Tests for session-end behavioral-rule reflection (agent.memory.reflector)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.memory.facts_store import FactsStore
from agent.memory.reflector import reflect_session


@pytest.fixture
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.model_entries = {"default": ModelEntry(base_url="http://x/v1", model="m")}
    return c


def _seed_round(tmp_path, draft="### Mistakes\nUser corrected: use uv not pip, every time."):
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


def test_saves_rule_on_success(tmp_path, cfg, monkeypatch):
    facts_store = _seed_round(tmp_path)
    reply = _fake_response(
        '{"rules": [{"rule": "Use uv, never pip", "category": "correction", "confidence": 0.9}]}')
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=reply)
    client.close = AsyncMock()
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)

    n = reflect_session("session-abc", cfg, facts_store=facts_store)
    assert n == 1


def test_never_raises_when_endpoint_down(tmp_path, cfg, monkeypatch):
    facts_store = _seed_round(tmp_path)

    def _boom(c, base_url="", api_key=""):
        raise RuntimeError("down")
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)

    assert reflect_session("session-abc", cfg, facts_store=facts_store) == 0


def test_no_facts_store_and_no_failures_short_circuits(cfg):
    assert reflect_session("session-abc", cfg, facts_store=None) == 0
