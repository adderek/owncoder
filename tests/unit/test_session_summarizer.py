"""Tests for the session Q/A summarizer (agent.memory.session_summarizer)."""
from __future__ import annotations

import types

import pytest

from agent.memory import session_summarizer as sumr


class _FakeStream:
    def __init__(self, tokens):
        self._tokens = tokens

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for t in self._tokens:
            delta = types.SimpleNamespace(content=t)
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class _FakeClient:
    def __init__(self, tokens):
        self._tokens = tokens
        self.closed = False
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, **k):
        return _FakeStream(self._tokens)

    async def close(self):
        self.closed = True


@pytest.fixture
def _fake_llm(monkeypatch):
    def _install(tokens):
        entry = types.SimpleNamespace(base_url="http://x/v1", api_key="local", model="m")
        reg = types.SimpleNamespace(default=entry, summarizer=entry, background=entry,
                                    role=lambda *_a, **_k: entry)
        monkeypatch.setattr("agent.config.make_registry", lambda cfg: reg)
        client = _FakeClient(tokens)
        monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                            lambda cfg, base_url="", api_key="": client)
        return client
    return _install


async def test_generate_writes_and_loads(tmp_path, _fake_llm):
    _fake_llm(["Hello ", "world"])
    entries = [(1, {"content": "q1"}, {"content": "a1"})]
    out = await sumr.generate(tmp_path, entries, "q", object())
    assert out == "Hello world"
    stored = sumr.load_stored(tmp_path, "q")
    assert stored["content"] == "Hello world"
    assert stored["summarized_up_to_turn"] == 1


async def test_generate_closes_client(tmp_path, _fake_llm):
    client = _fake_llm(["ok"])
    entries = [(1, {"content": "q1"}, {"content": "a1"})]
    await sumr.generate(tmp_path, entries, "q", object())
    assert client.closed is True


async def test_generate_no_entries_short_circuits(tmp_path):
    assert await sumr.generate(tmp_path, [], "q", object()) == ""
