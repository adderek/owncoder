"""Tests for session auto-naming (agent.memory.session_namer)."""
from __future__ import annotations

import asyncio
import types

import pytest

from agent.memory import session as sess
from agent.memory import session_namer as namer


class _FakeMsg:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeMsg(content)]


def _make_fake_client(reply):
    class _Client:
        def __init__(self, *a, **k):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        async def _create(self, *, model, messages, **k):
            return _FakeResp(reply)

        async def close(self):
            pass

    return _Client


@pytest.fixture
def _fake_llm(monkeypatch):
    def _install(reply):
        entry = types.SimpleNamespace(base_url="http://x/v1", api_key="local", model="m")
        reg = types.SimpleNamespace(default=entry, summarizer=entry, background=entry, role=lambda *_a, **_k: entry)
        monkeypatch.setattr("agent.config.make_registry", lambda cfg: reg)
        # _call_llm now routes through core.llm_retry.call_role_with_failover,
        # which builds its client via make_llm_client rather than a bare
        # AsyncOpenAI() — patch that factory instead of faking the openai
        # module, which would also break the real RateLimitError/… imports it uses.
        monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                            lambda cfg, base_url="", api_key="": _make_fake_client(reply)())
    return _install


def test_needs_meta():
    s = sess.Session(id="x")
    assert namer.needs_meta(s) is True
    s.name = "Foo"
    s.description = "bar"
    s.tags = ["t"]
    s.classification = "feature"
    assert namer.needs_meta(s) is False


def test_coerce_meta_sanitizes():
    raw = 'noise {"name":"Add Login Flow","description":"d","tags":["Auth","UI!!","auth"],"classification":"FEATURE","summary":"s"} trailing'
    m = namer._coerce_meta(raw)
    assert m["name"] == "Add Login Flow"
    assert m["short_name"] == "add-login-flow"
    assert m["classification"] == "feature"
    assert m["tags"] == ["auth", "ui"]  # lowercased, deduped, stripped


def test_coerce_meta_unknown_classification_falls_back():
    m = namer._coerce_meta('{"name":"X","classification":"wat"}')
    assert m["classification"] == "other"


def test_coerce_meta_rejects_garbage():
    assert namer._coerce_meta("not json") is None
    assert namer._coerce_meta('{"description":"no name"}') is None


def test_generate_session_meta(_fake_llm):
    _fake_llm('{"name":"Fix Auth","description":"d","tags":["auth"],"classification":"bugfix","summary":"s"}')
    s = sess.Session(id="x")
    msgs = [
        {"role": "user", "content": "auth broken"},
        {"role": "assistant", "content": "fixed token check"},
    ]
    m = asyncio.run(namer.generate_session_meta(s, msgs, object()))
    assert m["name"] == "Fix Auth"
    assert m["classification"] == "bugfix"


def test_generate_skips_short_conversation(_fake_llm):
    _fake_llm('{"name":"X","description":"d","tags":["a"],"classification":"feature","summary":"s"}')
    s = sess.Session(id="x")
    m = asyncio.run(namer.generate_session_meta(s, [{"role": "user", "content": "hi"}], object()))
    assert m is None


def test_generate_never_raises(monkeypatch):
    def _boom(cfg, base_url="", api_key=""):
        raise RuntimeError("down")
    entry = types.SimpleNamespace(base_url="http://x/v1", api_key="local", model="m")
    monkeypatch.setattr("agent.config.make_registry", lambda cfg: types.SimpleNamespace(summarizer=entry, role=lambda *_a, **_k: entry))
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)
    s = sess.Session(id="x")
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert asyncio.run(namer.generate_session_meta(s, msgs, object())) is None


def test_apply_meta_respects_user_name():
    s = sess.Session(id="x", name="My Name")
    meta = {"name": "Auto", "description": "d", "tags": ["t"], "classification": "feature", "summary": "s"}
    changed = namer.apply_meta(s, meta)
    assert changed is True
    assert s.name == "My Name"  # not overwritten
    assert s.description == "d"
    assert s.classification == "feature"


def test_apply_meta_overwrite():
    s = sess.Session(id="x", name="Old")
    namer.apply_meta(s, {"name": "New"}, overwrite=True)
    assert s.name == "New"
