"""Tests for the root Q/A one-line summarizer (agent.summarizer) — W7 failover."""
from __future__ import annotations

import types

import pytest

import agent.summarizer as sumr
from agent.config import Config


class _FakeStream:
    def __init__(self, tokens, exc=None):
        self._tokens = tokens
        self._exc = exc

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        if self._exc:
            raise self._exc
        for t in self._tokens:
            delta = types.SimpleNamespace(content=t, reasoning_content=None)
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


class _FakeClient:
    def __init__(self, tokens=None, raise_on_create=None):
        self._tokens = tokens or []
        self._raise_on_create = raise_on_create
        self.closed = False
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    async def _create(self, *, model, messages, **k):
        if self._raise_on_create:
            raise self._raise_on_create
        return _FakeStream(self._tokens)

    async def close(self):
        self.closed = True


@pytest.fixture
def cfg():
    c = Config()
    c.concurrency.gpu_pool = []  # forces _pick_summarizer_entry -> cpu/background entry
    c.model_entries = {
        "background": types.SimpleNamespace(
            base_url="http://cpu/v1", api_key="local", model="cpu-m",
            ctx_window=8192, dimensions=0,
        ),
    }
    return c


@pytest.fixture(autouse=True)
def _fake_registry(monkeypatch, cfg):
    entry = cfg.model_entries["background"]
    reg = types.SimpleNamespace(background=entry, role=lambda *_a, **_k: entry)
    monkeypatch.setattr("agent.config.make_registry", lambda c: reg)


async def test_primary_success_no_fallback(monkeypatch, cfg):
    client = _FakeClient(tokens=["hello ", "world"])
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)
    out = await sumr._call_llm_one_line(cfg, "sys", "some content")
    assert out == "hello world"
    assert client.closed is True


async def test_primary_failure_falls_back_to_failover(monkeypatch, cfg):
    from openai import APIConnectionError
    failing_client = _FakeClient(raise_on_create=APIConnectionError(request=types.SimpleNamespace()))
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": failing_client)

    fallback_client = _FakeClient(tokens=["fallback ", "answer"])

    async def _fake_open_stream(config, role, *, messages, metrics_role="", local_only=False, **k):
        entry = types.SimpleNamespace(base_url="http://fallback/v1", model="fb-m")
        return _FakeStream(["fallback ", "answer"]), "fallback", entry, fallback_client

    monkeypatch.setattr("agent.core.llm_retry.open_stream_with_failover", _fake_open_stream)

    out = await sumr._call_llm_one_line(cfg, "sys", "some content")
    assert out == "fallback answer"
    assert failing_client.closed is True
    assert fallback_client.closed is True


async def test_airgap_honored_on_fallback(monkeypatch, cfg):
    from openai import APIConnectionError
    failing_client = _FakeClient(raise_on_create=APIConnectionError(request=types.SimpleNamespace()))
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": failing_client)
    monkeypatch.setattr("agent.security.airgap.is_enabled", lambda c: True)

    seen_local_only = {}

    async def _fake_open_stream(config, role, *, messages, metrics_role="", local_only=False, **k):
        seen_local_only["value"] = local_only
        entry = types.SimpleNamespace(base_url="http://fallback/v1", model="fb-m")
        return _FakeStream(["ok"]), "fallback", entry, _FakeClient()

    monkeypatch.setattr("agent.core.llm_retry.open_stream_with_failover", _fake_open_stream)

    await sumr._call_llm_one_line(cfg, "sys", "some content")
    assert seen_local_only["value"] is True


async def test_gpu_slot_semaphore_unchanged_when_primary_succeeds(monkeypatch, cfg):
    """Primary path must not route through open_stream_with_failover at all
    when it succeeds — the GPU-aware pick stays untouched by W7."""
    client = _FakeClient(tokens=["ok"])
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda c, base_url="", api_key="": client)

    called = {"fallback": False}

    async def _fake_open_stream(*a, **k):
        called["fallback"] = True
        raise AssertionError("fallback should not be invoked when primary succeeds")

    monkeypatch.setattr("agent.core.llm_retry.open_stream_with_failover", _fake_open_stream)

    out = await sumr._call_llm_one_line(cfg, "sys", "some content")
    assert out == "ok"
    assert called["fallback"] is False
