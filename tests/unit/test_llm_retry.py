"""core.llm_retry — role-aware failover for one-shot background LLM calls."""
from __future__ import annotations

import types

import pytest

from agent.config.models import ModelEntry
from agent.config.model_probe import clear_availability_cache
from agent.core import llm_retry


class _E(Exception):
    def __init__(self, msg="", retry_after=None):
        super().__init__(msg)
        self.message = msg
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"retry-after": str(retry_after)}})()


def _cfg(entries, roles=None):
    return types.SimpleNamespace(
        model_entries=entries,
        model_roles=roles or {},
        model_pools={},
        agent=types.SimpleNamespace(model_mode="any"),
    )


@pytest.fixture(autouse=True)
def _clear_cooldowns():
    clear_availability_cache()
    yield
    clear_availability_cache()


def test_role_candidates_skips_embeddings_and_disabled():
    entries = {
        "a": ModelEntry(base_url="http://a", model="ma", tier="local"),
        "emb": ModelEntry(base_url="http://e", model="me", dimensions=768, tier="local"),
        "b": ModelEntry(base_url="http://b", model="mb", tier="free"),
    }
    cfg = _cfg(entries)
    names = [n for n, _ in llm_retry.role_candidates(cfg, "verify")]
    assert "emb" not in names
    assert "a" in names and "b" in names


def test_role_candidates_orphan_primary_falls_back_to_role_name():
    # A registry-resolved entry outside config.model_entries (bare test double,
    # or minimal config) must still be usable — keyed by the role name.
    entry = types.SimpleNamespace(base_url="http://x", model="m", dimensions=0)
    cfg = types.SimpleNamespace()  # no model_entries at all
    import agent.config as config_pkg
    orig = config_pkg.make_registry
    config_pkg.make_registry = lambda c: types.SimpleNamespace(role=lambda name: entry)
    try:
        cands = llm_retry.role_candidates(cfg, "verify")
    finally:
        config_pkg.make_registry = orig
    assert cands == [("verify", entry)]


@pytest.mark.asyncio
async def test_call_role_with_failover_switches_on_rate_limit(monkeypatch):
    from openai import RateLimitError

    entries = {
        "a": ModelEntry(base_url="http://a", model="ma", tier="local"),
        "b": ModelEntry(base_url="http://b", model="mb", tier="free"),
    }
    cfg = _cfg(entries, roles={"verify": "a"})

    calls = []

    class _Client:
        def __init__(self, model):
            self.model = model
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create))

        async def _create(self, *, model, **k):
            calls.append(model)
            if model == "ma":
                raise RateLimitError("rate limited: per day quota", response=_fake_response(), body=None)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))])

        async def close(self):
            pass

    def _fake_response():
        return type("R", (), {"headers": {}, "request": None, "status_code": 429})()

    monkeypatch.setattr(
        "agent.core.llm_client.make_llm_client",
        lambda c, base_url="", api_key="": _Client(
            "ma" if base_url == "http://a" else "mb"))

    resp, name, entry = await llm_retry.call_role_with_failover(
        cfg, "verify", messages=[{"role": "user", "content": "hi"}])
    assert name == "b"
    assert calls == ["ma", "mb"]
    assert resp.choices[0].message.content == "ok"
    # The failed entry is now on cooldown.
    from agent.config.model_probe import is_rate_limited
    assert is_rate_limited("http://a", "ma")


@pytest.mark.asyncio
async def test_call_role_with_failover_raises_when_all_candidates_fail(monkeypatch):
    entries = {"a": ModelEntry(base_url="http://a", model="ma", tier="local")}
    cfg = _cfg(entries, roles={"verify": "a"})

    def _boom(c, base_url="", api_key=""):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)

    with pytest.raises(RuntimeError, match="endpoint down"):
        await llm_retry.call_role_with_failover(
            cfg, "verify", messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_open_stream_with_failover_switches_on_connection_error(monkeypatch):
    from openai import APIConnectionError

    entries = {
        "a": ModelEntry(base_url="http://a", model="ma", tier="local"),
        "b": ModelEntry(base_url="http://b", model="mb", tier="free"),
    }
    cfg = _cfg(entries, roles={"background": "a"})

    class _Stream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="ok"))])

    class _Client:
        def __init__(self, model):
            self.model = model
            self.closed = False
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create))

        async def _create(self, *, model, stream, **k):
            if model == "ma":
                raise APIConnectionError(request=None)
            return _Stream()

        async def close(self):
            self.closed = True

    made = []

    def _factory(c, base_url="", api_key=""):
        client = _Client("ma" if base_url == "http://a" else "mb")
        made.append(client)
        return client

    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _factory)

    stream, name, entry, client = await llm_retry.open_stream_with_failover(
        cfg, "background", messages=[{"role": "user", "content": "hi"}])
    assert name == "b"
    tokens = [c.choices[0].delta.content async for c in stream]
    assert tokens == ["ok"]
    # The failed candidate's client was closed; the winning one stays open
    # for the caller to consume, and is the caller's job to close.
    assert made[0].closed is True
    assert made[1].closed is False
    await client.close()
    assert client.closed is True
