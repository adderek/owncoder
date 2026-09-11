"""Transport-error handling in run_turn.

A dropped socket or a request timeout is not evidence the endpoint is down: the
client may have given up mid-generation (router reload, closed keep-alive, a
switch on the LAN) while the server kept producing tokens. Observed with a model
streaming 91 t/s that got cooled down and failed over to a paid endpoint on a
single closed connection, with no retry anywhere — the SDK's own retries are
disabled (llm_client sets max_retries=0) on the assumption run_turn covers them,
and run_turn only covered 429, stalls and tool-parse errors.

Two rules under test:
  * retry the same request first (``llm.transport_retries``);
  * a cooldown is only worth setting when something else can take over — when
    nothing can, drop it and give the only endpoint left one more chance.
"""
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

import agent.config.model_probe as model_probe
from agent.config import Config
from agent.core import turn_errors
from agent.core.turn import NoUsableModelError, run_turn


def _cfg() -> Config:
    cfg = Config()
    cfg.llm.max_iterations = 50
    cfg.confidence_guard.enabled = False
    # The narration fallback re-prompts a first-iteration reply that carries no
    # tool call (a "justify" round trip), which would add a model call of its own
    # and blur the call counts asserted here. These tests are about transport
    # handling, so turn it off.
    cfg.llm.narration_fallback = False
    model_probe.clear_availability_cache()
    return cfg


def _conn_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "http://test/v1/chat/completions"))


class _StubChoice:
    def __init__(self, content=None, finish_reason="stop"):
        self.message = SimpleNamespace(content=content, tool_calls=None)
        self.finish_reason = finish_reason


class _StubResponse:
    def __init__(self, choice):
        self.choices = [choice]
        self.usage = None


class _ScriptedCompletions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def create(self, **kw):
        self.calls += 1
        item = self._script.pop(0) if self._script else _StubResponse(_StubChoice("done"))
        if isinstance(item, Exception):
            raise item
        return item


class _ScriptedClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(script))


def _no_sleep(monkeypatch):
    import agent.core.turn as turn_mod

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(turn_mod.asyncio, "sleep", _instant)


def _stub_tools(monkeypatch):
    import agent.core.turn as turn_mod

    monkeypatch.setattr(turn_mod, "execute_tool", None)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])


_MESSAGES = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]


async def test_transport_error_retries_the_same_endpoint(monkeypatch):
    cfg = _cfg()
    cfg.llm.transport_retries = 1
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch)

    client = _ScriptedClient([_conn_error(), _StubResponse(_StubChoice("done"))])
    response, _ = await run_turn(_MESSAGES, cfg, client)
    assert response == "done"
    assert client.chat.completions.calls == 2


async def test_transport_retries_zero_goes_straight_to_the_failure_path(monkeypatch):
    cfg = _cfg()
    cfg.llm.transport_retries = 0
    cfg.failover.enabled = False
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch)

    # One call fails outright; the only further call is the no-target
    # self-recovery retry, not a transport retry.
    client = _ScriptedClient([_conn_error(), _StubResponse(_StubChoice("done"))])
    response, _ = await run_turn(_MESSAGES, cfg, client)
    assert response == "done"
    assert client.chat.completions.calls == 2


async def test_no_failover_target_clears_the_cooldown_it_just_set(monkeypatch):
    cfg = _cfg()
    cfg.failover.enabled = True
    monkeypatch.setattr(turn_errors, "try_failover", lambda c, **kw: None)
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch)

    client = _ScriptedClient([_conn_error(), _StubResponse(_StubChoice("done"))])
    response, _ = await run_turn(_MESSAGES, cfg, client)
    assert response == "done"
    # The failing endpoint is the only one left, so the cooldown must not stick:
    # it would guarantee the next turn dies too.
    assert model_probe.is_rate_limited(cfg.llm.base_url, cfg.llm.model) is False


async def test_a_failover_target_keeps_the_cooldown(monkeypatch):
    cfg = _cfg()
    cfg.failover.enabled = True
    # No transport retry: the scripted failure must reach the failover path.
    cfg.llm.transport_retries = 0
    switched = {}

    def _failover(c, **kw):
        switched["to"] = "fallback-model"
        c.llm.model = "fallback-model"
        return _ScriptedClient([_StubResponse(_StubChoice("done"))])

    monkeypatch.setattr(turn_errors, "try_failover", _failover)
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch)

    failed = (cfg.llm.base_url, cfg.llm.model)
    client = _ScriptedClient([_conn_error()])
    response, _ = await run_turn(_MESSAGES, cfg, client)
    assert response == "done"
    assert switched["to"] == "fallback-model"
    # Something did take over, so the cooldown on the failed pair stays: the
    # tier ladder must not come back to it while it is still unproven.
    assert model_probe.is_rate_limited(*failed) is True


async def test_unrecoverable_transport_failures_are_bounded(monkeypatch):
    cfg = _cfg()
    cfg.llm.transport_retries = 1
    cfg.failover.enabled = False
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch)

    client = _ScriptedClient([_conn_error() for _ in range(10)])
    with pytest.raises(NoUsableModelError) as ei:
        await run_turn(_MESSAGES, cfg, client)
    assert isinstance(ei.value.cause, APIConnectionError)
    # initial + one transport retry + one self-recovery retry, then surface.
    assert client.chat.completions.calls == 3


def test_turn_errors_exposes_the_cooldown_clear(monkeypatch):
    cfg = _cfg()
    cfg.llm.base_url = "http://probe.test/v1"
    cfg.llm.model = "m1"
    turn_errors.mark_endpoint_cooldown(cfg)
    assert model_probe.is_rate_limited("http://probe.test/v1", "m1") is True
    turn_errors.clear_endpoint_cooldown(cfg)
    assert model_probe.is_rate_limited("http://probe.test/v1", "m1") is False
