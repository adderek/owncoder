"""Tests for agent.ipc — transport, messages, controller, agent_worker."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.ipc.local import LocalTransport
from agent.ipc.messages import (
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    PhaseEvent,
    UsageEvent,
    ReasoningEvent,
    ContextSizeEvent,
    ProgressEvent,
    LoopDetectedEvent,
    TruncationEvent,
    TurnDoneEvent,
    ErrorEvent,
)
from agent.ipc.controller import run_turn_ipc


# ---------------------------------------------------------------------------
# LocalTransport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_transport_send_receive():
    t = LocalTransport()
    t.send_nowait(TokenEvent("hello"))
    t.send_nowait(TokenEvent("world"))
    await t.close()

    received = []
    async for evt in t.receive():
        received.append(evt)

    assert received == [TokenEvent("hello"), TokenEvent("world")]


@pytest.mark.asyncio
async def test_local_transport_close_ends_iteration():
    t = LocalTransport()
    await t.close()
    received = [evt async for evt in t.receive()]
    assert received == []


@pytest.mark.asyncio
async def test_local_transport_multiple_types():
    t = LocalTransport()
    t.send_nowait(PhaseEvent("start"))
    t.send_nowait(TurnDoneEvent(response="ok", messages=[]))
    await t.close()

    events = [evt async for evt in t.receive()]
    assert isinstance(events[0], PhaseEvent)
    assert isinstance(events[1], TurnDoneEvent)


# ---------------------------------------------------------------------------
# LoopDetectedEvent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_detected_resolve_true():
    evt = LoopDetectedEvent(
        summary="read_file×3",
        max_count=3,
        _decision=asyncio.get_event_loop().create_future(),
    )
    await evt.resolve(True)
    assert await evt.wait() is True


@pytest.mark.asyncio
async def test_loop_detected_resolve_false():
    evt = LoopDetectedEvent(
        summary="read_file×3",
        max_count=3,
        _decision=asyncio.get_event_loop().create_future(),
    )
    await evt.resolve(False)
    assert await evt.wait() is False


@pytest.mark.asyncio
async def test_loop_detected_no_decision_returns_false():
    evt = LoopDetectedEvent(summary="x", max_count=1)
    assert await evt.wait() is False


@pytest.mark.asyncio
async def test_loop_detected_resolve_idempotent():
    evt = LoopDetectedEvent(
        summary="x",
        max_count=1,
        _decision=asyncio.get_event_loop().create_future(),
    )
    await evt.resolve(True)
    await evt.resolve(False)  # second call must not raise
    assert await evt.wait() is True


# ---------------------------------------------------------------------------
# run_turn_ipc — stub helpers (mirroring test_loop_guard.py pattern)
# ---------------------------------------------------------------------------


def _fake_tool_call(name: str, args: dict | None = None):
    args = args or {}
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _StubCompletions:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def create(self, **kw):
        resp = next(self._responses, None)
        if resp is None:
            choice = SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None),
                finish_reason="stop",
            )
            return SimpleNamespace(choices=[choice], usage=None)
        return resp


def _stop_response(content: str = "done"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice], usage=None)


def _stub_client(*responses):
    completions = _StubCompletions(responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


# ---------------------------------------------------------------------------
# run_turn_ipc — basic event collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_ipc_collects_tokens(monkeypatch):
    """Tokens from streaming reach on_token callback via IPC layer."""
    from agent.config import Config
    import agent.core.turn as turn_mod

    cfg = Config()

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_token = kwargs.get("on_token")
        if on_token:
            on_token("hello")
            on_token(" world")
        return "hello world", messages

    monkeypatch.setattr(turn_mod, "run_turn", _fake_run_turn)
    # agent_worker imports run_turn at call time so patch the module it reads from
    import agent.ipc.agent_worker as aw_mod
    # Re-patch inside agent_worker's own namespace
    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    tokens: list[str] = []
    result, _ = await run_turn_ipc(
        messages=[{"role": "user", "content": "hi"}],
        config=cfg,
        client=_stub_client(),
        on_token=lambda t: tokens.append(t),
    )

    assert result == "hello world"
    assert tokens == ["hello", " world"]


@pytest.mark.asyncio
async def test_run_turn_ipc_collects_phases(monkeypatch):
    from agent.config import Config

    cfg = Config()

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_phase = kwargs.get("on_phase")
        if on_phase:
            on_phase("start", "iter 1")
            on_phase("done", "")
        return "ok", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    phases: list[tuple] = []
    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_phase=lambda l, d="": phases.append((l, d)),
    )

    assert phases == [("start", "iter 1"), ("done", "")]


@pytest.mark.asyncio
async def test_run_turn_ipc_propagates_exception(monkeypatch):
    from agent.config import Config

    cfg = Config()

    async def _failing_run_turn(messages, config, client, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr("agent.core.turn.run_turn", _failing_run_turn)

    with pytest.raises(ValueError, match="boom"):
        await run_turn_ipc(
            messages=[{"role": "user", "content": "x"}],
            config=cfg,
            client=_stub_client(),
        )


@pytest.mark.asyncio
async def test_run_turn_ipc_tool_events(monkeypatch):
    from agent.config import Config

    cfg = Config()

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_tool_call = kwargs.get("on_tool_call")
        on_tool_result = kwargs.get("on_tool_result")
        if on_tool_call:
            on_tool_call("read_file", '{"path":"x.py"}')
        if on_tool_result:
            on_tool_result("read_file", True)
        return "done", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    calls: list[tuple] = []
    results: list[tuple] = []

    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_tool_call=lambda n, a: calls.append((n, a)),
        on_tool_result=lambda n, ok: results.append((n, ok)),
    )

    assert calls == [("read_file", '{"path":"x.py"}')]
    assert results == [("read_file", True)]


@pytest.mark.asyncio
async def test_run_turn_ipc_usage_forwarded(monkeypatch):
    from agent.config import Config

    cfg = Config()

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_usage = kwargs.get("on_usage")
        if on_usage:
            on_usage({"input_tokens": 10, "output_tokens": 5})
        return "ok", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    usages: list[dict] = []
    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_usage=lambda u: usages.append(u),
    )

    assert usages == [{"input_tokens": 10, "output_tokens": 5}]


# ---------------------------------------------------------------------------
# run_turn_ipc — LoopDetectedEvent bidirectional
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_ipc_loop_detected_stop(monkeypatch):
    """on_loop_detected returning False stops the worker correctly."""
    from agent.config import Config

    cfg = Config()
    decisions: list[bool] = []

    async def _looping_run_turn(messages, config, client, **kwargs):
        on_loop_detected = kwargs.get("on_loop_detected")
        decision = False
        if on_loop_detected:
            res = on_loop_detected("read_file×3", 3)
            if asyncio.iscoroutine(res):
                res = await res
            decision = bool(res)
        decisions.append(decision)
        return "stopped", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _looping_run_turn)

    result, _ = await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_loop_detected=lambda s, c: False,
    )

    assert result == "stopped"
    assert decisions == [False]


@pytest.mark.asyncio
async def test_run_turn_ipc_loop_detected_continue(monkeypatch):
    """on_loop_detected returning True lets worker continue."""
    from agent.config import Config

    cfg = Config()
    decisions: list[bool] = []

    async def _looping_run_turn(messages, config, client, **kwargs):
        on_loop_detected = kwargs.get("on_loop_detected")
        decision = False
        if on_loop_detected:
            res = on_loop_detected("x×3", 3)
            if asyncio.iscoroutine(res):
                res = await res
            decision = bool(res)
        decisions.append(decision)
        return "continued", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _looping_run_turn)

    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_loop_detected=lambda s, c: True,
    )

    assert decisions == [True]


@pytest.mark.asyncio
async def test_run_turn_ipc_loop_detected_no_callback(monkeypatch):
    """No on_loop_detected callback → decision defaults to False."""
    from agent.config import Config

    cfg = Config()
    decisions: list[bool] = []

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_loop_detected = kwargs.get("on_loop_detected")
        decision = False
        if on_loop_detected:
            res = on_loop_detected("x×3", 3)
            if asyncio.iscoroutine(res):
                res = await res
            decision = bool(res)
        decisions.append(decision)
        return "done", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_loop_detected=None,
    )

    assert decisions == [False]


# ---------------------------------------------------------------------------
# run_turn_ipc — truncation event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_turn_ipc_truncation_forwarded(monkeypatch):
    from agent.config import Config

    cfg = Config()
    truncated = []

    async def _fake_run_turn(messages, config, client, **kwargs):
        on_truncation = kwargs.get("on_truncation")
        if on_truncation:
            on_truncation()
        return "ok", messages

    monkeypatch.setattr("agent.core.turn.run_turn", _fake_run_turn)

    await run_turn_ipc(
        messages=[{"role": "user", "content": "x"}],
        config=cfg,
        client=_stub_client(),
        on_truncation=lambda: truncated.append(True),
    )

    assert truncated == [True]


def test_ipc_signatures_stay_in_sync_with_run_turn():
    """run_turn_ipc is a drop-in for run_turn; agent_worker.run forwards to it.

    Guards against signature drift — the class of bug where a run_turn parameter
    is added but never threaded through the IPC boundary (e.g. a callback that
    silently never fires in parallel mode).
    """
    import inspect
    from agent.core.turn import run_turn
    from agent.ipc.controller import run_turn_ipc
    from agent.ipc.agent_worker import run as worker_run

    def public(fn):
        return {p for p in inspect.signature(fn).parameters if not p.startswith("_")}

    rt = public(run_turn)
    ipc = public(run_turn_ipc)
    assert ipc == rt, (
        f"run_turn_ipc out of sync with run_turn; "
        f"missing={sorted(rt - ipc)} extra={sorted(ipc - rt)}"
    )

    # worker.run takes the same data params plus `send`; it builds the on_* callbacks
    # itself, so only its non-callback params must exist on run_turn.
    worker_data = {p for p in public(worker_run) if not p.startswith("on_")} - {"send"}
    assert worker_data <= rt, f"agent_worker.run has params unknown to run_turn: {sorted(worker_data - rt)}"
