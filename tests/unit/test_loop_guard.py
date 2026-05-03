import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.core.loop_detector import LoopDetector
from agent.core.turn import run_turn
from agent.config import Config


def _fake_tool_call(name: str, args: dict | None = None):
    args = args or {}
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_signature_stable_for_equivalent_args():
    a = LoopDetector.signature("read_file", '{"path": "x.py", "n": 1}')
    b = LoopDetector.signature("read_file", '{"n": 1, "path": "x.py"}')
    assert a == b


def test_signature_differs_by_args():
    a = LoopDetector.signature("read_file", '{"path": "x.py"}')
    b = LoopDetector.signature("read_file", '{"path": "y.py"}')
    assert a != b


def test_triggers_after_threshold():
    d = LoopDetector(window=10, threshold=3)
    sig = LoopDetector.signature("grep", '{"q":"foo"}')
    assert d.triggered(sig, d.observe(sig)) is False
    assert d.triggered(sig, d.observe(sig)) is False
    assert d.triggered(sig, d.observe(sig)) is True


def test_acknowledge_silences_signature():
    d = LoopDetector(window=10, threshold=2)
    sig = LoopDetector.signature("ls", "{}")
    d.observe(sig); d.observe(sig)
    assert d.triggered(sig, 2) is True
    d.acknowledge(sig)
    assert d.triggered(sig, 5) is False


def test_per_tool_threshold_overrides_default():
    d = LoopDetector(window=20, threshold=3, per_tool_threshold={"list_files": 5})
    sig = LoopDetector.signature("list_files", '{"path": "."}')
    # First 4 observations should not trigger (override = 5)
    for _ in range(4):
        assert d.triggered(sig, d.observe(sig)) is False
    # 5th observation triggers
    assert d.triggered(sig, d.observe(sig)) is True


def test_per_tool_threshold_only_applies_to_named_tool():
    d = LoopDetector(window=20, threshold=3, per_tool_threshold={"list_files": 10})
    other = LoopDetector.signature("read_file", '{"path": "x.py"}')
    # read_file still uses default threshold=3
    d.observe(other); d.observe(other)
    assert d.triggered(other, 3) is True


def test_window_evicts_old_signatures():
    d = LoopDetector(window=3, threshold=2)
    sig = LoopDetector.signature("a", "{}")
    other = LoopDetector.signature("b", "{}")
    d.observe(sig)                      # buf: [a]
    for _ in range(3):
        d.observe(other)                # buf: [other,other,other]; sig evicted
    assert d.observe(sig) == 1          # only this latest occurrence remains


class _StubChoice:
    def __init__(self, tool_calls):
        self.message = SimpleNamespace(content=None, tool_calls=tool_calls)
        self.finish_reason = "tool_calls"


class _StubResponse:
    def __init__(self, tool_calls):
        self.choices = [_StubChoice(tool_calls)]
        self.usage = None


class _StubCompletions:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    async def create(self, **kw):
        # Always return the same tool call to force a loop
        return _StubResponse([_fake_tool_call("read_file", {"path": "x.py"}) for _ in range(1)])


class _StubClient:
    def __init__(self, tool_calls):
        self.chat = SimpleNamespace(completions=_StubCompletions(tool_calls))


@pytest.mark.asyncio
async def test_run_turn_stops_on_loop_with_no_callback(monkeypatch):
    cfg = Config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 50  # ensure loop guard, not iter cap, is what stops us

    # Stub the tool execution to avoid hitting the real registry
    async def _fake_execute(tc, config=None):
        return json.dumps({"ok": True})

    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(None)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, out_messages = await run_turn(messages, cfg, client)
    assert "loop guard" in response
    # Should have stopped before hitting max_iterations
    assert sum(1 for m in out_messages if m.get("role") == "tool") < cfg.llm.max_iterations


@pytest.mark.asyncio
async def test_run_turn_continues_when_callback_returns_true(monkeypatch):
    cfg = Config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 5  # bound the test

    async def _fake_execute(tc, config=None):
        return json.dumps({"ok": True})

    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(None)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    calls = {"n": 0}

    def _allow(summary, count):
        calls["n"] += 1
        return True

    response, _ = await run_turn(messages, cfg, client, on_loop_detected=_allow)
    # Callback fired at least once; we still terminated via max_iterations note
    assert calls["n"] >= 1
    assert "iteration limit" in response or "loop guard" not in response
