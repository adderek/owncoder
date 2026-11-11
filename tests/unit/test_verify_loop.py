"""Tests for the post-edit verify loop in core/turn.py (VerifyConfig)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.config import Config
import agent.core.turn as turn_mod
from agent.core import markers
from agent.core.turn import run_turn


def _fake_tool_call(name: str, args: dict | None = None):
    args = args or {}
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _tool_call_response(name: str, args: dict | None = None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[_fake_tool_call(name, args)]),
        finish_reason="tool_calls",
    )
    return SimpleNamespace(choices=[choice], usage=None)


def _stop_response(content: str = "done"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice], usage=None)


class _StubCompletions:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def create(self, **kw):
        resp = next(self._responses, None)
        if resp is None:
            return _stop_response("done")
        return resp


class _StubClient:
    def __init__(self, *responses):
        self.chat = SimpleNamespace(completions=_StubCompletions(list(responses)))


def _base_messages():
    return [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]


async def _fake_execute_ok(tc, config=None):
    return json.dumps({"ok": True})


def _fake_verify_sequence(results):
    """Returns a (fn, calls) pair; fn returns the next (rc, output) each call."""
    calls = {"n": 0}

    def _fn(command, cwd, timeout_s):
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[idx]

    return _fn, calls


async def test_verify_disabled_no_subprocess_run(monkeypatch):
    cfg = Config()
    assert cfg.verify.enabled is False  # default

    def _boom(*a, **k):
        raise AssertionError("verify must not run when disabled")
    monkeypatch.setattr(turn_mod, "_run_verify_command", _boom)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(
        _tool_call_response("edit_file", {"path": "a.py"}),
        _stop_response("finished"),
    )
    response, _ = await run_turn(_base_messages(), cfg, client)
    assert "finished" in response


async def test_no_edits_verify_not_run(monkeypatch):
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "true"
    cfg.llm.narration_fallback = False  # isolate verify behavior from the unrelated nudge loop

    def _boom(command, cwd, timeout_s):
        raise AssertionError("verify should not run without a successful mutating tool call")
    monkeypatch.setattr(turn_mod, "_run_verify_command", _boom)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(_stop_response("nothing changed"))
    response, _ = await run_turn(_base_messages(), cfg, client)
    assert "nothing changed" in response


async def test_read_only_tool_does_not_set_dirty(monkeypatch):
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "true"

    def _boom(command, cwd, timeout_s):
        raise AssertionError("verify should not run after only read-only tool calls")
    monkeypatch.setattr(turn_mod, "_run_verify_command", _boom)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(
        _tool_call_response("read_file", {"path": "a.py"}),
        _stop_response("looked around"),
    )
    response, _ = await run_turn(_base_messages(), cfg, client)
    assert "looked around" in response


async def test_verify_fails_then_passes_continues_turn(monkeypatch):
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "pytest"
    cfg.verify.max_attempts = 2

    fake_fn, calls = _fake_verify_sequence([(1, "AssertionError: boom"), (0, "ok")])
    monkeypatch.setattr(turn_mod, "_run_verify_command", fake_fn)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(
        _tool_call_response("edit_file", {"path": "a.py"}),
        _stop_response("first attempt"),
        _stop_response("second attempt"),
    )
    response, out_messages = await run_turn(_base_messages(), cfg, client)

    assert calls["n"] == 2
    assert "second attempt" in response
    # The failing verify output must have been injected back as a user message.
    verify_notes = [m for m in out_messages if m.get("role") == "user" and "[verify]" in (m.get("content") or "")]
    assert len(verify_notes) == 1
    assert "boom" in verify_notes[0]["content"]
    assert "exit 1" in verify_notes[0]["content"]


async def test_verify_fails_max_attempts_ends_turn(monkeypatch):
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "pytest"
    cfg.verify.max_attempts = 1

    fake_fn, calls = _fake_verify_sequence([(1, "still broken")])
    monkeypatch.setattr(turn_mod, "_run_verify_command", fake_fn)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(
        _tool_call_response("edit_file", {"path": "a.py"}),
        _stop_response("done"),
    )
    response, _ = await run_turn(_base_messages(), cfg, client)

    assert calls["n"] == 1
    assert "verify" in response.lower()
    assert "failing" in response.lower()


async def test_verify_timeout_treated_as_failure(monkeypatch):
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "sleep 999"
    cfg.verify.timeout_s = 1

    # Patch the sandbox runner so _run_verify_command's real timeout branch runs.
    from agent.security import runner

    def _fake_run_shell(command, **kw):
        return runner.RunResult(-9, "partial", "", 1000, "fake", timed_out=True)
    monkeypatch.setattr(runner, "run_shell", _fake_run_shell)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    cfg.verify.max_attempts = 1
    client = _StubClient(
        _tool_call_response("edit_file", {"path": "a.py"}),
        _stop_response("done"),
    )
    response, _ = await run_turn(_base_messages(), cfg, client)
    assert "verify" in response.lower()
    assert "failing" in response.lower()


async def test_a_failing_verify_is_announced_while_it_happens(monkeypatch):
    """The note goes into history, so the live view has to hear about it too.

    Without this the turn stored a failure the watching user never saw: the
    answer read "done", and the failing suite only surfaced when the session
    was reloaded.
    """
    cfg = Config()
    cfg.verify.enabled = True
    cfg.verify.command = "pytest"
    cfg.verify.max_attempts = 2

    fake_fn, _ = _fake_verify_sequence([(1, "AssertionError: boom"), (0, "ok")])
    monkeypatch.setattr(turn_mod, "_run_verify_command", fake_fn)
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    announced: list[tuple[str, str]] = []
    client = _StubClient(
        _tool_call_response("edit_file", {"path": "a.py"}),
        _stop_response("first attempt"),
        _stop_response("second attempt"),
    )
    _, out_messages = await run_turn(
        _base_messages(), cfg, client,
        on_injected_message=lambda kind, text: announced.append((kind, text)))

    stored = [m["content"] for m in out_messages
              if m.get("role") == "user" and "[verify]" in (m.get("content") or "")]
    # Stored copies carry the harness source marker; the announcement the UI
    # gets is the same text without it (core/markers.py).
    assert [text for _, text in announced] == [markers.strip(c) for c in stored]
    assert all(markers.contains(c) for c in stored)
    # Labelled, so no view has to guess whether the user typed it.
    assert [kind for kind, _ in announced] == ["verify"]
    assert all(m.get("_injected_kind") == "verify" for m in out_messages
               if "[verify]" in (m.get("content") or ""))
    assert "boom" in announced[0][1]


def test_verify_runs_in_the_sandbox(monkeypatch):
    """[verify] executes agent-writable tests; it must never run on the host."""
    from agent.security import runner
    seen = {}

    def _fake_run_shell(command, **kw):
        seen.update(kw, command=command)
        return runner.RunResult(0, "ok", "", 1, "fake")
    monkeypatch.setattr(runner, "run_shell", _fake_run_shell)
    assert turn_mod._run_verify_command("pytest -q", "/proj", 180) == (0, "ok")
    assert seen["command"] == "pytest -q" and seen["cwd"] == "/proj"
    assert seen["timeout"] == 180 and seen["cpu_seconds"] == 180


def test_verify_fails_closed_without_sandbox(monkeypatch):
    from agent.security import runner

    def _unavailable(command, **kw):
        raise runner.SandboxUnavailable("no backend")
    monkeypatch.setattr(runner, "run_shell", _unavailable)
    rc, out = turn_mod._run_verify_command("pytest", "/proj", 5)
    assert rc != 0 and "no backend" in out
