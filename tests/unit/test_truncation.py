"""Regression tests for output-truncation detection in core/turn.py.

When the server cuts a response off at the output-token cap (finish_reason
"length") and the cut breaks a tool call's JSON arguments, the turn must tell
the model to retry smaller instead of running the tool with ``{}`` and
surfacing a misleading "missing field" error.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.config import Config
import agent.core.turn as turn_mod
from agent.core.turn import run_turn
from agent.core.turn_batch import has_broken_arguments


def _broken_tool_call(name: str = "write_file"):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(
            name=name,
            arguments='{"path": "a.py", "content": "abc',  # truncated mid-string
        ),
    )


def _truncated_response():
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[_broken_tool_call()]),
        finish_reason="length",
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
        return resp if resp is not None else _stop_response()


class _StubClient:
    def __init__(self, *responses):
        self.chat = SimpleNamespace(completions=_StubCompletions(list(responses)))


# --- pure helper: has_broken_arguments ---

def test_has_broken_arguments_true_for_truncated_json():
    assert has_broken_arguments([_broken_tool_call()]) is True


def test_has_broken_arguments_false_for_valid_json():
    tc = SimpleNamespace(id="c", function=SimpleNamespace(name="w", arguments='{"path": "a.py"}'))
    assert has_broken_arguments([tc]) is False


def test_has_broken_arguments_false_for_empty_and_dict():
    empty = SimpleNamespace(id="c", function=SimpleNamespace(name="w", arguments=""))
    parsed = SimpleNamespace(id="c", function=SimpleNamespace(name="w", arguments={"path": "a.py"}))
    assert has_broken_arguments([empty, parsed]) is False


# --- turn-level: truncated call is nudged, not executed ---

async def test_truncated_tool_call_is_nudged_not_executed(monkeypatch):
    cfg = Config()
    cfg.llm.narration_fallback = False

    executed = []

    async def _spy_execute(tc, config=None):
        executed.append(tc.function.name)
        return json.dumps({"ok": True})

    monkeypatch.setattr(turn_mod, "execute_tool", _spy_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(_truncated_response(), _stop_response("retried fine"))
    response, messages = await run_turn(
        [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}],
        cfg, client,
    )

    # The broken call must never reach execution.
    assert executed == []
    # The model is told what actually went wrong, not handed a validation error.
    nudge = [
        m for m in messages
        if m.get("role") == "user" and "output-token limit" in m.get("content", "")
    ]
    assert nudge, "expected a truncation nudge message"
    assert "retried fine" in response
