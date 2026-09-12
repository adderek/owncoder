"""`no_tool_needed` — the typed form of the NO_TOOL_NEEDED: sentinel.

It exists so every legitimate response class is a tool call. Prose-with-
justification was the one class reachable only as text, so sending
`tool_choice: "required"` would have silently deleted it and taken away the
model's ability to explain instead of acting.

Measured on ornith-1.0-35B before this landed: asked a question needing no tool,
`tool_choice: "auto"` returned 420 characters of prose and no call;
`tool_choice: "required"` returned the SAME 420 characters AND this call. So the
grammar permits content alongside a call and no explanation is lost.
"""
from __future__ import annotations

import json
import types


def _call(name: str, args: dict):
    return types.SimpleNamespace(
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_it_is_not_a_turn_signal():
    """No >>> token, absent from the signal set, invisible to the meta-loop —
    whose parser matches a closed keyword set it must not be added to."""
    from agent.tools.turn_signals import (NO_TOOL_TOOL_NAME, SIGNAL_TOOL_NAMES,
                                          build_signal_line)
    assert NO_TOOL_TOOL_NAME not in SIGNAL_TOOL_NAMES
    assert build_signal_line(NO_TOOL_TOOL_NAME, "anything") == ""


def test_meta_loop_parser_ignores_it():
    """parse_signal returns (text, signal). A NO_TOOL_NEEDED line must parse to
    no signal and be left in the text, because the meta-loop's keyword set is
    closed and this is not one of its directives."""
    from agent.core.turn_signals import parse_signal
    text, signal = parse_signal(">>>NO_TOOL_NEEDED: whatever")
    assert signal is None
    assert "NO_TOOL_NEEDED" in text


def test_reason_extracted_from_result():
    from agent.tools.turn_signals import extract_no_tool_reason
    calls = [_call("no_tool_needed", {"reason": "answered directly"})]
    results = [json.dumps({"reason": "answered directly", "ack": "no tool needed"})]
    assert extract_no_tool_reason(calls, results) == "answered directly"


def test_reason_rebuilt_from_args_when_result_unreadable():
    """Results can be truncated or redacted; the call arguments are the fallback."""
    from agent.tools.turn_signals import extract_no_tool_reason
    calls = [_call("no_tool_needed", {"reason": "already done"})]
    assert extract_no_tool_reason(calls, ["<redacted>"]) == "already done"


def test_none_when_not_called_so_absence_is_distinguishable():
    from agent.tools.turn_signals import extract_no_tool_reason
    calls = [_call("read_file", {"path": "/etc/hostname"})]
    assert extract_no_tool_reason(calls, ['{"content":"x"}']) is None


def test_empty_string_when_called_without_a_usable_reason():
    """"" (called, no reason) must not be confused with None (not called)."""
    from agent.tools.turn_signals import extract_no_tool_reason
    calls = [_call("no_tool_needed", {})]
    assert extract_no_tool_reason(calls, ["not json"]) == ""


def test_handler_returns_a_trimmed_reason():
    from agent.tools.turn_signals import no_tool_needed
    assert no_tool_needed("  because it is a question  ")["reason"] == "because it is a question"
