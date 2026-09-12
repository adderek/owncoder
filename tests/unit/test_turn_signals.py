"""Signal parsing is anchored to the closing line.

A signal quoted inside a message — a relayed reply, or prose explaining this
format — must survive untouched. Scanning the whole body with `sub()` deleted
such a quoted `>>>DONE`, which is how the marker went missing from a forwarded
passage.
"""
from __future__ import annotations


def test_signal_is_only_the_closing_line():
    from agent.core.turn_signals import parse_signal
    text, signal = parse_signal("Here is the answer.\n>>>DONE: summary")
    assert signal is not None and signal.kind == "done"
    assert signal.payload == "summary"
    assert text == "Here is the answer."


def test_quoted_signal_in_middle_is_left_alone():
    """The reported failure: a relayed quote containing >>>DONE lost that line."""
    from agent.core.turn_signals import parse_signal
    body = "He wrote:\n>>>DONE: released\nand then he stopped."
    text, signal = parse_signal(body)
    assert signal is None
    assert text == body


def test_bare_marker_parses_and_strips():
    from agent.core.turn_signals import parse_signal
    text, signal = parse_signal(">>>DONE")
    assert signal is not None and signal.kind == "done"
    assert signal.payload == ""
    assert text == ""


def test_trailing_blank_lines_are_skipped():
    from agent.core.turn_signals import parse_signal
    text, signal = parse_signal("Answer.\n>>>ASK: which one?\n\n")
    assert signal is not None and signal.kind == "ask_user"
    assert signal.payload == "which one?"
    assert text == "Answer."


def test_signal_fenced_as_code_is_not_a_signal():
    from agent.core.turn_signals import parse_signal
    body = "The syntax is:\n\n```\n>>>DONE: summary\n```"
    text, signal = parse_signal(body)
    assert signal is None
    assert text == body


def test_strip_signals_keeps_quoted_text():
    from agent.core.turn_signals import strip_signals
    body = "He wrote:\n>>>BLOCKED: waiting\nand stopped."
    assert strip_signals(body) == body
    assert strip_signals("Answer.\n>>>DONE: x") == "Answer."
