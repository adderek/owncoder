"""Session-end steps after Ctrl+C: visible, error-tolerant, skippable."""
from __future__ import annotations

import io

from rich.console import Console

from agent.cli.chat import _run_teardown_steps


def _console():
    buf = io.StringIO()
    return Console(file=buf, force_terminal=False, width=200), buf


def test_each_step_reported_with_result():
    con, buf = _console()
    _run_teardown_steps(con, [("saving notes", "note(s)", lambda: 2),
                              ("distilling skills", "skill(s)", lambda: 0),
                              ("checking compiled prompts", "verdict(s)", lambda: [1])])
    out = buf.getvalue()
    assert "Ctrl+C again to skip" in out
    assert "saving notes: 2 note(s)" in out
    assert "distilling skills: nothing new" in out
    assert "checking compiled prompts: 1 verdict(s)" in out


def test_failure_does_not_stop_later_steps():
    con, buf = _console()
    ran = []

    def boom():
        raise RuntimeError("llm down")
    _run_teardown_steps(con, [("a", "x", boom), ("b", "x", lambda: ran.append(1) or 1)])
    assert "a: failed" in buf.getvalue() and ran == [1]


def test_ctrl_c_skips_current_and_rest():
    con, buf = _console()
    ran = []

    def interrupted():
        raise KeyboardInterrupt
    _run_teardown_steps(con, [("a", "x", interrupted), ("b", "x", lambda: ran.append(1))])
    assert "skipped: a and the rest" in buf.getvalue() and ran == []
