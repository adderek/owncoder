"""`agent commit` helpers — first tests for cli/commit.py.

Written after the failure miner (evals/mine.py) surfaced three recorded crashes
from this command. Two were already fixed by the candidate-fallback chain, but
reproducing them turned up a live one: the confirmation prompts were the only
interactive prompts in cli/ without an EOFError guard, so running `agent commit`
from a script or with piped output died with a traceback and a crash dump
instead of saying what was wrong.
"""
from __future__ import annotations

import builtins

import pytest

from agent.cli.commit import (
    CommitModelError, _ask, _err_brief, _resolve_confirmation, _split_diff,
)


class _Console:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


@pytest.fixture
def no_stdin(monkeypatch):
    """Make any read of stdin raise EOFError, as a closed pipe does."""
    def _boom(*a, **kw):
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr(builtins, "input", _boom)


# ── _ask ──────────────────────────────────────────────────────────────────

def test_ask_returns_the_answer_when_there_is_a_terminal(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: "n")
    assert _ask(_Console(), "go?", choices=["y", "n"], default="y") == "n"


def test_ask_falls_back_to_the_default_on_empty_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: "")
    assert _ask(_Console(), "go?", choices=["y", "n"], default="y") == "y"


def test_ask_returns_none_on_eof_by_default(no_stdin):
    """None is 'the caller decides' — cmd_commit uses it to refuse to commit."""
    assert _ask(_Console(), "go?", choices=["y", "n"], default="y") is None


def test_ask_returns_the_requested_verdict_on_eof(no_stdin):
    assert _ask(_Console(), "go?", choices=["y", "n"], default="y", on_eof="n") == "n"


def test_ask_treats_ctrl_c_as_a_decline(monkeypatch):
    def _interrupt(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", _interrupt)
    assert _ask(_Console(), "go?", choices=["y", "n"], default="y", on_eof="n") == "n"


def test_ask_never_raises_for_either_interruption(no_stdin):
    """The whole point: no traceback reaches main() from a closed stdin."""
    assert _ask(_Console(), "q", choices=["y"], default="y") is None


# ── the non-interactive contract ──────────────────────────────────────────

class _Args:
    def __init__(self, **kw):
        self.yes = kw.get("yes", False)
        self.print_only = kw.get("print_only", False)


def test_confirmation_exits_non_zero_rather_than_committing_unapproved(no_stdin):
    """The recorded crash. A commit nobody approved is the one outcome worse
    than no commit, so no terminal means stop — and loudly enough that a script
    notices."""
    console = _Console()
    with pytest.raises(SystemExit) as exc:
        _resolve_confirmation(_Args(), console)
    assert exc.value.code == 1
    joined = " ".join(console.lines)
    assert "no terminal to confirm on" in joined
    assert "-y" in joined and "--print" in joined      # says how to opt in


def test_yes_skips_the_prompt_entirely(no_stdin):
    """--yes must not touch stdin at all, or scripts break the same way."""
    assert _resolve_confirmation(_Args(yes=True), _Console()) == "y"


def test_print_only_neither_asks_nor_commits(no_stdin):
    assert _resolve_confirmation(_Args(print_only=True), _Console()) == "print"


def test_print_only_wins_over_yes(no_stdin):
    """--print is the more conservative of the two; it must not commit."""
    assert _resolve_confirmation(_Args(yes=True, print_only=True), _Console()) == "print"


@pytest.mark.parametrize("answer", ["y", "n", "e", "rpt"])
def test_an_interactive_answer_is_passed_through(monkeypatch, answer):
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: answer)
    assert _resolve_confirmation(_Args(), _Console()) == answer


def test_a_declined_prompt_does_not_exit_non_zero(monkeypatch):
    """Typing "n" is a decision, not a failure — it must stay distinguishable
    from having no terminal."""
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: "n")
    assert _resolve_confirmation(_Args(), _Console()) == "n"


def test_yes_and_print_only_flags_are_registered():
    """Without these, the EOF refusal would leave no way to script the command."""
    from agent.cli.main import build_parser

    parser = build_parser()
    args = parser.parse_args(["commit", ".", "-y"])
    assert args.yes is True
    assert args.print_only is False
    args = parser.parse_args(["commit", ".", "--print"])
    assert args.print_only is True
    assert args.yes is False
    args = parser.parse_args(["commit", "."])
    assert args.yes is False and args.print_only is False


# ── pre-existing helpers, previously untested ─────────────────────────────

def test_err_brief_is_one_capped_line():
    brief = _err_brief(ValueError("first line\nsecond line"))
    assert brief == "ValueError: first line"
    assert "\n" not in _err_brief(RuntimeError("x" * 500))
    assert len(_err_brief(RuntimeError("x" * 500))) <= 220


def test_err_brief_handles_an_empty_message():
    assert _err_brief(RuntimeError()) == "RuntimeError"


def test_commit_model_error_carries_per_model_diagnostics():
    failures = [{"entry": "gpu", "model": "m", "base_url": "http://x", "error": "boom"}]
    err = CommitModelError(failures)
    assert err.failures == failures
    assert "no usable model" in str(err)


def test_split_diff_returns_one_chunk_when_it_fits():
    diff = "diff --git a/a b/a\n@@ -1 +1 @@\n-x\n+y\n"
    assert _split_diff(diff, 10_000) == [diff]


def test_split_diff_prefers_file_boundaries():
    a = "diff --git a/a b/a\n" + "+x\n" * 40
    b = "diff --git a/b b/b\n" + "+y\n" * 40
    chunks = _split_diff(a + b, len(a) + 5)
    assert len(chunks) == 2
    assert chunks[0].startswith("diff --git a/a")
    assert chunks[1].startswith("diff --git a/b")


def test_split_diff_keeps_the_file_header_on_each_hunk_of_a_big_file():
    header = "diff --git a/big b/big\nindex 1..2 100644\n"
    hunks = "".join(f"@@ -{i} +{i} @@\n" + "+line\n" * 20 for i in range(4))
    chunks = _split_diff(header + hunks, 400)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("diff --git a/big"), "a chunk without its header loses context"


def test_split_diff_never_splits_inside_a_line():
    diff = "diff --git a/a b/a\n" + "".join(f"+line{i}\n" for i in range(200))
    for chunk in _split_diff(diff, 300):
        assert chunk.endswith("\n")
