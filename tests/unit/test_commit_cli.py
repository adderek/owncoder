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


# ── model listing (-m with no value) ──────────────────────────────────────

def test_bare_m_flag_lists_models_instead_of_erroring():
    """`agent commit -m` used to die with 'expected one argument'."""
    from agent.cli.main import build_parser

    parser = build_parser()
    assert parser.parse_args(["commit", "-m"]).model == "__list__"
    assert parser.parse_args(["commit", "-m", "gpu-gemma4"]).model == "gpu-gemma4"
    assert parser.parse_args(["commit"]).model is None
    assert parser.parse_args(["commit", "-m"]).probe is True
    assert parser.parse_args(["commit", "-m", "--no-probe"]).probe is False


class _Entry:
    def __init__(self, model, base_url, tags=()):
        self.model = model
        self.base_url = base_url
        self.api_key = ""
        self.tags = list(tags)
        self.tokens_per_sec = 0.0


class _Registry:
    def __init__(self, entries):
        self._entries = entries

    def names(self):
        return list(self._entries)

    def get(self, name):
        return self._entries.get(name)


def _listing(monkeypatch, entries, live, probe=True):
    """Render the table with a stubbed probe; return the printed text."""
    import agent.cli.commit as mod
    from rich.console import Console

    monkeypatch.setattr(mod, "_probe_endpoints", lambda registry, timeout=2: live)
    console = Console(width=200, force_terminal=False, no_color=True)
    with console.capture() as cap:
        mod._print_model_list(console, _Registry(entries), [], probe=probe)
    return cap.get()


def test_listing_marks_a_served_model_live(monkeypatch):
    entries = {"remote": _Entry("27b-q8", "http://192.168.31.42:8081/v1")}
    out = _listing(monkeypatch, entries,
                   {"http://192.168.31.42:8081/v1": {"Qwen3.6-27b-q8_0.gguf"}})
    assert "✓" in out


def test_listing_names_what_a_one_model_server_has_loaded(monkeypatch):
    """The :8081 box serves one gguf at a time — say which, don't just say ✗."""
    entries = {"remote": _Entry("coder-next-ud-iq4_xs", "http://192.168.31.42:8081/v1")}
    out = _listing(monkeypatch, entries,
                   {"http://192.168.31.42:8081/v1": {"/models/gguf/Qwen3.6-27B-Q8_0.gguf"}})
    assert "~" in out
    assert "loaded:" in out
    assert "Qwen3.6-27B-Q8_0.gguf" in out, "the path must be shortened to a basename"


def test_listing_calls_a_multi_preset_endpoint_a_router(monkeypatch):
    entries = {"remote": _Entry("missing-model", "http://192.168.31.42:8081/v1")}
    out = _listing(monkeypatch, entries,
                   {"http://192.168.31.42:8081/v1": {"a.gguf", "b.gguf"}})
    assert "router:" in out


def test_listing_reports_an_unreachable_endpoint(monkeypatch):
    entries = {"local": _Entry("qwen", "http://localhost:8081/v1")}
    out = _listing(monkeypatch, entries, {"http://localhost:8081/v1": None})
    assert "✗" in out and "unreachable" in out


def test_no_probe_drops_the_availability_columns(monkeypatch):
    entries = {"local": _Entry("qwen", "http://localhost:8081/v1")}
    out = _listing(monkeypatch, entries, {}, probe=False)
    assert "endpoint serves" not in out and "unreachable" not in out


def test_probe_queries_each_endpoint_once(monkeypatch):
    """Entries sharing a base_url must cost one GET, not one each."""
    import agent.cli.commit as mod

    calls: list[str] = []

    def fake_list(base_url, api_key="", timeout=3):
        calls.append(base_url)
        return {"x"}

    monkeypatch.setattr("agent.config.model_probe.list_endpoint_models", fake_list)
    entries = {f"e{i}": _Entry(f"m{i}", "http://192.168.31.42:8081/v1") for i in range(5)}
    entries["other"] = _Entry("m", "http://192.168.31.42:8083/v1")
    result = mod._probe_endpoints(_Registry(entries))
    assert sorted(calls) == ["http://192.168.31.42:8081/v1", "http://192.168.31.42:8083/v1"]
    assert set(result) == set(calls)
