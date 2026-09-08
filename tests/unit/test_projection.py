"""Projection — one event stream, three client budgets."""
from __future__ import annotations

import json

from agent.ipc.messages import (
    ChangesetEvent,
    ReasoningEvent,
    SignalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEndEvent,
)
from agent.ui_server.projection import (
    PROFILES,
    approx_tokens,
    clip_to_tokens,
    plain,
    project,
    resolve_profile,
)
from agent.ui_server.view_model import ViewModel


def _view(text="hello world", *, tools=(), changeset=None, signal=None,
          reasoning="") -> ViewModel:
    vm = ViewModel()
    vm.add_user_message("do it")
    if reasoning:
        vm.apply(ReasoningEvent(token=reasoning))
    vm.apply(TokenEvent(token=text))
    for name, ok in tools:
        vm.apply(ToolCallEvent(name=name, args="{}"))
        vm.apply(ToolResultEvent(name=name, ok=ok))
    if changeset is not None:
        vm.apply(ChangesetEvent(changeset=changeset))
    vm.apply(TurnEndEvent(response=text))
    if signal is not None:
        vm.apply(signal)
    return vm


def test_unknown_profile_falls_back_to_full():
    assert resolve_profile("nope").name == "full"
    assert resolve_profile(None).name == "full"
    assert resolve_profile(" GLANCE ").name == "glance"


def _live_view(text="hello", reasoning="") -> ViewModel:
    """A turn still in progress: reasoning only exists before TurnEndEvent."""
    vm = ViewModel()
    vm.add_user_message("do it")
    if reasoning:
        vm.apply(ReasoningEvent(token=reasoning))
    vm.apply(TokenEvent(token=text))
    return vm


def test_full_keeps_markdown_and_streaming():
    out = project(_view("# Title\n\n**bold** and `code`"), "full")
    assert out["profile"] == "full"
    assert out["stream_tokens"] is True
    assert out["entries"][-1]["text"] == "# Title\n\n**bold** and `code`"


def test_reasoning_is_full_only():
    assert project(_live_view(reasoning="because"), "full")["reasoning"] == "because"
    assert project(_live_view(reasoning="because"), "glance")["reasoning"] == ""


def test_glance_strips_markdown():
    out = project(_view("# Title\n\n**bold** and `code`"), "glance")
    assert out["stream_tokens"] is False
    text = out["entries"][-1]["text"]
    assert "#" not in text and "**" not in text and "`" not in text
    assert "Title" in text and "bold" in text


def test_glance_keeps_only_the_last_entry():
    vm = _view("first")
    vm.add_user_message("again")
    vm.apply(TokenEvent(token="second"))
    vm.apply(TurnEndEvent(response="second"))
    out = project(vm, "glance")
    assert len(out["entries"]) == 1
    assert out["entries"][0]["text"] == "second"


def test_glance_clips_to_its_budget():
    long_text = "word " * 300
    out = project(_view(long_text), "glance")
    text = out["entries"][-1]["text"]
    assert text.endswith("…")
    assert approx_tokens(text) <= PROFILES["glance"].max_out_tokens + 1


def test_compact_keeps_markdown():
    assert project(_view("**bold**"), "compact")["entries"][-1]["text"] == "**bold**"


def test_signal_payload_capped_to_max_options():
    sig = SignalEvent(kind="ask_user", payload="\n".join(f"opt{i}" for i in range(6)))
    out = project(_view(signal=sig), "glance")
    assert out["pending_signal"]["payload"].count("\n") + 1 == 3
    assert out["pending_signal"]["kind"] == "ask_user"


def test_changeset_metadata_only_for_glance():
    cs = {"turn_id": 1, "files": [{"path": "a.py", "added": 2}], "prose": "x"}
    out = project(_view(changeset=cs), "glance")
    assert out["entries"][-1]["changeset"] == {"files": [{"path": "a.py", "added": 2}]}
    out_full = project(_view(changeset=cs), "full")
    assert out_full["entries"][-1]["changeset"] == cs


def test_tool_results_reach_every_profile():
    out = project(_view(tools=[("grep_code", True), ("read_file", False)]), "glance")
    assert [(t["name"], t["ok"]) for t in out["entries"][-1]["tools"]] == [
        ("grep_code", True), ("read_file", False)]


def test_projection_is_json_serializable():
    out = project(_view(tools=[("grep_code", True)]), "compact")
    assert json.loads(json.dumps(out))["profile"] == "compact"


def test_clip_to_tokens_budget_zero_is_unlimited():
    assert clip_to_tokens("x" * 1000, 0) == "x" * 1000
    assert clip_to_tokens("short", 100) == "short"


def test_clip_prefers_word_boundary():
    clipped = clip_to_tokens("alpha beta gamma delta", 3)
    assert clipped == "alpha beta…"


def test_plain_strips_ansi_and_links():
    assert plain("\x1b[31mred\x1b[0m") == "red"
    assert plain("see [docs](http://x)") == "see docs"
