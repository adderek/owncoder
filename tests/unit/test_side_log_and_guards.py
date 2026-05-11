"""Tests for SideLogWriter, the shrink-guard in _apply_code_from_history,
and _collapse_tool_rounds' side-log integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def reset_file_tool_state(monkeypatch):
    """Reset global state in agent.tools.rules / agent.tools.files so tests
    that call write_file don't inherit a read-only or ignored tree from
    earlier tests in the same process."""
    import agent.tools.rules as rules_mod
    import agent.tools.files as files_mod
    import agent.tools.files.paths as files_paths_mod
    monkeypatch.setattr(rules_mod, "_rules", None)
    monkeypatch.setattr(files_mod, "_config", None)
    monkeypatch.setattr(files_paths_mod, "_config", None)
    try:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        monkeypatch.setattr(_sec_policy, "_config", None, raising=False)
        monkeypatch.setattr(_sec_fs, "_root_dev", None, raising=False)
        monkeypatch.setattr(_sec_fs, "_root_ino", None, raising=False)
    except Exception:
        pass
    yield


def test_side_log_append_and_read(tmp_path):
    from agent.memory.side_log import SideLogWriter

    w = SideLogWriter(tmp_path)
    s0 = w.append("tool_calls.jsonl", {"tool": "read_file", "arguments": {"path": "a"}})
    s1 = w.append("tool_calls.jsonl", {"tool": "write_file", "arguments": {"path": "b"}})
    assert (s0, s1) == (0, 1)

    r0 = w.read("tool_calls.jsonl", 0)
    assert r0["tool"] == "read_file"
    assert r0["arguments"] == {"path": "a"}
    assert r0["seq"] == 0
    assert "ts" in r0

    r1 = w.read("tool_calls.jsonl", 1)
    assert r1["tool"] == "write_file"


def test_side_log_resumes_seq_across_instances(tmp_path):
    from agent.memory.side_log import SideLogWriter

    w1 = SideLogWriter(tmp_path)
    w1.append("tool_calls.jsonl", {"tool": "a"})
    w1.append("tool_calls.jsonl", {"tool": "b"})

    # Fresh writer should not re-use existing seqs.
    w2 = SideLogWriter(tmp_path)
    s = w2.append("tool_calls.jsonl", {"tool": "c"})
    assert s == 2


def test_collapse_writes_full_detail_to_side_log(tmp_path):
    from agent.core.history_ops import _collapse_tool_rounds
    from agent.memory.side_log import SideLogWriter

    side_log = SideLogWriter(tmp_path)
    long_content = "x" * 5000
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "foo.py", "content": long_content}),
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": json.dumps({"ok": True})},
        {"role": "assistant", "content": "Done."},
    ]
    collapsed = _collapse_tool_rounds(messages, side_log=side_log, turn_id=7)

    # Summary message stays compact.
    summary = next(m for m in collapsed if m.get("role") == "assistant" and "<agent_exec " in m.get("content", ""))
    assert long_content not in summary["content"], "full content must not appear in session summary"
    assert summary.get("_tool_refs") == [0], "summary must reference side-log seq"

    # Full content lives in tool_calls.jsonl.
    jsonl = (tmp_path / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1
    row = json.loads(jsonl[0])
    assert row["tool"] == "write_file"
    assert row["turn"] == 7
    assert row["tool_call_id"] == "tc1"
    assert row["arguments"]["content"] == long_content


def test_collapse_without_side_log_is_unchanged():
    # Backward-compat: existing callers that pass no side_log keep the old
    # behaviour (no _tool_refs, no IO).
    from agent.core.history_ops import _collapse_tool_rounds

    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"x.py"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": '{"content":"hi"}'},
        {"role": "assistant", "content": "Done."},
    ]
    collapsed = _collapse_tool_rounds(messages)
    summary = next(m for m in collapsed if m.get("role") == "assistant")
    assert "_tool_refs" not in summary


def test_apply_code_refuses_to_shrink_existing_file(tmp_path, monkeypatch, reset_file_tool_state):
    # Simulate the Dusty/agents.js incident: existing file is large, assistant
    # message contains only a 4-line illustrative excerpt.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "agents.js"
    target.write_text("// " + ("big file\n" * 500), encoding="utf-8")
    original = target.read_text(encoding="utf-8")

    from agent.core.history_ops import _apply_code_from_history

    assistant_content = (
        "I will write an update to `agents.js`:\n"
        "```js\nconst budget = loadBudget();\n```"
    )
    messages = [
        {"role": "user", "content": "analyze"},
        {"role": "assistant", "content": assistant_content},
    ]
    result = _apply_code_from_history(messages, on_tool_call=None)
    assert result is not None
    human, summary = result
    assert "Refused" in human
    assert summary["role"] == "assistant"
    assert "refused" in summary["content"].lower()
    # File untouched.
    assert target.read_text(encoding="utf-8") == original


def test_apply_code_allows_full_rewrite(tmp_path, monkeypatch, reset_file_tool_state):
    # Guard should not block legitimate near-same-size rewrites.
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "small.py"
    target.write_text("print('old')\n", encoding="utf-8")

    from agent.core.history_ops import _apply_code_from_history

    new_body = "print('new line one')\nprint('new line two')\n"
    messages = [
        {
            "role": "assistant",
            "content": f"Let me write the replacement for `small.py`:\n```python\n{new_body}```",
        },
    ]
    result = _apply_code_from_history(messages, on_tool_call=None)
    assert result is not None
    human, summary = result
    assert "Refused" not in human
    assert "new line" in target.read_text(encoding="utf-8")
    assert summary["role"] == "assistant"
    assert "ok" in summary["content"]


async def test_narration_fallback_kill_switch(tmp_path, monkeypatch, reset_file_tool_state):
    """When llm.narration_fallback=False, narrated writes don't silently happen."""
    from agent.core.turn import run_turn
    from agent.config import Config
    from agent.tools import load_all_tools
    from agent._test_helpers import make_response, make_client

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "app.py"
    target.write_text("original\n", encoding="utf-8")
    original = target.read_text(encoding="utf-8")

    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.allow_shell = False
    cfg.llm.narration_fallback = False
    load_all_tools(config=cfg)

    client = make_client(make_response(
        content="I'll write a new version of `app.py`:\n```python\nprint('replacement line one')\nprint('replacement line two')\nprint('replacement line three')\n```",
    ))
    messages = [{"role": "user", "content": "rewrite"}]
    await run_turn(messages, cfg, client)
    assert target.read_text(encoding="utf-8") == original


async def test_run_turn_writes_reasoning_to_side_log(tmp_path, monkeypatch, reset_file_tool_state):
    import json as _json
    from agent.core.turn import run_turn
    from agent.config import Config
    from agent.memory.side_log import SideLogWriter
    from agent.tools import load_all_tools
    from agent._test_helpers import make_response, make_client

    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.allow_shell = False
    load_all_tools(config=cfg)

    side_dir = tmp_path / "_side"
    writer = SideLogWriter(side_dir)

    client = make_client(make_response(
        content="Hello.",
        reasoning="thinking hard about this",
    ))
    messages = [{"role": "user", "content": "hi"}]
    _, new_msgs = await run_turn(messages, cfg, client, side_log=writer, turn_index=5)

    rows = (side_dir / "reasoning.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = _json.loads(rows[0])
    assert row["content"] == "thinking hard about this"
    assert row["turn"] == 5

    last_assistant = [m for m in new_msgs if m.get("role") == "assistant"][-1]
    assert last_assistant.get("_reasoning_ref") == 0


def test_sessions_split_command_extracts_tool_rounds(tmp_path, monkeypatch):
    """`agent sessions --split <id>` extracts inline tool detail into tool_calls.jsonl."""
    import json as _json
    import types
    from agent.memory import session as session_mod
    from agent.main import _split_sessions

    sdir = tmp_path / "2026" / "04" / "18" / "20260418T120000.000Z"
    sdir.mkdir(parents=True)
    session_json = sdir / "session.json"
    session_json.write_text(_json.dumps({
        "id": "20260418T120000.000Z",
        "messages": [
            {"role": "user", "content": "do x"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": _json.dumps({"path": "x.py"})},
                }],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": _json.dumps({"content": "huge file content " * 200})},
            {"role": "assistant", "content": "done"},
        ],
    }, indent=2), encoding="utf-8")

    monkeypatch.setattr(session_mod, "_session_dir", tmp_path)

    class _C:
        def print(self, *a, **k): pass

    _split_sessions("20260418T120000.000Z", _C(), dry_run=False)

    jsonl = sdir / "tool_calls.jsonl"
    assert jsonl.exists()
    rows = [_json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["tool"] == "read_file"
    assert rows[0]["arguments"] == {"path": "x.py"}
    assert "huge file content" in rows[0]["result"]

    rewritten = _json.loads(session_json.read_text(encoding="utf-8"))
    summaries = [m for m in rewritten["messages"] if m.get("role") == "assistant" and "<agent_exec " in m.get("content", "")]
    assert summaries and summaries[0].get("_tool_refs") == [0]
    assert (sdir / "session.json.bak").exists()


def test_apply_code_writes_side_log_row(tmp_path, monkeypatch, reset_file_tool_state):
    from agent.core.history_ops import _apply_code_from_history
    from agent.memory.side_log import SideLogWriter
    import json as _json

    monkeypatch.chdir(tmp_path)
    target = tmp_path / "hello.py"
    target.write_text("print('old')\n", encoding="utf-8")

    side_dir = tmp_path / "_sidelog"
    writer = SideLogWriter(side_dir)
    new_body = "print('new')\nprint('row')\n"
    messages = [
        {
            "role": "assistant",
            "content": f"Let me write the replacement for `hello.py`:\n```python\n{new_body}```",
        },
    ]
    result = _apply_code_from_history(messages, on_tool_call=None, side_log=writer, turn_id=3)
    assert result is not None
    human, summary = result
    assert summary.get("_tool_refs") == [0]

    rows = (side_dir / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = _json.loads(rows[0])
    assert row["tool"] == "write_file (extracted)"
    assert row["source"] == "narration_fallback"
    assert row["turn"] == 3
    assert row["arguments"]["path"] == "hello.py"
    assert row["result"]["outcome"] == "ok"
