"""Regressions for the session-diagnostics findings (DIAGNOSTICS_FIXES.md).

B1  failure_report stamped failures with a stale session id, because the
    ContextVar carrying it does not cross task boundaries (session switches
    arrive on the UI loop, turns run in their own tasks).
B3  llm_calls.jsonl carried no model identity, so "which model ran turn N"
    was unanswerable from the session directory alone.
B4  tool_calls.jsonl was ~50% duplicates: history collapsing re-logged calls
    already written at execution time.
B2  the auto-tier guard escalated to a metered model on malformed-call
    failures, which a stronger model does not fix.
B6  reflect_session filtered failures hard on session_id, so one bad label
    lost the whole session's material.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest


def _fr_config(tmp_path):
    return SimpleNamespace(
        tools=SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=SimpleNamespace(redact_tool_output=False),
        llm=SimpleNamespace(model="m", ctx_window=0),
    )


def _index_rows(tmp_path):
    idx = tmp_path / ".agent" / "failures" / "index.jsonl"
    return [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines()]


# --- B1 -------------------------------------------------------------------

def test_session_id_survives_a_switch_made_in_another_task(tmp_path):
    """set_session() in one task must apply to a report() in a sibling task."""
    from agent import failure_report as fr

    cfg = _fr_config(tmp_path)

    async def main():
        # A task created *before* the switch: it inherits the old context, so
        # a ContextVar-only implementation reports the stale id.
        started = asyncio.Event()
        release = asyncio.Event()

        async def worker():
            started.set()
            await release.wait()
            fr.report("invalid_tool_call", {"tool": "read_file"}, config=cfg)

        fr.set_session("old-session")
        t = asyncio.create_task(worker())
        await started.wait()
        await asyncio.to_thread(fr.set_session, "new-session")  # switch off-task
        release.set()
        await t

    asyncio.run(main())
    assert [r["session_id"] for r in _index_rows(tmp_path)] == ["new-session"]


def test_explicit_session_id_wins(tmp_path):
    from agent import failure_report as fr

    fr.set_session("ambient")
    fr.report("invalid_tool_call", {"tool": "x"}, config=_fr_config(tmp_path),
              session_id="explicit")
    assert _index_rows(tmp_path)[-1]["session_id"] == "explicit"


# --- B4 -------------------------------------------------------------------

def test_collapse_reuses_existing_side_log_row(tmp_path):
    """A tool call logged at execution time is not re-appended on collapse."""
    from agent.core.history_ops import _collapse_tool_rounds
    from agent.memory.side_log import SideLogWriter

    side_log = SideLogWriter(tmp_path)
    seq = side_log.append("tool_calls.jsonl", {
        "turn": 1, "tool_call_id": "tc1", "tool": "read_file",
        "arguments": {"path": "a.py"}, "result": "{}", "ok": True,
        "duration_ms": 3.0,
    })
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tc1", "type": "function", "function": {
                "name": "read_file", "arguments": json.dumps({"path": "a.py"})}},
        ]},
        {"role": "tool", "tool_call_id": "tc1", "content": "{}"},
    ]
    collapsed = _collapse_tool_rounds(messages, side_log=side_log, turn_id=1)

    rows = (tmp_path / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1, "collapse must not duplicate an already-logged call"
    summary = next(m for m in collapsed if "<agent_exec " in (m.get("content") or ""))
    assert summary["_tool_refs"] == [seq]
    # The surviving row is the richer execution-time one.
    assert json.loads(rows[0])["duration_ms"] == 3.0


def test_side_log_call_index_rebuilds_from_disk(tmp_path):
    from agent.memory.side_log import SideLogWriter

    SideLogWriter(tmp_path).append("tool_calls.jsonl", {"tool_call_id": "tc9"})
    fresh = SideLogWriter(tmp_path)
    assert fresh.seq_for_call_id("tool_calls.jsonl", "tc9") == 0
    assert fresh.seq_for_call_id("tool_calls.jsonl", "nope") is None
    assert fresh.seq_for_call_id("tool_calls.jsonl", None) is None
    assert fresh.append("tool_calls.jsonl", {"tool_call_id": "tc10"}) == 1


# --- B2 -------------------------------------------------------------------

def test_malformed_calls_do_not_count_as_non_convergence():
    from agent.core.confidence import (ConfidenceMonitor, SCHEMA_DOMINANT_SHARE,
                                       classify_error)

    assert classify_error('{"error": "Missing required arguments: path"}') == "schema"
    assert classify_error('{"error": "Invalid JSON arguments: x"}') == "schema"
    assert classify_error('{"error": "empty_args"}') == "schema"
    assert classify_error('{"error": "File not found: /a/b.py"}') == "other"

    m = ConfidenceMonitor(window=8, inject_cooldown=0)
    for _ in range(6):
        m.observe_result('{"error": "Missing required arguments: path"}', is_error=True)
        m.tick_iter()
    sig = m.should_intervene()
    assert sig.triggered
    assert sig.schema_error_share >= SCHEMA_DOMINANT_SHARE
    msg = ConfidenceMonitor.intervention_message(sig)
    assert "malformed" in msg

    env = ConfidenceMonitor(window=8, inject_cooldown=0)
    for _ in range(6):
        env.observe_result('{"error": "File not found: /a/b.py"}', is_error=True)
        env.tick_iter()
    assert env.should_intervene().schema_error_share == 0.0


# --- B6 -------------------------------------------------------------------

def test_failures_fall_back_to_the_session_time_window(tmp_path):
    from agent.memory.reflector import _read_session_failures

    cfg = SimpleNamespace(tools=SimpleNamespace(working_dir=str(tmp_path),
                                                agent_dir=".agent"))
    d = tmp_path / ".agent" / "failures"
    d.mkdir(parents=True)
    rows = [
        {"ts": "2026-07-31T23:52:00+00:00", "kind": "invalid_tool_call",
         "tool": "read_file", "reason": "before_session", "session_id": "other"},
        {"ts": "2026-08-01T11:13:43+00:00", "kind": "invalid_tool_call",
         "tool": "read_file", "reason": "mislabelled", "session_id": "other"},
    ]
    d.joinpath("index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out = _read_session_failures(cfg, "20260801T084006.020Z_cd44")
    assert "mislabelled" in out
    assert "before_session" not in out, "fallback must not reach back before the session"

    rows.append({"ts": "2026-08-01T11:20:00+00:00", "kind": "invalid_tool_call",
                 "tool": "edit_file", "reason": "correctly_labelled",
                 "session_id": "20260801T084006.020Z_cd44"})
    d.joinpath("index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = _read_session_failures(cfg, "20260801T084006.020Z_cd44")
    assert "correctly_labelled" in out
    assert "mislabelled" not in out, "exact matches must suppress the fallback"


# --- B3 -------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_calls_row_names_the_model(tmp_path, monkeypatch):
    from agent._test_helpers import make_client, make_response
    from agent.config import Config
    from agent.config.models import ModelEntry
    from agent.core.turn import run_turn
    from agent.memory.side_log import SideLogWriter
    from agent.tools import load_all_tools

    monkeypatch.chdir(tmp_path)
    config = Config()
    config.tools.working_dir = str(tmp_path)
    config.tools.agent_dir = str(tmp_path / ".agent")
    config.tools.allow_shell = False
    config.llm.narration_fallback = False
    config.llm.base_url = "http://x/v1"
    config.llm.model = "deepseek-v4-flash"
    config.model_entries = {"fast": ModelEntry(base_url="http://x/v1",
                                               model="deepseek-v4-flash",
                                               cost_out_per_1k=0.0)}
    load_all_tools(config=config)
    side_log = SideLogWriter(tmp_path / "_side")

    response = make_response(content="done.")
    response.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                     prompt_tokens_details=None)
    client = make_client(response)
    await run_turn([{"role": "user", "content": "hi"}], config, client,
                   side_log=side_log, turn_index=3)

    rows = [json.loads(l) for l in
            (tmp_path / "_side" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows, "usage must be logged"
    assert rows[0]["entry_name"] == "fast"
    assert rows[0]["model"] == "deepseek-v4-flash"
    assert rows[0]["tier"] in {"local", "free", "paid", "bundled"}
    assert rows[0]["role"] == "main"
