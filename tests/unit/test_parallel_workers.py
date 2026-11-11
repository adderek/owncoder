"""A spawn_agents worker that runs out of time still hands back what it had."""
from __future__ import annotations

import asyncio

import agent.core.turn as turn_mod
from agent.config import Config
from agent.config.models import ModelEntry
from agent.tools.parallel import main as par


def _config(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    e = ModelEntry()
    e.base_url, e.model = "http://127.0.0.1:1/v1", "fake"
    c.model_entries = {"w": e}
    return c


def test_partial_of_reports_last_text_and_tools():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "found the bug in a.py"},
    ]
    assert par._partial_of(msgs, 2) == {
        "partial": "found the bug in a.py", "tool_calls_done": ["read_file"]}


def test_partial_of_is_empty_when_nothing_happened():
    seeded = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    assert par._partial_of(seeded, 2) == {}
    assert par._partial_of([], 2) == {}


async def test_timeout_keeps_the_checkpointed_work(tmp_path, monkeypatch):
    async def slow_turn(messages, partial_sink=None, **_kw):
        partial_sink[:] = messages + [{"role": "assistant", "content": "half an answer"}]
        await asyncio.sleep(30)

    monkeypatch.setattr(turn_mod, "run_turn", slow_turn)
    res = await par._run_worker("task", "w", _config(tmp_path), None, 1, set())
    assert res["output"] is None
    assert res["error"] == "timeout after 1s"
    assert res["partial"] == "half an answer"
    assert res["tool_calls_done"] == []


async def test_finished_worker_result_is_unchanged(tmp_path, monkeypatch):
    async def quick_turn(messages, **_kw):
        return "done", messages

    monkeypatch.setattr(turn_mod, "run_turn", quick_turn)
    res = await par._run_worker("task", "w", _config(tmp_path), None, 5, set())
    assert res == {"model": "w", "output": "done", "error": None, "tokens": {}}
