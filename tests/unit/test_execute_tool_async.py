"""execute_tool must support async (coroutine) tools.

spawn_agents and ask_internet are registered as `async def`. A prior version ran
every tool via run_in_executor, which for an async fn returned an un-awaited
coroutine that then failed to JSON-serialise ("Object of type coroutine is not
JSON serializable").
"""
from __future__ import annotations

import asyncio
import json
import warnings

from agent.tools import register
from agent.core.tool_calls import execute_tool, _FakeToolCall


def test_async_tool_is_awaited():
    @register(
        "async_tool_under_test",
        {"description": "x", "parameters": {"type": "object", "properties": {}, "required": []}},
    )
    async def _async_tool():
        await asyncio.sleep(0)
        return {"ok": True, "value": 42}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # "coroutine was never awaited"
        out = asyncio.run(execute_tool(_FakeToolCall("async_tool_under_test", {}), None))

    assert json.loads(out) == {"ok": True, "value": 42}


def test_sync_tool_still_works():
    @register(
        "sync_tool_under_test",
        {"description": "x", "parameters": {"type": "object", "properties": {}, "required": []}},
    )
    def _sync_tool():
        return {"ok": True, "value": 7}

    out = asyncio.run(execute_tool(_FakeToolCall("sync_tool_under_test", {}), None))
    assert json.loads(out) == {"ok": True, "value": 7}


def test_stripped_unknown_args_reported():
    """Repairing a call (stripping unknown args) must leave a failure-report
    trail so arg-shape mistakes stay visible in the data."""
    from unittest.mock import patch

    @register(
        "strip_report_tool_under_test",
        {"description": "x", "parameters": {"type": "object",
         "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    )
    def _tool(path: str):
        return {"ok": True, "path": path}

    with patch("agent.failure_report.report") as rep:
        out = asyncio.run(execute_tool(
            _FakeToolCall("strip_report_tool_under_test",
                          {"path": "a.py", "bogus": 1}), None))
    assert json.loads(out)["ok"] is True
    kinds = [c.args[0] for c in rep.call_args_list]
    assert "repaired_tool_call" in kinds


def test_missing_required_args_reported():
    from unittest.mock import patch

    @register(
        "missing_report_tool_under_test",
        {"description": "x", "parameters": {"type": "object",
         "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    )
    def _tool(path: str):
        return {"ok": True}

    with patch("agent.failure_report.report") as rep:
        out = asyncio.run(execute_tool(
            _FakeToolCall("missing_report_tool_under_test", {}), None))
    assert "Missing required" in json.loads(out)["error"]
    assert any(c.args[0] == "invalid_tool_call" and
               c.args[1].get("reason") == "missing_required_args"
               for c in rep.call_args_list)
