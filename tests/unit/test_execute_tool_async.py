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
