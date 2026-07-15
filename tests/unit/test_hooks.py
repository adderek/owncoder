"""Tool-call shell hooks: matching, pre-block, post-notes, env passing."""
import json

import pytest

from agent.config.models import Config, HookConfig
from agent.core import hooks


def _cfg(*entries):
    c = Config()
    c.hooks.entries = list(entries)
    return c


@pytest.mark.asyncio
async def test_pre_hook_blocks_on_nonzero():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command="echo bad; exit 1", block=True, name="deny"))
    allow, msg = await hooks.run_pre_tool(c, "edit_file", {"path": "x.py"})
    assert not allow
    assert "deny" in msg and "bad" in msg


@pytest.mark.asyncio
async def test_pre_hook_nonblocking_allows_despite_nonzero():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"],
                        command="exit 5", block=False))
    allow, _ = await hooks.run_pre_tool(c, "anything", {})
    assert allow


@pytest.mark.asyncio
async def test_pre_hook_only_matching_tools():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command="exit 1", block=True))
    allow, _ = await hooks.run_pre_tool(c, "read_file", {})
    assert allow  # no matching hook


@pytest.mark.asyncio
async def test_post_hook_notes_on_nonzero():
    c = _cfg(HookConfig(event="post_tool", tools=["*"],
                        command="echo lint-warn; exit 1", name="lint"))
    notes = await hooks.run_post_tool(c, "edit_file", {"path": "x"}, "{}")
    assert notes and "lint" in notes[0] and "lint-warn" in notes[0]


@pytest.mark.asyncio
async def test_post_hook_silent_on_success():
    c = _cfg(HookConfig(event="post_tool", tools=["*"], command="true"))
    notes = await hooks.run_post_tool(c, "edit_file", {}, "{}")
    assert notes == []


@pytest.mark.asyncio
async def test_env_tool_path_passed():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command='test "$TOOL_PATH" = "a.py"', block=True))
    ok_allow, _ = await hooks.run_pre_tool(c, "edit_file", {"path": "a.py"})
    bad_allow, _ = await hooks.run_pre_tool(c, "edit_file", {"path": "b.py"})
    assert ok_allow and not bad_allow


@pytest.mark.asyncio
async def test_disabled_section_is_noop():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"], command="exit 1", block=True))
    c.hooks.enabled = False
    allow, _ = await hooks.run_pre_tool(c, "edit_file", {})
    assert allow


@pytest.mark.asyncio
async def test_timeout_blocks_when_blocking():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"],
                        command="sleep 5", block=True, timeout_s=0.3))
    allow, msg = await hooks.run_pre_tool(c, "edit_file", {})
    assert not allow and "timed out" in msg


@pytest.mark.asyncio
async def test_none_config_is_noop():
    allow, _ = await hooks.run_pre_tool(None, "edit_file", {})
    assert allow
    assert await hooks.run_post_tool(None, "edit_file", {}, "{}") == []


@pytest.mark.asyncio
async def test_end_to_end_through_execute_tool():
    from agent.tools import register
    from agent.core.tool_calls import execute_tool

    @register("hooktest_echo", {"description": "echo",
              "parameters": {"type": "object",
                             "properties": {"path": {"type": "string"}},
                             "required": ["path"]}})
    def _echo(path):
        return {"ok": True, "path": path}

    class _Call:
        id = "c1"

        def __init__(self, name, args):
            self.function = type("F", (), {"name": name,
                                           "arguments": json.dumps(args)})()

    c = _cfg(HookConfig(event="pre_tool", tools=["hooktest_echo"],
                        command="exit 1", block=True, name="blk"))
    res = json.loads(await execute_tool(_Call("hooktest_echo", {"path": "p"}), c))
    assert res.get("blocked_by_hook")

    c.hooks.entries = [HookConfig(event="post_tool", tools=["hooktest_echo"],
                                  command="echo RAN; exit 2", name="pn")]
    res2 = await execute_tool(_Call("hooktest_echo", {"path": "p"}), c)
    assert "[hook notes]" in res2 and "RAN" in res2
