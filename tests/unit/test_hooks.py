"""Tool-call shell hooks: matching, pre-block, post-notes, env passing."""
import json

import pytest

from agent.config.models import Config, HookConfig
from agent.core import hooks


pytestmark = pytest.mark.usefixtures("sandbox_policy")


def _cfg(*entries):
    c = Config()
    c.hooks.entries = list(entries)
    return c


async def test_pre_hook_blocks_on_nonzero():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command="echo bad; exit 1", block=True, name="deny"))
    allow, msg = await hooks.run_pre_tool(c, "edit_file", {"path": "x.py"})
    assert not allow
    assert "deny" in msg and "bad" in msg


async def test_pre_hook_nonblocking_allows_despite_nonzero():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"],
                        command="exit 5", block=False))
    allow, _ = await hooks.run_pre_tool(c, "anything", {})
    assert allow


async def test_pre_hook_only_matching_tools():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command="exit 1", block=True))
    allow, _ = await hooks.run_pre_tool(c, "read_file", {})
    assert allow  # no matching hook


async def test_post_hook_notes_on_nonzero():
    c = _cfg(HookConfig(event="post_tool", tools=["*"],
                        command="echo lint-warn; exit 1", name="lint"))
    notes = await hooks.run_post_tool(c, "edit_file", {"path": "x"}, "{}")
    assert notes and "lint" in notes[0] and "lint-warn" in notes[0]


async def test_post_hook_silent_on_success():
    c = _cfg(HookConfig(event="post_tool", tools=["*"], command="true"))
    notes = await hooks.run_post_tool(c, "edit_file", {}, "{}")
    assert notes == []


async def test_env_tool_path_passed():
    c = _cfg(HookConfig(event="pre_tool", tools=["edit_file"],
                        command='test "$TOOL_PATH" = "a.py"', block=True))
    ok_allow, _ = await hooks.run_pre_tool(c, "edit_file", {"path": "a.py"})
    bad_allow, _ = await hooks.run_pre_tool(c, "edit_file", {"path": "b.py"})
    assert ok_allow and not bad_allow


async def test_disabled_section_is_noop():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"], command="exit 1", block=True))
    c.hooks.enabled = False
    allow, _ = await hooks.run_pre_tool(c, "edit_file", {})
    assert allow


async def test_timeout_blocks_when_blocking():
    c = _cfg(HookConfig(event="pre_tool", tools=["*"],
                        command="sleep 5", block=True, timeout_s=0.3))
    allow, msg = await hooks.run_pre_tool(c, "edit_file", {})
    assert not allow and "timed out" in msg


async def test_none_config_is_noop():
    allow, _ = await hooks.run_pre_tool(None, "edit_file", {})
    assert allow
    assert await hooks.run_post_tool(None, "edit_file", {}, "{}") == []


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


async def test_hook_env_scrubbed(monkeypatch):
    monkeypatch.setenv("HOOKTEST_API_TOKEN", "s3cret")
    c = _cfg(HookConfig(event="pre_tool", tools=["*"], block=True,
                        command='test -z "$HOOKTEST_API_TOKEN"'))
    allow, msg = await hooks.run_pre_tool(c, "edit_file", {})
    assert allow, msg


async def test_hook_machinery_failure_denies_the_call(monkeypatch):
    """An exception while evaluating hooks is a denial, never a silent allow."""
    from agent.core import tool_calls

    async def boom(*_a, **_k):
        raise RuntimeError("hook engine broke")

    monkeypatch.setattr(hooks, "run_pre_tool", boom)
    blocked, _note = await tool_calls.pre_tool_gates("read_file", {"path": "x"}, Config())
    assert blocked and blocked.get("blocked_by_hook")
    assert "not run" in blocked["error"]


async def test_pre_turn_blocks_and_sees_prompt_and_endpoint():
    c = _cfg(HookConfig(
        event="pre_turn", block=True, name="dlp",
        command='case "$USER_PROMPT" in *ACME*) echo "customer name, $TURN_SOURCE, $LLM_BASE_URL"; exit 1;; esac'))
    c.llm.base_url = "https://cloud.example/v1"
    allow, msg = await hooks.run_pre_turn(c, "summarise the ACME contract", "http")
    assert not allow
    assert "dlp" in msg and "customer name, http, https://cloud.example/v1" in msg
    allow, _ = await hooks.run_pre_turn(c, "summarise the contract", "http")
    assert allow


async def test_pre_turn_ignores_tool_hooks_and_nonblocking_failures():
    c = _cfg(HookConfig(event="pre_tool", command="exit 1", block=True),
             HookConfig(event="pre_turn", command="exit 1", block=False))
    assert (await hooks.run_pre_turn(c, "hi"))[0]
    assert (await hooks.run_pre_turn(None, "hi"))[0]


async def test_pre_turn_timeout_refuses():
    c = _cfg(HookConfig(event="pre_turn", command="sleep 5", block=True, timeout_s=1))
    allow, msg = await hooks.run_pre_turn(c, "hi")
    assert not allow and "timed out" in msg


async def test_refused_turn_reaches_no_model_and_leaves_no_trace(tmp_path, monkeypatch):
    import agent.core.agent as agent_mod
    from agent.core.agent import Agent

    c = _cfg(HookConfig(event="pre_turn", command="echo nope; exit 1", block=True, name="gate"))
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    a = Agent(c)

    async def must_not_run(*_a, **_k):
        raise AssertionError("model was called")

    monkeypatch.setattr(agent_mod, "run_turn", must_not_run)
    before = list(a.messages)
    out = await a.chat("secret prompt")
    assert "gate" in out and "nope" in out
    assert a.messages == before
