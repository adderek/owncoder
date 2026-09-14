"""`requires_confirm` must reach the user, not dead-end in the result dict.

Tools mark a refusal with `requires_confirm` (destructive argv, confirm_create).
Nothing consumed that flag before: `run_argv(["rm", ...])` was hard-blocked with
no way to approve it. `execute_tool` now asks through the permission asker and,
on an explicit "Allow once", retries the same call with `_confirmed=True`.
"""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from agent.config import Config
from agent.core.tool_calls import execute_tool, _FakeToolCall
from agent.tools import register
import agent.security.permissions as perms

_SEEN: list[bool] = []


@register(
    "confirm_probe_tool",
    {"description": "x", "parameters": {"type": "object", "properties": {}, "required": []}},
)
def _probe(_confirmed: bool = False):
    _SEEN.append(_confirmed)
    if not _confirmed:
        return {"error": "Destructive command 'rm' requires explicit confirmation.", "requires_confirm": True}
    return {"ok": True}


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    _SEEN.clear()
    perms.reset()
    perms.set_asker(None)
    yield c
    _SEEN.clear()
    perms.reset()
    perms.set_asker(None)


def _call(cfg, args=None):
    return json.loads(asyncio.run(execute_tool(_FakeToolCall("confirm_probe_tool", args or {}), cfg)))


def test_no_asker_fails_closed(cfg):
    out = _call(cfg)
    assert out["approval"] == "denied"
    assert "requires_confirm" not in out
    assert _SEEN == [False], "the tool must not run a second time"


def test_allow_once_retries_the_same_call(cfg):
    async def asker(_q, options):
        return options[0]      # Allow once

    perms.set_asker(asker)
    out = _call(cfg)
    assert out == {"ok": True}
    assert _SEEN == [False, True]
    assert perms.session_rules() == [], "a confirmation must not become a session grant"


def test_deny_does_not_run(cfg):
    async def asker(_q, options):
        return options[1]      # Deny

    perms.set_asker(asker)
    assert _call(cfg)["approval"] == "denied"
    assert _SEEN == [False]


def test_timeout_does_not_run(cfg):
    cfg.permissions.ask_timeout_s = 0.01

    async def slow(_q, _o):
        await asyncio.sleep(5)
        return "Allow once"

    perms.set_asker(slow)
    assert _call(cfg)["approval"] == "denied"
    assert _SEEN == [False]


def test_model_cannot_preset_the_grant(cfg):
    """`_confirmed` is harness-only — a model passing it must be ignored."""
    assert _call(cfg, {"_confirmed": True})["approval"] == "denied"
    assert _SEEN == [False]


def test_shell_gate_honours_the_grant_and_nothing_else(cfg):
    from agent.tools.shell.main import _precheck_argv

    blocked, _ = _precheck_argv(["rm", "-rf", "x"], False, None)
    assert blocked and blocked.get("requires_confirm") is True

    after, _ = _precheck_argv(["rm", "-rf", "x"], False, None, confirmed=True)
    assert after is None or after.get("requires_confirm") is not True


def test_the_flag_is_not_advertised_to_the_model():
    """The model must not be able to see or set the bypass."""
    from agent.tools.shell.main import run_argv
    from agent.tools import get_schemas

    assert "_confirmed" in inspect.signature(run_argv).parameters
    schema = next(s for s in get_schemas() if s["function"]["name"] == "run_argv")
    assert "_confirmed" not in schema["function"]["parameters"]["properties"]
