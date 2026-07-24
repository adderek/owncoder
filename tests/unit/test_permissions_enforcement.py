"""Permission enforcement at the execute_tool boundary.

The policy layer only decides whether a call is *attempted*. These tests pin the
two properties that matter: a denied call never reaches the tool body, and an
`allow` verdict grants nothing the enforcement layers below already refuse.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.config import Config
from agent.config.models import PermissionRule
from agent.core.tool_calls import execute_tool, _FakeToolCall
from agent.tools import register
import agent.security.permissions as perms

_ran: list[str] = []


@register(
    "perm_probe_tool",
    {"description": "test probe", "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }},
)
def _perm_probe_tool(path: str):
    _ran.append(path)
    return {"ok": True, "path": path}


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    _ran.clear()
    perms.reset()
    perms.set_asker(None)
    yield c
    _ran.clear()
    perms.reset()
    perms.set_asker(None)


def _call(cfg, path="a.py"):
    return asyncio.run(execute_tool(_FakeToolCall("perm_probe_tool", {"path": path}), cfg))


class TestEnforcement:
    def test_allowed_call_runs(self, cfg):
        assert json.loads(_call(cfg))["ok"] is True
        assert _ran == ["a.py"]

    def test_denied_call_never_reaches_the_tool(self, cfg):
        cfg.permissions.rules = [PermissionRule(tool="perm_probe_tool", verdict=perms.DENY,
                                                reason="test denial")]
        out = json.loads(_call(cfg))
        assert out["permission_denied"] is True
        assert "test denial" in out["error"]
        assert _ran == [], "denied tool body must not execute"

    def test_ask_without_ui_denies(self, cfg):
        cfg.permissions.rules = [PermissionRule(tool="perm_probe_tool", verdict=perms.ASK)]
        assert json.loads(_call(cfg))["permission_denied"] is True
        assert _ran == []

    def test_ask_answered_allow_runs(self, cfg):
        cfg.permissions.rules = [PermissionRule(tool="perm_probe_tool", verdict=perms.ASK)]

        async def asker(_q, options):
            return options[0]   # Allow once

        perms.set_asker(asker)
        assert json.loads(_call(cfg))["ok"] is True
        assert _ran == ["a.py"]

    def test_match_narrows_to_specific_arguments(self, cfg, tmp_path):
        # `match` needs a tool with a primary argument, so this goes through a
        # real one: read_file's primary arg is `path`.
        from agent.tools.files import setup as files_setup
        from agent.security import policy as _policy

        _policy.setup(cfg)
        files_setup(cfg)
        (tmp_path / "secret.env").write_text("K=v\n")
        (tmp_path / "ok.txt").write_text("hello\n")
        cfg.permissions.rules = [PermissionRule(tool="read_file", match="*.env",
                                                verdict=perms.DENY, reason="no env files")]

        blocked = json.loads(asyncio.run(execute_tool(
            _FakeToolCall("read_file", {"path": "secret.env"}), cfg)))
        assert blocked["permission_denied"] is True

        allowed = json.loads(asyncio.run(execute_tool(
            _FakeToolCall("read_file", {"path": "ok.txt"}), cfg)))
        assert "permission_denied" not in allowed

    def test_default_deny_blocks_everything_unmatched(self, cfg):
        cfg.permissions.default = perms.DENY
        assert json.loads(_call(cfg))["permission_denied"] is True

    def test_no_config_means_no_policy(self):
        # execute_tool(config=None) is used by internal call paths that predate
        # the policy layer; they must keep working rather than fail closed on a
        # config that does not exist.
        out = asyncio.run(execute_tool(_FakeToolCall("perm_probe_tool", {"path": "x"}), None))
        assert json.loads(out)["ok"] is True


class TestNarrowingInvariant:
    """An `allow` verdict is 'no *additional* restriction' — it cannot re-open
    anything the enforcement layers below the policy layer refuse."""

    def test_allow_does_not_bypass_write_deny_globs(self, cfg, tmp_path):
        from agent.security import policy as _policy
        from agent.tools.files import setup as files_setup, write_file

        cfg.permissions.rules = [PermissionRule(tool="write_file", verdict=perms.ALLOW)]
        _policy.setup(cfg)
        files_setup(cfg)
        (tmp_path / ".git").mkdir(exist_ok=True)
        out = write_file(".git/config", "[core]\n")
        assert "error" in out, "write-deny glob must still refuse a permitted tool"

    def test_allow_does_not_bypass_airgap(self, cfg):
        from agent.security import airgap

        cfg.security.airgap = True
        cfg.permissions.rules = [PermissionRule(tool="web_fetch", verdict=perms.ALLOW)]
        assert airgap.is_enabled(cfg) is True
        # The air-gap check lives inside the network tools, below the policy
        # layer, so a permissive rule changes nothing about egress.
        assert perms.evaluate("web_fetch", {"url": "http://x"}, cfg).verdict == perms.ALLOW
        with pytest.raises(Exception):
            airgap.check_url(cfg, "https://example.com")
