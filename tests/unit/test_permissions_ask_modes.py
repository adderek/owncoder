"""Ask-flow modes: no-deadline wait, program grants, classifier auto, approve-all."""
from __future__ import annotations

import asyncio

import pytest

import agent.classify.guard as guard
import agent.security.permissions as perms
from agent.classify.client import ClassifierUnavailable, Verdict
from agent.config import Config
from agent.config.models import PermissionRule

PY = {"argv": ["python3", "-c", "print(1)"]}


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.permissions.builtin_rules = False
    c.permissions.rules = [PermissionRule(tool="run_argv", verdict=perms.ASK)]
    perms.reset()
    perms.set_asker(None)
    guard.reset()
    yield c
    perms.reset()
    perms.set_asker(None)
    guard.reset()


def _run(coro):
    return asyncio.run(coro)


def _check(cfg, args=PY):
    return _run(perms.check("run_argv", args, cfg))


def _verdict(monkeypatch, label="safe", conf=0.9, exc=None):
    async def fake(config, tool, args_text):
        if exc:
            raise exc
        return Verdict(probe="action_risk", label=label, p=conf, confidence=conf)
    monkeypatch.setattr(guard, "verdict_for", fake)
    monkeypatch.setattr(perms, "_classifier_ready", lambda c: True)


class TestTimeout:
    def test_zero_means_no_deadline(self, cfg):
        cfg.permissions.ask_timeout_s = 0
        assert perms.ask_timeout(cfg) is None

    def test_session_override(self, cfg):
        perms.run_permissions_command(cfg, "timeout off")
        assert perms.ask_timeout(cfg) is None
        perms.run_permissions_command(cfg, "timeout 12")
        assert perms.ask_timeout(cfg) == 12.0
        perms.run_permissions_command(cfg, "timeout default")
        assert perms.ask_timeout(cfg) == 300.0

    def test_wait_reasks_without_deadline(self, cfg):
        cfg.permissions.ask_timeout_s = 0.05
        seen = []

        async def asker(_q, options):
            seen.append((list(options), perms.ask_timeout(cfg)))
            if len(seen) == 1:
                return perms._WAIT
            await asyncio.sleep(0.2)          # past the configured deadline
            return perms._ALLOW_ONCE

        perms.set_asker(asker)
        assert _check(cfg).allowed
        assert perms._WAIT in seen[0][0] and seen[0][1] == 0.05
        assert perms._WAIT not in seen[1][0] and seen[1][1] is None
        assert perms.ask_timeout(cfg) == 0.05, "no-deadline must not leak past the prompt"

    def test_wait_offered_on_confirm(self, cfg):
        opts_seen = []

        async def asker(_q, options):
            opts_seen.append(list(options))
            return perms._WAIT if len(opts_seen) == 1 else perms._ALLOW_ONCE

        perms.set_asker(asker)
        assert _run(perms.confirm_action("rm -rf build?", cfg)) is True
        assert len(opts_seen) == 2


class TestProgramGrant:
    def test_program_grant_covers_new_code(self, cfg):
        calls = []

        async def asker(_q, options):
            calls.append(options)
            return next(o for o in options if o.startswith(perms._PROGRAM_PREFIX))

        perms.set_asker(asker)
        assert _check(cfg).allowed
        assert _check(cfg, {"argv": ["python3", "-c", "import os; os.listdir()"]}).allowed
        assert len(calls) == 1
        # prefix only matches the whole program name
        perms.set_asker(None)
        assert not _check(cfg, {"argv": ["python3x", "-c", "1"]}).allowed

    def test_no_program_option_for_non_argv_tools(self, cfg):
        assert not any(o.startswith(perms._PROGRAM_PREFIX)
                       for o in perms._ask_options("web_fetch", {"url": "x"}, cfg, False))

    def test_forged_program_answer_denies(self, cfg):
        async def asker(_q, _options):
            return perms._PROGRAM_PREFIX + "bash"

        perms.set_asker(asker)
        assert not _check(cfg).allowed
        assert perms.session_rules() == []


class TestAuto:
    def test_option_hidden_without_classifier(self, cfg):
        assert perms._AUTO not in perms._ask_options("run_argv", PY, cfg, False)

    def test_auto_once_allows_safe(self, cfg, monkeypatch):
        _verdict(monkeypatch, "safe")

        async def asker(_q, options):
            return perms._AUTO
        perms.set_asker(asker)
        d = _check(cfg)
        assert d.allowed and "classifier" in d.reason

    def test_auto_once_denies_risky(self, cfg, monkeypatch):
        _verdict(monkeypatch, "destructive")

        async def asker(_q, options):
            return perms._AUTO
        perms.set_asker(asker)
        assert not _check(cfg).allowed

    def test_auto_once_denies_when_unavailable(self, cfg, monkeypatch):
        _verdict(monkeypatch, exc=ClassifierUnavailable("down"))

        async def asker(_q, options):
            return perms._AUTO
        perms.set_asker(asker)
        assert not _check(cfg).allowed

    def test_session_auto_falls_through_to_human(self, cfg, monkeypatch):
        _verdict(monkeypatch, "needs_review")
        assert "on" in perms.run_permissions_command(cfg, "auto on").lower()
        asked = []

        async def asker(_q, options):
            asked.append(1)
            return perms._DENY_ONCE
        perms.set_asker(asker)
        assert not _check(cfg).allowed and asked

    def test_session_auto_allows_safe_without_prompt(self, cfg, monkeypatch):
        _verdict(monkeypatch, "safe")
        perms.run_permissions_command(cfg, "auto on")

        async def asker(_q, _o):
            raise AssertionError("must not prompt")
        perms.set_asker(asker)
        assert _check(cfg).allowed


class TestApproveAll:
    def _arm(self, cfg):
        out = perms.run_permissions_command(cfg, "yolo")
        code = out.rsplit(" ", 1)[-1]
        return out, code

    def test_disabled_by_default(self, cfg):
        out, _ = self._arm(cfg)
        assert "disabled" in out
        assert not perms._approve_all

    def test_two_phase(self, cfg):
        cfg.permissions.allow_approve_all = True
        out, code = self._arm(cfg)
        assert "WARNING" in out and not perms._approve_all
        assert "ON" in perms.run_permissions_command(cfg, f"yolo {code}")
        assert _check(cfg).allowed                         # no asker, still allowed
        assert _run(perms.confirm_action("rm -rf x?", cfg)) is True
        perms.run_permissions_command(cfg, "yolo off")
        assert not _check(cfg).allowed

    def test_wrong_code_burns_pending(self, cfg):
        cfg.permissions.allow_approve_all = True
        _, code = self._arm(cfg)
        assert "NOT" in perms.run_permissions_command(cfg, "yolo 000000")
        assert "No pending" in perms.run_permissions_command(cfg, f"yolo {code}")
        assert not perms._approve_all

    def test_expired_code(self, cfg, monkeypatch):
        cfg.permissions.allow_approve_all = True
        _, code = self._arm(cfg)
        monkeypatch.setattr(perms, "_approve_all_pending",
                            (perms._approve_all_pending[0], 0.0))
        assert "expired" in perms.run_permissions_command(cfg, f"yolo {code}")
        assert not perms._approve_all

    def test_deny_rules_still_deny(self, cfg):
        cfg.permissions.allow_approve_all = True
        cfg.permissions.rules.insert(0, PermissionRule(tool="run_argv", match="rm *",
                                                       verdict=perms.DENY))
        _, code = self._arm(cfg)
        perms.run_permissions_command(cfg, f"yolo {code}")
        assert not _check(cfg, {"argv": ["rm", "-rf", "/"]}).allowed

    def test_reset_clears(self, cfg):
        cfg.permissions.allow_approve_all = True
        _, code = self._arm(cfg)
        perms.run_permissions_command(cfg, f"yolo {code}")
        perms.reset()
        assert not perms._approve_all


def test_project_layer_cannot_enable_approve_all():
    from agent.config.loader import _merge_permissions
    c = Config()
    _merge_permissions(c, [({"permissions": {"allow_approve_all": True}}, True)])
    assert c.permissions.allow_approve_all is False
    _merge_permissions(c, [({"permissions": {"allow_approve_all": True}}, False)])
    assert c.permissions.allow_approve_all is True
