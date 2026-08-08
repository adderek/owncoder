"""Unit tests for security.permissions — the allow/ask/deny policy layer."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.config import Config
from agent.config.models import PermissionRule
import agent.security.permissions as perms


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    perms.reset()
    perms.set_asker(None)
    yield c
    perms.reset()
    perms.set_asker(None)


def _rule(tool, verdict, match="", reason=""):
    return PermissionRule(tool=tool, match=match, verdict=verdict, reason=reason)


def _run(coro):
    return asyncio.run(coro)


class TestEvaluate:
    def test_default_allow_keeps_current_behavior(self, cfg):
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.ALLOW

    def test_default_applies_when_no_rule_matches(self, cfg):
        cfg.permissions.default = perms.DENY
        cfg.permissions.rules = [_rule("web_fetch", perms.ALLOW)]
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.DENY

    def test_first_match_wins(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY), _rule("run_argv", perms.ALLOW)]
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.DENY

    def test_tool_glob_matches(self, cfg):
        cfg.permissions.rules = [_rule("web_*", perms.DENY)]
        assert perms.evaluate("web_fetch", {"url": "http://x"}, cfg).verdict == perms.DENY
        assert perms.evaluate("read_file", {"path": "a.py"}, cfg).verdict == perms.ALLOW

    def test_argv_list_matched_as_command_line(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK, "git push*")]
        d = perms.evaluate("run_argv", {"argv": ["git", "push", "origin"]}, cfg)
        assert d.verdict == perms.ASK
        assert perms.evaluate("run_argv", {"argv": ["git", "status"]}, cfg).verdict == perms.ALLOW

    def test_path_glob_matched(self, cfg):
        cfg.permissions.rules = [_rule("write_file", perms.DENY, "deploy/*")]
        assert perms.evaluate("write_file", {"path": "deploy/prod.yaml"}, cfg).verdict == perms.DENY
        assert perms.evaluate("write_file", {"path": "src/a.py"}, cfg).verdict == perms.ALLOW

    def test_regex_match_prefix(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY, r"re:^git\s+push")]
        assert perms.evaluate("run_argv", {"argv": ["git", "push"]}, cfg).verdict == perms.DENY

    def test_rule_with_match_skips_call_without_that_arg(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY, "git*")]
        assert perms.evaluate("run_argv", {}, cfg).verdict == perms.ALLOW

    def test_session_rule_takes_precedence(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY)]
        perms.add_session_rule("run_argv", "", perms.ALLOW)
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.ALLOW

    def test_clear_session_rules(self, cfg):
        perms.add_session_rule("run_argv", "", perms.ALLOW)
        assert perms.clear_session_rules() == 1
        assert perms.session_rules() == []


class TestBuiltinBaseline:
    """The default rule set. Before it, `rules = []` + `default = "allow"` meant
    the policy layer never asked about anything."""

    @pytest.mark.parametrize("argv", [
        ["git", "push", "--force", "origin", "main"],
        ["git", "push", "-f"],
        ["git", "push", "--force-with-lease"],
        ["git", "push", "--mirror", "origin"],
        ["git", "push", "origin", "--delete", "old-branch"],
        ["git", "push", "origin", ":old-branch"],
        ["git", "filter-branch", "--all"],
        ["git", "clean", "-fdx"],
        ["git", "config", "core.hooksPath", ".githooks"],
        ["sudo", "systemctl", "restart", "nginx"],
        ["ssh", "box", "uptime"],
        ["curl", "-X", "POST", "https://example.invalid/x"],
        ["rm", "-rf", "build"],
        ["rm", "-r", "-f", "build"],
        ["npm", "publish"],
        ["gh", "release", "create", "v1"],
        ["terraform", "apply"],
        ["kubectl", "delete", "pod", "x"],
        ["crontab", "-e"],
        ["/usr/bin/git", "push", "--force"],
        ["bash", "-c", "git push --force origin main"],
    ])
    def test_irreversible_or_outbound_calls_ask(self, cfg, argv):
        assert perms.evaluate("run_argv", {"argv": argv}, cfg).verdict == perms.ASK

    @pytest.mark.parametrize("argv", [
        ["ls", "-la"],
        ["git", "status"],
        ["git", "push", "origin", "main"],     # ordinary push: not destructive
        ["git", "commit", "-m", "x"],
        ["git", "reset", "--hard"],            # reflog + checkpoint journal recover it
        ["git", "clean", "-n"],                # dry run
        ["pytest", "-q"],
        ["npm", "install"],
        ["npm", "run", "build"],
        ["rm", "stale.txt"],
        ["rm", "-r", "build"],                 # recursive but not forced
        ["grep", "-rf", "patterns", "src"],    # -rf on a tool that is not rm
    ])
    def test_everyday_calls_are_not_asked_about(self, cfg, argv):
        """Alarm fatigue is the failure mode: a baseline that fires on routine
        work teaches people to approve without reading."""
        assert perms.evaluate("run_argv", {"argv": argv}, cfg).verdict == perms.ALLOW

    def test_scheduling_and_command_deletion_ask(self, cfg):
        assert perms.evaluate("schedule_task", {"spec": "@daily"}, cfg).verdict == perms.ASK
        assert perms.evaluate("delete_command", {"name": "deploy"}, cfg).verdict == perms.ASK

    def test_reading_secrets_is_left_to_the_fs_gate(self, cfg):
        """Not in the baseline on purpose: read_deny_globs already refuses, and a
        prompt for an already-blocked call is noise."""
        assert perms.evaluate("read_file", {"path": ".env"}, cfg).verdict == perms.ALLOW

    def test_configured_rules_win_over_the_baseline(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ALLOW, "git push*")]
        d = perms.evaluate("run_argv", {"argv": ["git", "push", "--force"]}, cfg)
        assert d.verdict == perms.ALLOW

    def test_baseline_can_be_turned_off(self, cfg):
        cfg.permissions.builtin_rules = False
        assert perms.evaluate("run_argv", {"argv": ["git", "push", "--force"]},
                              cfg).verdict == perms.ALLOW

    def test_baseline_rules_are_all_valid(self, cfg):
        for rule in perms.builtin_rules():
            perms.validate_rule(rule, label="builtin")
            assert rule.verdict == perms.ASK, "baseline must ask, never deny"
            assert rule.reason, f"{rule.tool}: baseline rule with no reason"
            assert rule.origin == "builtin"

    def test_baseline_sits_last_in_precedence(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY, "git*")]
        rules = perms.active_rules(cfg)
        builtin_at = [i for i, r in enumerate(rules) if r.origin == "builtin"]
        configured_at = [i for i, r in enumerate(rules) if r.origin == "config"]
        assert min(builtin_at) > max(configured_at)

    def test_unanswerable_asks_reports_the_baseline(self, cfg):
        """`agent run` warns up front instead of failing at the call."""
        perms.set_asker(None)
        assert perms.unanswerable_asks(cfg)


class TestBypasses:
    def test_security_suite_internal_calls_bypass(self, cfg):
        cfg.permissions.default = perms.DENY
        d = perms.evaluate("run_argv", {"argv": ["ls"]}, cfg, internal_security=True)
        assert d.verdict == perms.ALLOW

    def test_quarantined_side_is_not_consulted(self, cfg):
        cfg.permissions.default = perms.DENY
        cfg.runtime_quarantined = True
        assert perms.evaluate("web_fetch", {"url": "http://x"}, cfg).verdict == perms.ALLOW


class TestValidation:
    def test_missing_tool_is_rejected(self):
        with pytest.raises(perms.PermissionConfigError):
            perms.validate_rule(_rule("", perms.ALLOW))

    def test_unknown_verdict_is_rejected(self):
        with pytest.raises(perms.PermissionConfigError):
            perms.validate_rule(_rule("run_argv", "maybe"))

    def test_match_on_tool_without_primary_arg_is_rejected(self):
        with pytest.raises(perms.PermissionConfigError):
            perms.validate_rule(_rule("git_status", perms.ASK, "anything"))

    def test_regex_on_path_arg_is_rejected(self):
        with pytest.raises(perms.PermissionConfigError):
            perms.validate_rule(_rule("write_file", perms.DENY, "re:^deploy"))

    def test_bad_regex_is_rejected(self):
        with pytest.raises(perms.PermissionConfigError):
            perms.validate_rule(_rule("run_argv", perms.DENY, "re:[unclosed"))

    def test_bad_default_is_rejected(self, cfg):
        cfg.permissions.default = "sometimes"
        with pytest.raises(perms.PermissionConfigError):
            perms.validate(cfg)

    def test_valid_rules_pass(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK, "git push*"),
                                 _rule("write_file", perms.DENY, "deploy/*")]
        perms.validate(cfg)


class TestAskFlow:
    def test_ask_without_asker_denies(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).verdict == perms.DENY

    def test_allow_once_does_not_stick(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]

        async def asker(_q, options):
            return options[0]   # Allow once

        perms.set_asker(asker)
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).allowed
        assert perms.session_rules() == []

    def test_allow_for_session_sticks(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]
        calls = []

        async def asker(_q, options):
            calls.append(1)
            return options[1]   # Allow for session

        perms.set_asker(asker)
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).allowed
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).allowed
        assert len(calls) == 1, "second identical call must not re-prompt"

    def test_deny_for_session_sticks(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]

        async def asker(_q, options):
            return options[3]   # Deny for session

        perms.set_asker(asker)
        assert not _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).allowed
        perms.set_asker(None)   # a re-prompt would now deny anyway; rule must decide
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.DENY

    def test_timeout_denies(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]
        cfg.permissions.ask_timeout_s = 0.01

        async def slow(_q, _o):
            await asyncio.sleep(5)
            return "Allow once"

        perms.set_asker(slow)
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).verdict == perms.DENY

    def test_asker_exception_denies(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]

        async def boom(_q, _o):
            raise RuntimeError("ui gone")

        perms.set_asker(boom)
        assert _run(perms.check("run_argv", {"argv": ["ls"]}, cfg)).verdict == perms.DENY

    def test_question_is_redacted(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK)]
        args = {"argv": ["curl", "-H", "Authorization: Bearer sk-abcdef1234567890abcdef"]}
        d = perms.evaluate("run_argv", args, cfg)
        assert "sk-abcdef1234567890abcdef" not in perms.format_question("run_argv", args, d, cfg)


class TestFileRules:
    def test_saved_rule_is_reloaded_and_wins_over_config(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ALLOW)]
        perms.save_file_rule(cfg, PermissionRule(tool="run_argv", verdict=perms.DENY))
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.DENY

    def test_corrupt_file_is_ignored_not_fatal(self, cfg):
        path = perms.rules_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert perms.load_file_rules(cfg) == 0
        assert perms.evaluate("run_argv", {"argv": ["ls"]}, cfg).verdict == perms.ALLOW

    def test_invalid_rule_in_file_is_skipped(self, cfg):
        path = perms.rules_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rules": [
            {"tool": "run_argv", "verdict": "nonsense"},
            {"tool": "web_fetch", "verdict": "deny"},
        ]}), encoding="utf-8")
        assert perms.load_file_rules(cfg) == 1

    def test_rules_file_is_write_denied_to_agent_tools(self):
        from agent.security.fs import _DEFAULT_WRITE_DENY_GLOBS
        assert ".agent/permissions.json" in _DEFAULT_WRITE_DENY_GLOBS


class TestCommand:
    def test_list_renders(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.ASK, "git push*", "publishes")]
        out = perms.run_permissions_command(cfg, "list")
        assert "run_argv" in out and "git push*" in out

    def test_add_writes_durable_rule(self, cfg):
        out = perms.run_permissions_command(cfg, "add deny write_file deploy/*")
        assert "Added" in out
        assert perms.evaluate("write_file", {"path": "deploy/x"}, cfg).verdict == perms.DENY

    def test_add_rejects_bad_verdict(self, cfg):
        assert "Rejected" in perms.run_permissions_command(cfg, "add sometimes run_argv")

    def test_default_change(self, cfg):
        perms.run_permissions_command(cfg, "default deny")
        assert cfg.permissions.default == perms.DENY

    def test_clear_drops_session_rules(self, cfg):
        perms.add_session_rule("run_argv", "", perms.ALLOW)
        assert "1" in perms.run_permissions_command(cfg, "clear")


class TestDenialResult:
    def test_denial_is_structured_and_visible(self, cfg):
        cfg.permissions.rules = [_rule("run_argv", perms.DENY, reason="no shell")]
        d = perms.evaluate("run_argv", {"argv": ["ls"]}, cfg)
        out = perms.denial_result("run_argv", d)
        assert out["permission_denied"] is True
        assert "no shell" in out["error"]
        assert out["rule"]["tool"] == "run_argv"


class TestProjectLayerNarrowing:
    """A cloned repo's config is untrusted input: it may restrict the agent,
    never loosen it. The project layer is whatever load_config gets as
    extra_path — i.e. the config that ships inside the repo."""

    def _load(self, tmp_path, body):
        from agent.config import load_config
        p = tmp_path / "agent.toml"
        p.write_text(body, encoding="utf-8")
        return load_config(p)

    def test_project_allow_rule_is_dropped(self, tmp_path):
        c = self._load(tmp_path, """
[permissions]
default = "deny"

[[permissions.rules]]
tool = "run_argv"
verdict = "allow"
""")
        assert [r.tool for r in c.permissions.rules] == []
        assert c.permissions.default == perms.DENY

    def test_project_deny_rule_is_kept(self, tmp_path):
        c = self._load(tmp_path, """
[[permissions.rules]]
tool = "web_fetch"
verdict = "deny"
reason = "no egress"
""")
        assert [r.verdict for r in c.permissions.rules] == [perms.DENY]
        assert c.permissions.rules[0].origin == "project"

    def test_project_cannot_loosen_default(self, tmp_path):
        c = self._load(tmp_path, """
[permissions]
default = "allow"
""")
        # Ignored, so the built-in default stands rather than a repo-chosen one.
        assert c.permissions.default == perms.ALLOW

    def test_malformed_rule_fails_the_load(self, tmp_path):
        with pytest.raises(perms.PermissionConfigError):
            self._load(tmp_path, """
[[permissions.rules]]
tool = "git_status"
match = "anything"
verdict = "deny"
""")
