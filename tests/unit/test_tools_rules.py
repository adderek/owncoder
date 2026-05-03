"""Unit tests for agent/tools/rules.py — rule matching and enforcement."""
from __future__ import annotations

import pytest
from agent.tools.rules import (
    PathMatcher,
    ReadonlyMatcher,
    CommandAllowlist,
    ApprovalRules,
    ApprovalRule,
    Rules,
    RulesConfig,
    BoundaryConfig,
    _split_command_segments,
)


class TestPathMatcher:
    def test_empty_matches_nothing(self):
        m = PathMatcher()
        assert not m.matches("anything.py")
        assert m.empty

    def test_glob_pattern(self):
        m = PathMatcher(["*.pyc", "__pycache__/**"])
        assert m.matches("foo.pyc")
        assert m.matches("__pycache__/bar.py")
        assert not m.matches("foo.py")

    def test_directory_pattern(self):
        m = PathMatcher([".git/**", "node_modules/**"])
        assert m.matches(".git/config")
        assert m.matches("node_modules/pkg/index.js")
        assert not m.matches("src/main.py")


class TestReadonlyMatcher:
    def test_empty(self):
        m = ReadonlyMatcher()
        matched, reason = m.matches("anything")
        assert not matched

    def test_pattern_with_reason(self):
        m = ReadonlyMatcher(
            patterns=["*.lock"],
            reasons={"*.lock": "lock files should not be modified"},
        )
        matched, reason = m.matches("package.lock")
        assert matched
        assert "lock files" in reason

    def test_pattern_without_reason(self):
        m = ReadonlyMatcher(patterns=["*.log"])
        matched, reason = m.matches("debug.log")
        assert matched
        assert reason is None


class TestCommandAllowlist:
    def test_allowed_prefix(self):
        al = CommandAllowlist(["git ", "python ", "pytest"])
        ok, msg = al.is_allowed("git status")
        assert ok
        ok, msg = al.is_allowed("python -m pytest")
        assert ok

    def test_denied(self):
        al = CommandAllowlist(["git ", "python "])
        ok, msg = al.is_allowed("rm -rf /")
        assert not ok
        assert "allowlist" in msg.lower()


class TestApprovalRules:
    def test_always_approval(self):
        rules = ApprovalRules([ApprovalRule(tool="run_command", condition="always")])
        needs, reason = rules.needs_approval("run_command", {})
        assert needs

    def test_no_rules(self):
        rules = ApprovalRules()
        needs, reason = rules.needs_approval("run_command", {})
        assert not needs

    def test_line_threshold(self):
        rules = ApprovalRules([ApprovalRule(tool="write_file", condition=">10 lines")])
        needs, _ = rules.needs_approval("write_file", {"content": "x\n" * 5})
        assert not needs
        needs, _ = rules.needs_approval("write_file", {"content": "x\n" * 20})
        assert needs

    def test_matching_pattern(self):
        rules = ApprovalRules([ApprovalRule(tool="run_command", condition="matching docker*")])
        needs, _ = rules.needs_approval("run_command", {"cmd": "docker build ."})
        assert needs
        needs, _ = rules.needs_approval("run_command", {"cmd": "git status"})
        assert not needs


class TestRulesCheckMethods:
    def test_check_read_ignored(self):
        rules = Rules(ignore=PathMatcher(["secret/**"]))
        ok, _ = rules.check_read("secret/keys.txt")
        assert not ok

    def test_check_read_allowed(self):
        rules = Rules()
        ok, _ = rules.check_read("src/main.py")
        assert ok

    def test_check_write_readonly(self):
        rules = Rules(readonly=ReadonlyMatcher(["*.lock"]))
        ok, msg = rules.check_write("yarn.lock")
        assert not ok
        assert "read-only" in msg

    def test_check_write_allowed(self):
        rules = Rules()
        ok, _ = rules.check_write("src/main.py")
        assert ok

    def test_check_command_blocked(self):
        cfg = RulesConfig(blocked_patterns=["rm -rf"])
        rules = Rules(config=cfg)
        ok, msg = rules.check_command("rm -rf /tmp")
        assert not ok

    def test_check_network_denied(self):
        boundary = BoundaryConfig(allow_network=False)
        rules = Rules(boundary=boundary)
        ok, msg = rules.check_network_command("curl http://example.com")
        assert not ok

    def test_check_network_allowed(self):
        rules = Rules()
        ok, _ = rules.check_network_command("curl http://example.com")
        assert ok


class TestSplitCommandSegments:
    def test_pipe(self):
        assert _split_command_segments("cat file | grep foo") == ["cat file", "grep foo"]

    def test_chain(self):
        assert _split_command_segments("make && make test") == ["make", "make test"]

    def test_semicolon(self):
        assert _split_command_segments("echo a; echo b") == ["echo a", "echo b"]

    def test_simple(self):
        assert _split_command_segments("ls -la") == ["ls -la"]
