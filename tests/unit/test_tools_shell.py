"""Unit tests for agent/tools/shell.py — dangerous pattern detection and shell control."""
from __future__ import annotations

import pytest
from agent.config import Config
from agent.tools.shell import (
    _check_dangerous,
    _truncate_stream,
    run_shell_line,
    setup as shell_setup,
    ToolDisabledError,
)
from agent.tools.shell.main import get_transcript, run_argv


@pytest.fixture(autouse=True)
def _setup_shell(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.allow_shell = False
    shell_setup(cfg)
    yield


class TestCheckDangerous:
    @pytest.mark.parametrize("cmd", [
        "rm -rf /",
        "rm -fr /tmp/stuff",
        "sudo apt install foo",
        "dd if=/dev/zero of=/dev/sda",
        "shutdown -h now",
        "reboot",
        "FOO=1 rm -rf /tmp/x",
        "ls; rm -rf /tmp/x",
    ])
    def test_dangerous_detected(self, cmd):
        assert _check_dangerous(cmd) is not None

    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "cat file.txt",
        "python script.py",
        "git status",
        "echo hello",
        "grep -r pattern .",
    ])
    def test_safe_commands(self, cmd):
        assert _check_dangerous(cmd) is None


class TestShellDisabled:
    def test_shell_disabled_raises(self, tmp_path):
        with pytest.raises(ToolDisabledError):
            run_shell_line("echo hello")


def _enabled_cfg(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.allow_shell = True
    cfg.security.require_sandbox = False  # allow "none" backend in CI
    shell_setup(cfg)
    return cfg


class TestRunShellLine:
    def test_simple_command_runs_as_argv(self, tmp_path):
        _enabled_cfg(tmp_path)
        r = run_shell_line("echo hello")
        assert r["returncode"] == 0
        assert "hello" in r["stdout"]

    def test_shell_operators_run_via_sh(self, tmp_path):
        _enabled_cfg(tmp_path)
        r = run_shell_line("echo hello | tr a-z A-Z")
        assert r["returncode"] == 0
        assert "HELLO" in r["stdout"]

    def test_dangerous_blocked(self, tmp_path):
        _enabled_cfg(tmp_path)
        r = run_shell_line("rm -rf /")
        assert "error" in r
        assert "requires_confirm" in r

    def test_nonzero_exit_code(self, tmp_path):
        _enabled_cfg(tmp_path)
        r = run_shell_line("exit 42; true", cwd=str(tmp_path))
        assert r["returncode"] == 42

    def test_transcript_records_run(self, tmp_path):
        from agent.tools.shell.main import _transcript
        _transcript.clear()
        _enabled_cfg(tmp_path)
        run_shell_line("echo transcript_test")
        t = get_transcript()
        assert any("transcript_test" in " ".join(entry.get("argv", [])) for entry in t)


class TestNetworkPrecheck:
    """`network=false` must be enforced, not merely recorded.

    The precheck runs unconditionally, so a boundary that forbids network
    blocks the command even on the 'none' backend, where there is no network
    namespace to unshare and egress would otherwise just happen.
    """

    def _cfg(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.require_sandbox = False
        shell_setup(cfg)
        return cfg

    def test_network_command_blocked_when_boundary_denies(self, tmp_path):
        from agent.tools.rules import BoundaryConfig, Rules
        from agent.tools.rules.core import get_rules, set_rules

        self._cfg(tmp_path)
        prev = get_rules()
        set_rules(Rules(boundary=BoundaryConfig(allow_network=False)))
        try:
            r = run_argv(["curl", "http://example.com"])
        finally:
            set_rules(prev)
        assert "error" in r
        assert "Network access denied" in r["error"]
    """Shell operators must be detected so run_shell_line hands them to `sh -c`
    — input redirects must mirror output redirects."""

    @pytest.mark.parametrize("cmd", [
        "grep foo < input.txt",   # space-separated input redirect (regression)
        "cat <a",
        "echo hi > out.txt",
        "sort < a > b",
        "ls | wc",
        "echo `whoami`",
        "echo $(date)",
    ])
    def test_shell_operators_not_translated(self, cmd):
        from agent.tools.shell.main import _try_translate_to_argv
        assert _try_translate_to_argv(cmd) is None

    @pytest.mark.parametrize("cmd,expected", [
        ("python3 script.py arg", ["python3", "script.py", "arg"]),
        ("grep -n foo bar.py", ["grep", "-n", "foo", "bar.py"]),
    ])
    def test_simple_commands_translated(self, cmd, expected):
        from agent.tools.shell.main import _try_translate_to_argv
        assert _try_translate_to_argv(cmd) == expected


class TestTruncateStream:
    def test_short_passes_through(self):
        text, trunc = _truncate_stream("hello")
        assert text == "hello"
        assert trunc is False

    def test_long_truncated_with_marker(self):
        big = "A" * 50_000
        text, trunc = _truncate_stream(big, cap=1_000)
        assert trunc is True
        assert len(text) < 50_000
        assert "truncated" in text
        assert text.startswith("A")
        assert text.endswith("A")


class TestRunArgvNetworkGuard:
    """run_argv(network=True) must be blocked when security.network != 'on'."""

    def test_network_true_blocked_when_security_network_off(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.network = "off"
        shell_setup(cfg)
        result = run_argv(["curl", "https://example.com"], network=True)
        assert "error" in result
        assert "network=true blocked" in result["error"]

    def test_network_false_not_blocked_by_policy(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.network = "off"
        shell_setup(cfg)
        # network=False is the default — guard must not fire
        result = run_argv(["echo", "hello"], network=False)
        assert "network=true blocked" not in result.get("error", "")

    def test_network_true_allowed_when_security_network_on(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.network = "on"
        shell_setup(cfg)
        result = run_argv(["echo", "hello"], network=True)
        # Guard must not reject — downstream may fail for other reasons but not the guard
        assert "network=true blocked" not in result.get("error", "")

    def test_run_argv_dangerous_blocked(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        shell_setup(cfg)
        result = run_argv(["rm", "-rf", "foo"])
        assert result.get("requires_confirm") is True
        assert "rm" in result.get("error", "").lower() or "destructive" in result.get("error", "").lower()

    def test_run_argv_sudo_blocked(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        shell_setup(cfg)
        result = run_argv(["sudo", "apt", "install", "something"])
        assert result.get("requires_confirm") is True

    def test_run_argv_stdout_truncation(self, tmp_path):
        _enabled_cfg(tmp_path)
        # python -c "print('x' * N)" — cheap, no extra deps
        r = run_argv(["python3", "-c", "print('x' * 100000)"])
        assert r["returncode"] == 0
        assert len(r["stdout"]) < 100_000
        assert "truncated" in r
        assert r["truncated"]["stdout_chars"] >= 100_000
