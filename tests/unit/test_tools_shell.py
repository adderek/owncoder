"""Unit tests for agent/tools/shell.py — dangerous pattern detection and shell control."""
from __future__ import annotations

import pytest
from agent.config import Config
from agent.tools.shell import (
    _check_dangerous,
    _truncate_stream,
    run_command,
    setup as shell_setup,
    ToolDisabledError,
)
from agent.tools.shell.main import get_transcript


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


class TestRunCommandDisabled:
    def test_shell_disabled_raises(self, tmp_path):
        with pytest.raises(ToolDisabledError):
            run_command("echo hello")


class TestRunCommandEnabled:
    def test_echo(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.allow_legacy_shell = True
        shell_setup(cfg)
        r = run_command("echo hello")
        assert r["returncode"] == 0
        assert "hello" in r["stdout"]

    def test_dangerous_blocked(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        shell_setup(cfg)
        r = run_command("rm -rf /")
        assert "error" in r
        assert "requires_confirm" in r

    def test_nonzero_exit_code(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.allow_legacy_shell = True
        shell_setup(cfg)
        r = run_command("exit 42", cwd=str(tmp_path))
        assert r["returncode"] == 42

    def test_transcript_records_run(self, tmp_path):
        from agent.tools.shell.main import _transcript
        _transcript.clear()
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.allow_legacy_shell = True
        shell_setup(cfg)
        run_command("echo transcript_test")
        t = get_transcript()
        assert len(t) >= 1
        assert any("transcript_test" in entry.get("cmd", "") for entry in t)


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

    def test_run_command_stdout_truncation(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.allow_shell = True
        cfg.security.allow_legacy_shell = True
        shell_setup(cfg)
        # python -c "print('x' * N)" — cheap, no extra deps
        r = run_command("python3 -c \"print('x' * 100000)\"")
        assert r["returncode"] == 0
        assert len(r["stdout"]) < 100_000
        assert "truncated" in r
        assert r["truncated"]["stdout_chars"] >= 100_000
