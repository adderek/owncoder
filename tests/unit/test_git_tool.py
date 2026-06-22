"""Unit tests for agent.tools.git.main._run_git hardening."""
from __future__ import annotations

import subprocess

import agent.tools.git.main as gm


def test_run_git_times_out_returns_124(monkeypatch, tmp_path):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=k.get("timeout", 30))

    monkeypatch.setattr(subprocess, "run", _boom)
    out, err, rc = gm._run_git("log", cwd=str(tmp_path), timeout=5)
    assert rc == 124
    assert out == ""
    assert "timed out" in err


def test_run_git_passes_timeout_and_noninteractive_env(monkeypatch, tmp_path):
    seen = {}

    class _Result:
        stdout, stderr, returncode = "ok", "", 0

    def _capture(cmd, **kwargs):
        seen.update(kwargs)
        seen["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(subprocess, "run", _capture)
    out, err, rc = gm._run_git("status", cwd=str(tmp_path))
    assert rc == 0 and out == "ok"
    assert seen["timeout"] == 30.0
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["cmd"] == ["git", "status"]


def _setup_repo(tmp_path):
    from agent.config import Config
    from agent.config.models import ToolsConfig
    gm.setup(Config(tools=ToolsConfig(working_dir=str(tmp_path))))


def test_git_blame_refuses_secret_file(tmp_path):
    _setup_repo(tmp_path)
    (tmp_path / ".env").write_text("SECRET=abc123\n")
    result = gm.git_blame(".env")
    assert "error" in result
    assert "blame" not in result


def test_git_diff_refuses_secret_file(tmp_path):
    _setup_repo(tmp_path)
    (tmp_path / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----\n")
    result = gm.git_diff(path="id_rsa")
    assert "error" in result
    assert "diff" not in result


def test_git_blame_allows_normal_file(tmp_path):
    # Non-secret path must not be blocked by the read-deny guard (it may still
    # fail for other git reasons, but not with the protected-secret error).
    _setup_repo(tmp_path)
    result = gm.git_blame("src/main.py")
    assert result.get("error") != gm._READ_PROTECTED_ERR
