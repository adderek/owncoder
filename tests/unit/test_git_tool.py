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
