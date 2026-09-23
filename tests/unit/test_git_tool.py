"""Unit tests for agent.tools.git.main._run_git hardening."""
from __future__ import annotations

import shutil
import subprocess

import pytest

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
    assert seen["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert seen["cmd"] == ["git", "--no-pager", "-c", "core.fsmonitor=false",
                           "-c", "log.showSignature=false", "status"]


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


def _git(tmp_path, *args):
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                        "HOME": str(tmp_path), "PATH": "/usr/bin:/bin"})


def _repo_with_secret_change(tmp_path):
    _setup_repo(tmp_path)
    _git(tmp_path, "init", "-q")
    (tmp_path / ".env").write_text("SECRET=old\n")
    (tmp_path / "app.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / ".env").write_text("SECRET=topsecret\n")
    (tmp_path / "app.py").write_text("x = 2\n")


def test_git_diff_without_path_withholds_secret_file(tmp_path):
    _repo_with_secret_change(tmp_path)
    result = gm.git_diff()
    assert "topsecret" not in result["diff"]
    assert "x = 2" in result["diff"]
    assert result["withheld"] == [".env"]


def test_git_diff_dot_path_withholds_secret_file(tmp_path):
    _repo_with_secret_change(tmp_path)
    result = gm.git_diff(path=".")
    assert "topsecret" not in result["diff"]
    assert "x = 2" in result["diff"]


def test_git_diff_staged_withholds_secret_file(tmp_path):
    _repo_with_secret_change(tmp_path)
    _git(tmp_path, "add", ".")
    result = gm.git_diff(staged=True)
    assert "topsecret" not in result["diff"]
    assert "x = 2" in result["diff"]


def test_git_diff_ignores_external_diff_driver(tmp_path):
    _repo_with_secret_change(tmp_path)
    marker = tmp_path / "pwned"
    _git(tmp_path, "config", "diff.external", f"touch {marker}")
    gm.git_diff(path="app.py")
    assert not marker.exists()


@pytest.fixture
def sandboxed(tmp_path, monkeypatch):
    """Security policy rooted at tmp_path/p — nested repos run git through it."""
    from agent.config import Config
    from agent.security import fs as sec_fs, policy as sec_policy, runner as sec_runner
    root = tmp_path / "p"
    root.mkdir()
    cfg = Config()
    cfg.tools.working_dir = str(root)
    cfg.tools.agent_dir = str(root / ".agent")
    cfg.security.require_sandbox = False
    monkeypatch.setattr(sec_runner, "_BACKEND", None)
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    sec_policy.setup(cfg)
    gm.setup(cfg)
    yield root
    sec_policy._policy = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None


def test_git_repo_param_runs_in_subrepo(sandboxed):
    sub = sandboxed / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "f.txt").write_text("a\n")
    _git(sub, "add", ".")
    _git(sub, "commit", "-qm", "subcommit")
    assert "subcommit" in gm.git_log(repo="sub")["log"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_nested_repo_config_does_not_run_on_host(sandboxed, tmp_path):
    """A nested repo's config may be the agent's (`git init` in a sandboxed
    command), so what it executes must stay in the sandbox."""
    sub = sandboxed / "sub"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "f.txt").write_text("a\n")
    _git(sub, "add", ".")
    (sub / "f.txt").write_text("b\n")
    marker = tmp_path / "pwned"      # outside the project: invisible in the sandbox
    _git(sub, "config", "core.pager", f"touch {marker}")
    (sub / ".gitattributes").write_text("*.txt filter=x\n")
    _git(sub, "config", "filter.x.clean", f"sh -c 'touch {marker}; cat'")
    gm.git_diff(repo="sub")
    gm.git_status(repo="sub")
    assert not marker.exists()


def test_git_repo_param_rejects_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    _setup_repo(root)
    result = gm.git_status(repo="..")
    assert "outside the project root" in result["error"]
