"""Unit tests for the security harness (agent/security/*)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.config import Config
from agent.security import fs as sec_fs
from agent.security import policy as sec_policy
from agent.security import runner as sec_runner


@pytest.fixture(autouse=True)
def _reset_backend():
    sec_runner._BACKEND = None
    sec_runner._DEGRADED_WARNING_SHOWN = False
    yield
    sec_runner._BACKEND = None
    sec_runner._DEGRADED_WARNING_SHOWN = False


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.security.require_sandbox = False  # allow "none" backend in CI
    sec_policy.setup(cfg)
    sec_fs.init_root_pin()
    yield tmp_path
    # Teardown: clear policy + pin so unrelated tests later don't see our
    # tmp_path as the project root.
    sec_policy._policy = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None


class TestFsGate:
    def test_resolves_relative_inside_root(self, project):
        p = sec_fs.safe_resolve("hello.txt")
        assert str(p).startswith(str(project))

    def test_rejects_absolute_outside(self, project):
        with pytest.raises(sec_fs.PathEscape):
            sec_fs.safe_resolve("/etc/passwd")

    def test_rejects_dotdot_escape(self, project):
        with pytest.raises(sec_fs.PathEscape):
            sec_fs.safe_resolve("../../../etc/passwd")

    def test_rejects_symlink_to_outside(self, project):
        victim = project.parent / "victim.txt"
        victim.write_text("secret")
        link = project / "link"
        os.symlink(victim, link)
        with pytest.raises((sec_fs.PathEscape, sec_fs.SymlinkDenied)):
            sec_fs.safe_resolve("link")

    def test_rejects_symlink_dir_component(self, project):
        real_dir = project / "real"
        real_dir.mkdir()
        (real_dir / "inside.txt").write_text("ok")
        link = project / "via"
        os.symlink(real_dir, link)
        with pytest.raises(sec_fs.SymlinkDenied):
            sec_fs.safe_resolve("via/inside.txt")

    def test_allows_nonexistent_child(self, project):
        # Writing a new file under root should resolve OK.
        p = sec_fs.safe_resolve("new/file.txt")
        assert str(p).startswith(str(project))


class TestEnvScrub:
    def test_denies_token_vars(self, project):
        env_in = {
            "PATH": "/usr/bin",
            "GITHUB_TOKEN": "abc",
            "AWS_SECRET_ACCESS_KEY": "xyz",
            "MY_KEY": "leak",
            "HOME": "/root",
            "FOO": "bar",
        }
        out = sec_policy.get().env_for_child(env_in)
        assert "GITHUB_TOKEN" not in out
        assert "AWS_SECRET_ACCESS_KEY" not in out
        assert "MY_KEY" not in out
        assert "FOO" not in out  # not on allow list
        assert out["PATH"] == "/usr/bin"
        assert out["HOME"] == "/root"  # in allow list


@pytest.mark.skipif(
    shutil.which("bwrap") is None and shutil.which("firejail") is None,
    reason="no sandbox backend installed — skipping runner test",
)
class TestRunnerSandboxed:
    def test_echo_inside_sandbox(self, project):
        backend = sec_runner.select_backend()
        if backend == "none":
            pytest.skip("no functional sandbox backend on this host (probe failed)")
        r = sec_runner.run(["echo", "hi"], timeout=5)
        assert r.returncode == 0
        assert "hi" in r.stdout
        assert r.backend in ("bwrap", "firejail")

    def test_network_off_by_default(self, project):
        if sec_runner.select_backend() == "none":
            pytest.skip("no functional sandbox backend — network test requires real isolation")
        r = sec_runner.run(
            ["sh", "-c", "getent hosts example.com || exit 7"],
            timeout=5,
        )
        assert r.returncode != 0

    def test_seccomp_blocks_unshare(self, project):
        if sec_runner.select_backend() != "bwrap":
            pytest.skip("seccomp filter only applied with bwrap backend")
        from agent.security.seccomp_filter import build_filter_fd
        if build_filter_fd() is None:
            pytest.skip("libseccomp not available")
        r = sec_runner.run(
            ["sh", "-c", "unshare -r 2>&1; echo rc:$?"],
            timeout=5,
        )
        assert "Operation not permitted" in r.stdout or "rc:1" in r.stdout

    def test_seccomp_blocks_userfaultfd(self, project):
        """userfaultfd needs no capabilities but is blocked by seccomp filter."""
        if sec_runner.select_backend() != "bwrap":
            pytest.skip("seccomp filter only applied with bwrap backend")
        from agent.security.seccomp_filter import build_filter_fd
        if build_filter_fd() is None:
            pytest.skip("libseccomp not available")
        # syscall nr 323 = userfaultfd on x86-64; -1 return means EPERM from seccomp
        r = sec_runner.run(
            ["python3", "-c",
             "import ctypes, ctypes.util; libc=ctypes.CDLL(None); "
             "ret=libc.syscall(323,0); import ctypes as c; "
             "print('uffd:', ret)"],
            timeout=5,
        )
        assert "uffd: -1" in r.stdout

    def test_seccomp_allows_normal_commands(self, project):
        if sec_runner.select_backend() != "bwrap":
            pytest.skip("seccomp filter only applied with bwrap backend")
        r = sec_runner.run(["sh", "-c", "echo ok; ls /usr/bin | head -3"], timeout=5)
        assert r.returncode == 0
        assert "ok" in r.stdout


class TestRunnerHostFallback:
    def test_runs_without_backend_when_allowed(self, project):
        # Force "none" backend.
        sec_policy.get().cfg.sandbox_backend = "none"
        sec_runner._BACKEND = None
        r = sec_runner.run(["echo", "no-sandbox"], timeout=5)
        assert r.returncode == 0
        assert "no-sandbox" in r.stdout
        assert r.backend == "none"

    def test_require_sandbox_raises_when_none_available(self, project, monkeypatch):
        sec_policy.get().cfg.require_sandbox = True
        sec_policy.get().cfg.sandbox_backend = "auto"
        sec_runner._BACKEND = None
        monkeypatch.setattr(shutil, "which", lambda _n: None)
        with pytest.raises(sec_runner.SandboxUnavailable):
            sec_runner.run(["echo", "x"])

    def test_cwd_outside_root_rejected(self, project):
        sec_policy.get().cfg.sandbox_backend = "none"
        sec_runner._BACKEND = None
        with pytest.raises(ValueError):
            sec_runner.run(["echo", "x"], cwd="/tmp")


class TestAuditLog:
    def test_run_emits_audit(self, project):
        sec_policy.get().cfg.sandbox_backend = "none"
        sec_runner._BACKEND = None
        sec_runner.run(["echo", "secret-stdout-marker"], timeout=5)
        log = project / ".agent" / "audit.jsonl"
        assert log.exists()
        content = log.read_text()
        lines = content.splitlines()
        assert any('"run.start"' in l for l in lines)
        assert any('"run.end"' in l for l in lines)
        # argv is logged, but the command's stdout must not be. `echo secret-stdout-marker`
        # produces only the marker on stdout; anything beyond the argv mention means leakage.
        # argv appears in run.start and run.end events (2 lines). The
        # command's stdout must not appear — only its sha256.
        assert content.count("secret-stdout-marker") == 2
        assert "stdout_sha256" in content
