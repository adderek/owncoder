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


class TestRunnerProcessGroup:
    def test_child_started_in_new_session(self, project, monkeypatch):
        # The timeout path kills the whole tree with os.killpg(proc.pid, ...),
        # which only works if the child is its own session/process-group leader.
        # Guard that run() passes start_new_session=True to Popen.
        monkeypatch.setattr(sec_runner, "_BACKEND", "none")
        monkeypatch.setattr(sec_runner, "select_backend", lambda: "none")

        seen = {}
        real_popen = subprocess.Popen

        class _FakeProc:
            pid = 999999

            def communicate(self, input=None, timeout=None):
                return b"", b""

            @property
            def returncode(self):
                return 0

        def _capture(cmd, **kwargs):
            seen.update(kwargs)
            return _FakeProc()

        monkeypatch.setattr(subprocess, "Popen", _capture)
        sec_runner.run(["echo", "hi"], timeout=5)
        assert seen.get("start_new_session") is True


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


class TestWriteDenylist:
    """H1 — write-denylist in safe_open and Rules.check_write."""

    def test_blocks_git_hook_write(self, project):
        git_dir = project / ".git"
        git_dir.mkdir()
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_open(".git/hooks/pre-commit", "w")

    def test_blocks_agent_toml_write(self, project):
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_open("agent.toml", "w")

    def test_blocks_claude_md_write(self, project):
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_open("CLAUDE.md", "w")

    def test_allows_normal_source_write(self, project):
        from agent.tools.rules.core import Rules
        rules = Rules()
        allowed, msg = rules.check_write("src/foo.py")
        assert allowed is True

    def test_check_write_blocks_git(self, project):
        from agent.tools.rules.core import Rules
        (project / ".git").mkdir()
        rules = Rules()
        allowed, msg = rules.check_write(".git/hooks/pre-commit")
        assert allowed is False
        assert "protected" in (msg or "")

    def test_check_write_blocks_agent_toml(self, project):
        from agent.tools.rules.core import Rules
        rules = Rules()
        allowed, msg = rules.check_write("agent.toml")
        assert allowed is False

    def test_check_write_blocks_claude_md(self, project):
        from agent.tools.rules.core import Rules
        rules = Rules()
        allowed, msg = rules.check_write("CLAUDE.md")
        assert allowed is False


class TestReadDenylist:
    """H5 — secret file read guard."""

    def test_blocks_env_file_read(self, project):
        (project / ".env").write_text("SECRET=abc")
        with pytest.raises(sec_fs.ReadProtected):
            sec_fs.safe_open(".env", "r")

    def test_blocks_pem_read(self, project):
        (project / "server.pem").write_text("-----BEGIN CERTIFICATE-----")
        with pytest.raises(sec_fs.ReadProtected):
            sec_fs.safe_open("server.pem", "r")

    def test_blocks_id_rsa_read(self, project):
        (project / "id_rsa").write_text("-----BEGIN RSA PRIVATE KEY-----")
        with pytest.raises(sec_fs.ReadProtected):
            sec_fs.safe_open("id_rsa", "r")

    def test_allows_normal_source_read(self, project):
        (project / "main.py").write_text("print('hello')")
        f = sec_fs.safe_open("main.py", "r")
        f.close()

    def test_check_read_blocks_env(self, project):
        from agent.tools.rules.core import Rules
        (project / ".env").write_text("SECRET=abc")
        rules = Rules()
        allowed, msg = rules.check_read(".env")
        assert allowed is False
        assert "secret" in (msg or "").lower()

    def test_check_read_allows_normal_file(self, project):
        from agent.tools.rules.core import Rules
        (project / "main.py").write_text("x=1")
        rules = Rules()
        allowed, msg = rules.check_read("main.py")
        assert allowed is True


class TestSandboxSecretMasking:
    """Shell path must not bypass the fs read-deny gate: secret files are
    masked with /dev/null inside the sandbox."""

    def test_secret_mask_paths_finds_secrets(self, project):
        (project / ".env").write_text("SECRET=abc")
        (project / "server.pem").write_text("-----BEGIN CERTIFICATE-----")
        (project / "main.py").write_text("print('hi')")
        ssh = project / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----")

        masked = {p.relative_to(project).as_posix() for p in sec_runner._secret_mask_paths(project)}
        assert ".env" in masked
        assert "server.pem" in masked
        assert ".ssh/id_rsa" in masked
        assert "main.py" not in masked

    def test_bwrap_argv_masks_secrets_with_devnull(self, project):
        (project / ".env").write_text("SECRET=abc")
        argv = sec_runner._bwrap_argv(["cat", ".env"], cwd=project, network=False)
        env_path = str(project / ".env")
        # /dev/null is ro-bound over the secret so reads return empty.
        assert "/dev/null" in argv
        assert env_path in argv
        i = argv.index(env_path)
        assert argv[i - 1] == "/dev/null"
        assert argv[i - 2] == "--ro-bind"

    def test_secret_scan_skips_noise_dirs(self, project):
        venv = project / ".venv"
        venv.mkdir()
        (venv / ".env").write_text("SECRET=should-not-be-walked")
        masked = sec_runner._secret_mask_paths(project)
        assert all(".venv" not in p.parts for p in masked)


class TestWriteDenyBasename:
    """A protected basename is denied even in a subdirectory (parity with read-deny)."""

    def test_blocks_nested_agent_toml_write(self, project):
        (project / "sub").mkdir()
        with pytest.raises(sec_fs.WriteProtected):
            sec_fs.safe_open("sub/agent.toml", "w")


class TestH6FailClosed:
    """H6 — default require_sandbox=True, fail-closed when no backend."""

    def test_default_require_sandbox_is_true(self):
        from agent.config.models import SecurityConfig
        assert SecurityConfig().require_sandbox is True

    def test_fail_closed_no_backend(self, project, monkeypatch):
        sec_policy.get().cfg.require_sandbox = True
        sec_policy.get().cfg.sandbox_backend = "auto"
        sec_runner._BACKEND = None
        monkeypatch.setattr(shutil, "which", lambda _n: None)
        with pytest.raises(sec_runner.SandboxUnavailable):
            sec_runner.run(["echo", "x"])

    def test_explicit_opt_out_runs_host(self, project):
        sec_policy.get().cfg.require_sandbox = False
        sec_policy.get().cfg.sandbox_backend = "none"
        sec_runner._BACKEND = None
        r = sec_runner.run(["echo", "opt-out"], timeout=5)
        assert r.returncode == 0
        assert "opt-out" in r.stdout


class TestH2PatchTimeout:
    """H2 — patch_file subprocess calls carry timeout."""

    def test_patch_subprocess_has_timeout(self):
        import ast, textwrap
        src = Path(__file__).parent.parent.parent / "tools/files/patch.py"
        tree = ast.parse(src.read_text())
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ]
        assert calls, "no subprocess.run calls found in patch.py"
        for call in calls:
            kw_names = [kw.arg for kw in call.keywords]
            assert "timeout" in kw_names, f"subprocess.run at line {call.lineno} missing timeout="


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
