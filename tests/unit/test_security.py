"""Unit tests for the security harness (agent/security/*)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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


class TestProjectVenvOnPath:
    """A project venv must become the default python3 for sandboxed commands.

    Without this, `python3` inside the sandbox is /usr/bin/python3 (the only
    interpreter mounted) and every project dependency is missing, which reads
    as "this machine needs pip install" rather than "use the project venv".
    """

    def _venv(self, project, name=".venv", pyver=None):
        bindir = project / name / "bin"
        bindir.mkdir(parents=True)
        (bindir / "python3").write_text("#!/bin/sh\n")
        if pyver:
            (project / name / "lib" / pyver / "site-packages").mkdir(parents=True)
        return bindir

    def _system_pyver(self):
        from pathlib import Path as _P
        p = _P("/usr/bin/python3")
        return p.resolve().name if p.exists() else ""

    def test_venv_bin_prepended(self, project):
        bindir = self._venv(project)
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PATH"] == f"{bindir}:/usr/bin"
        assert out["VIRTUAL_ENV"] == str(project / ".venv")

    def test_unvenved_project_untouched(self, project):
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PATH"] == "/usr/bin"
        assert "VIRTUAL_ENV" not in out

    def test_plain_venv_dir_also_found(self, project):
        bindir = self._venv(project, "venv")
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PATH"].startswith(str(bindir))

    def test_dot_venv_wins_over_venv(self, project):
        dot = self._venv(project, ".venv")
        self._venv(project, "venv")
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PATH"].startswith(str(dot))

    def test_disabled_by_config(self, project):
        self._venv(project)
        sec_policy.get().cfg.project_venv_on_path = False
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PATH"] == "/usr/bin"

    def test_empty_host_path(self, project):
        bindir = self._venv(project)
        out = sec_policy.get().env_for_child({})
        assert out["PATH"] == str(bindir)   # no trailing separator

    def test_site_packages_exposed_for_matching_python(self, project):
        # `/usr/bin/python3 script.py` (absolute path, or a #! shebang) never
        # consults PATH — PYTHONPATH is what makes the project's packages
        # importable for it.
        pyver = self._system_pyver()
        if not pyver:
            pytest.skip("no /usr/bin/python3 on this host")
        self._venv(project, pyver=pyver)
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert out["PYTHONPATH"] == str(project / ".venv" / "lib" / pyver / "site-packages")

    def test_site_packages_skipped_for_other_python(self, project):
        # A venv built on a different minor version: mixing it into the system
        # interpreter breaks compiled packages, so PATH is the only fix used.
        self._venv(project, pyver="python3.0")
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin"})
        assert "PYTHONPATH" not in out

    def test_existing_pythonpath_preserved(self, project):
        pyver = self._system_pyver()
        if not pyver:
            pytest.skip("no /usr/bin/python3 on this host")
        self._venv(project, pyver=pyver)
        out = sec_policy.get().env_for_child({"PATH": "/usr/bin", "PYTHONPATH": "/host/pp"})
        assert out["PYTHONPATH"].endswith(":/host/pp")
        assert out["PYTHONPATH"].startswith(str(project / ".venv"))


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

    def test_sys_executable_runnable_inside_sandbox(self, project):
        """Regression: a uv/venv interpreter outside /usr must still exec.

        Before the interpreter binds, web_fetch/web_search died with
        `bwrap: execvp <uv python>: No such file or directory`.
        """
        import sys as _sys
        if sec_runner.select_backend() == "none":
            pytest.skip("no functional sandbox backend")
        r = sec_runner.run([_sys.executable, "-c", "print('ok')"], timeout=15)
        assert r.returncode == 0, r.stderr
        assert "ok" in r.stdout

    def test_interpreter_paths_skip_usr_and_root(self, project):
        import sys as _sys
        root = sec_policy.get().root
        paths = sec_runner._interpreter_paths(root)
        assert all(not p.startswith("/usr") for p in paths)
        assert all(Path(p) != root and root not in Path(p).parents for p in paths)
        if not _sys.prefix.startswith("/usr"):
            assert paths

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


class TestSeccompCloneNamespace:
    def test_clone3_and_namespace_clone_blocked(self, tmp_path):
        from agent.security.seccomp_filter import _get_lib
        if _get_lib() is None:
            pytest.skip("libseccomp not available")
        repo_root = Path(__file__).resolve().parents[3]
        script = (
            "import ctypes, os\n"
            "from agent.security.seccomp_filter import build_filter_fd\n"
            "fd = build_filter_fd()\n"
            "data = os.read(fd, 1 << 20)\n"
            "os.close(fd)\n"
            "n = len(data) // 8\n"
            "class P(ctypes.Structure):\n"
            "    _fields_ = [('len', ctypes.c_ushort), ('filter', ctypes.c_void_p)]\n"
            "buf = ctypes.create_string_buffer(data)\n"
            "prog = P(n, ctypes.cast(buf, ctypes.c_void_p))\n"
            "libc = ctypes.CDLL(None, use_errno=True)\n"
            "libc.prctl(38, 1, 0, 0, 0)\n"
            "assert libc.prctl(22, 2, ctypes.byref(prog), 0, 0) == 0\n"
            "ctypes.set_errno(0)\n"
            "libc.syscall(435, 0)\n"
            "assert ctypes.get_errno() == 38, ('clone3 not blocked', ctypes.get_errno())\n"
        )
        env = dict(os.environ, PYTHONPATH=str(repo_root))
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert r.returncode == 0, r.stderr


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

    def test_blocks_top_level_ssh_files(self, project):
        # `**/.ssh/*` did not match a top-level `.ssh/` — only nested ones.
        (project / ".ssh").mkdir()
        (project / ".ssh" / "config").write_text("Host *\n")
        (project / ".ssh" / "known_hosts").write_text("github.com ssh-rsa AAA\n")
        for name in ("config", "known_hosts"):
            with pytest.raises(sec_fs.ReadProtected):
                sec_fs.safe_open(f".ssh/{name}", "r")

    def test_blocks_top_level_aws_credentials(self, project):
        (project / ".aws").mkdir()
        (project / ".aws" / "credentials").write_text("[default]\naws_access_key_id=AKIA\n")
        with pytest.raises(sec_fs.ReadProtected):
            sec_fs.safe_open(".aws/credentials", "r")

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


class TestSandboxWriteDenyOverlay:
    """Shell path must not bypass the fs write-deny gate: protected paths are
    bound read-only inside the sandbox (grants, permissions, core, fetcher)."""

    def test_write_deny_paths_finds_policy_files(self, project):
        agent_dir = project / ".agent"
        agent_dir.mkdir(exist_ok=True)   # policy.setup() already made it (scratch)
        (agent_dir / "path_grants.json").write_text("[]")
        (agent_dir / "permissions.json").write_text("{}")
        (agent_dir / "core.md").write_text("rules")
        ws = agent_dir / "web_search"
        ws.mkdir()
        (ws / "_http_fetcher.py").write_text("print('x')")

        found = {p.relative_to(project).as_posix()
                 for p in sec_runner._write_deny_paths(project)}
        assert ".agent/path_grants.json" in found
        assert ".agent/permissions.json" in found
        assert ".agent/core.md" in found
        # `prefix/**` collapses to one read-only bind of the directory.
        assert ".agent/web_search" in found

    def test_diagnostics_dir_is_readonly_but_present(self, project):
        """Readable, not forgeable: records can be read inside the sandbox but
        a contaminated model cannot rewrite or delete its own failure trail."""
        d = project / ".agent" / "diagnostics" / "failures"
        d.mkdir(parents=True)
        (d / "index.jsonl").write_text("{}\n")

        found = {p.relative_to(project).as_posix()
                 for p in sec_runner._write_deny_paths(project)}
        assert ".agent/diagnostics" in found
        argv = sec_runner._bwrap_argv(["sh", "-c", "true"], cwd=project, network=False)
        i = argv.index(str(project / ".agent" / "diagnostics"))
        assert argv[i - 1] == "--ro-bind"

    def test_broader_glob_covers_narrower_file(self, project):
        """A `prefix/**` dir bind must swallow a narrower file glob beneath it,
        or the same path is bound twice and bwrap rejects the duplicate."""
        ws = project / ".agent" / "web_search"
        ws.mkdir(parents=True)
        fetcher = ws / "_http_fetcher.py"
        fetcher.write_text("x")

        found = sec_runner._write_deny_paths(project)
        assert ws in found
        assert fetcher not in found

    def test_bwrap_argv_binds_write_deny_readonly(self, project):
        agent_dir = project / ".agent"
        agent_dir.mkdir(exist_ok=True)   # policy.setup() already made it (scratch)
        grants = agent_dir / "path_grants.json"
        grants.write_text("[]")

        argv = sec_runner._bwrap_argv(
            ["sh", "-c", "echo x > .agent/path_grants.json"],
            cwd=project, network=False)
        i = argv.index(str(grants))
        assert argv[i - 1] == "--ro-bind"
        assert argv[i - 2] != "--ro-bind-try"


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


class TestScratchDir:
    """Temp files must survive across commands and be visible to the file
    tools — the sandbox's per-command tmpfs /tmp gave neither."""

    def test_scratch_created_private(self, project):
        d = sec_policy.get().ensure_scratch()
        assert d == project / ".agent" / "tmp"
        assert d.is_dir()
        assert oct(d.stat().st_mode & 0o777) == "0o700"

    def test_temp_env_points_at_scratch(self, project):
        env = sec_policy.get().env_for_child({})
        scratch = str(sec_policy.get().scratch_dir())
        assert env["AGENT_TMP"] == scratch
        assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == scratch

    def test_host_tmpdir_does_not_leak_in(self, project):
        env = sec_policy.get().env_for_child({"TMPDIR": "/tmp"})
        assert env["TMPDIR"] == str(sec_policy.get().scratch_dir())

    def test_scratch_is_under_the_default_grant(self, project):
        """No new grant needed: what the shell writes, the file tools can read."""
        f = sec_policy.get().ensure_scratch() / "note.txt"
        f.write_text("hi")
        assert sec_fs.safe_resolve(str(f)).read_text() == "hi"

    def test_reset_scratch_empties_but_keeps_dir(self, project):
        d = sec_policy.get().ensure_scratch()
        (d / "file").write_text("x")
        (d / "sub").mkdir()
        (d / "sub" / "deep").write_text("y")
        sec_policy.reset_scratch()
        assert d.is_dir()
        assert list(d.iterdir()) == []

    def test_bwrap_binds_scratch_over_tmp(self, project, monkeypatch):
        # The fixture's root lives under /tmp; pretend otherwise, which is the
        # normal case for a real project.
        monkeypatch.setattr(sec_runner, "_under_tmp", lambda p: False)
        argv = sec_runner._bwrap_argv(["true"], cwd=project, network=False)
        i = argv.index(str(sec_policy.get().scratch_dir()))
        assert argv[i - 1] == "--bind" and argv[i + 1] == "/tmp"
        assert "--tmpfs" not in argv[i - 1:i + 2]
        # Must be set up before the root bind, or it would mask the root.
        assert i < argv.index(str(project))

    def test_bwrap_keeps_tmpfs_when_project_lives_under_tmp(self, project):
        argv = sec_runner._bwrap_argv(["true"], cwd=project, network=False)
        i = argv.index("/tmp")
        assert argv[i - 1] == "--tmpfs"

    def test_bwrap_keeps_tmpfs_when_disabled(self, project, monkeypatch):
        monkeypatch.setattr(sec_runner, "_under_tmp", lambda p: False)
        sec_policy.get().cfg.scratch_bind_tmp = False
        argv = sec_runner._bwrap_argv(["true"], cwd=project, network=False)
        i = argv.index("/tmp")
        assert argv[i - 1] == "--tmpfs"


class TestPathRequestReason:
    def test_request_without_reason_is_rejected(self, project):
        from agent.tools.request_path import request_path_access
        r = request_path_access(path="/var/tmp/x", mode="ro", reason="   ")
        assert r["status"] == "error"
        from agent.security import path_grants as pg
        assert not pg.has_pending()

    def test_reason_reaches_the_grant(self, project):
        from agent.tools.request_path import request_path_access
        from agent.security import path_grants as pg
        r = request_path_access(path="/var/tmp/x", mode="ro",
                                reason="read the crash dump the user mentioned")
        assert r["status"] == "pending"
        g = [g for g in pg.get_all() if g.state == "pending"][0]
        assert g.reason == "read the crash dump the user mentioned"

    def test_scratch_is_exempt_from_write_deny(self, project):
        """A throwaway file in the scratch is not policy, whatever it is named:
        the shell can write it, so the file tools must not refuse it."""
        d = sec_policy.get().ensure_scratch()
        for name in ("agent.toml", "AGENT.md", ".agent.ignore", "notes.txt"):
            assert not sec_fs._is_write_protected(project, d / name), name
        # …while the real config paths stay protected.
        assert sec_fs._is_write_protected(project, project / "agent.toml")
        assert sec_fs._is_write_protected(project, project / ".agent" / "path_grants.json")

    def test_scratch_not_bound_readonly_in_sandbox(self, project):
        d = sec_policy.get().ensure_scratch()
        (d / "agent.toml").write_text("x = 1")
        assert (d / "agent.toml") not in sec_runner._write_deny_paths(project)


class TestScratchSymlinkHardening:
    """The sandboxed shell can write `.agent/`, so it can swap the scratch for
    a symlink. Two verified consequences if that is trusted: bwrap binds the
    link's *target* onto /tmp (read-write access outside the root), and
    reset_scratch deletes the target's contents from the unconfined host
    process. Both must be refused."""

    def _victim(self, tmp_path):
        v = tmp_path.parent / "victim-dir"
        shutil.rmtree(v, ignore_errors=True)
        (v / "sub").mkdir(parents=True)
        (v / "precious.txt").write_text("keep me")
        (v / "sub" / "also.txt").write_text("keep me too")
        return v

    def test_symlinked_scratch_is_removed_not_followed(self, project, tmp_path):
        victim = self._victim(tmp_path)
        d = sec_policy.get().scratch_dir()
        shutil.rmtree(d)
        d.symlink_to(victim, target_is_directory=True)

        assert sec_policy.get().ensure_scratch() == d   # rebuilt as a real dir
        assert not d.is_symlink() and d.is_dir()
        assert sorted(p.name for p in victim.iterdir()) == ["precious.txt", "sub"]

    def test_reset_scratch_does_not_wipe_a_symlink_target(self, project, tmp_path):
        victim = self._victim(tmp_path)
        d = sec_policy.get().scratch_dir()
        shutil.rmtree(d)
        d.symlink_to(victim, target_is_directory=True)

        sec_policy.reset_scratch()
        assert sorted(p.name for p in victim.iterdir()) == ["precious.txt", "sub"]
        assert (victim / "sub" / "also.txt").read_text() == "keep me too"

    def test_symlinked_ancestor_disables_the_scratch(self, project, tmp_path):
        """`.agent` itself swapped: the scratch is not a link, but still lands
        outside the project. Nothing may be created, bound or deleted there."""
        victim = self._victim(tmp_path)
        agent_dir = project / ".agent"
        shutil.rmtree(agent_dir)
        agent_dir.symlink_to(victim, target_is_directory=True)

        pol = sec_policy.get()
        assert not pol.scratch_path_is_clean()
        assert pol.ensure_scratch() is None
        sec_policy.reset_scratch()
        assert sorted(p.name for p in victim.iterdir()) == ["precious.txt", "sub"]

    def test_untrusted_scratch_falls_back_to_tmpfs(self, project, tmp_path, monkeypatch):
        monkeypatch.setattr(sec_runner, "_under_tmp", lambda p: False)
        monkeypatch.setattr(sec_policy.Policy, "ensure_scratch", lambda self: None)
        argv = sec_runner._bwrap_argv(["true"], cwd=project, network=False)
        i = argv.index("/tmp")
        assert argv[i - 1] == "--tmpfs"

    def test_untrusted_scratch_leaves_temp_env_alone(self, project, monkeypatch):
        monkeypatch.setattr(sec_policy.Policy, "ensure_scratch", lambda self: None)
        env = sec_policy.get().env_for_child({"TMPDIR": "/tmp"})
        assert "AGENT_TMP" not in env
        assert env["TMPDIR"] == "/tmp"   # host value, i.e. the sandbox's own tmpfs


class TestMaskScanFailsClosed:
    """A truncated scan means some secret stayed readable and some policy file
    stayed writable inside the sandbox, with nothing downstream able to tell
    that from "the tree has no more matches". Refuse rather than pretend."""

    def _tree(self, project, n):
        d = project / "many"
        d.mkdir()
        for i in range(n):
            (d / f"f{i}.txt").write_text("x")
        (project / ".env").write_text("SECRET=1")
        return d

    def test_file_cap_refuses_the_command(self, project, monkeypatch):
        monkeypatch.setattr(sec_runner, "_SECRET_SCAN_FILE_CAP", 5)
        self._tree(project, 20)
        with pytest.raises(sec_runner.SandboxMaskIncomplete):
            sec_runner._secret_mask_paths(project)

    def test_match_cap_refuses_the_command(self, project, monkeypatch):
        monkeypatch.setattr(sec_runner, "_SECRET_MASK_MATCH_CAP", 3)
        d = project / "keys"
        d.mkdir()
        for i in range(10):
            (d / f"k{i}.pem").write_text("-----BEGIN-----")
        with pytest.raises(sec_runner.SandboxMaskIncomplete):
            sec_runner._secret_mask_paths(project)

    def test_refusal_reaches_the_tool_layer(self, project, monkeypatch):
        # Callers already refuse to run without isolation; this must ride the
        # same path instead of crashing the turn.
        assert issubclass(sec_runner.SandboxMaskIncomplete, sec_runner.SandboxUnavailable)
        monkeypatch.setattr(sec_runner, "_SECRET_SCAN_FILE_CAP", 5)
        self._tree(project, 20)
        monkeypatch.setattr(sec_runner, "select_backend", lambda: "bwrap")
        with pytest.raises(sec_runner.SandboxUnavailable):
            sec_runner._bwrap_argv(["true"], cwd=project, network=False)

    def test_opt_in_fail_open_still_runs(self, project, monkeypatch):
        monkeypatch.setattr(sec_runner, "_SECRET_SCAN_FILE_CAP", 5)
        self._tree(project, 20)
        sec_policy.get().cfg.mask_scan_fail_open = True
        # Partial list, no exception — the documented escape hatch.
        assert isinstance(sec_runner._secret_mask_paths(project), list)

    def test_normal_tree_is_unaffected(self, project):
        (project / ".env").write_text("SECRET=1")
        assert sec_runner._secret_mask_paths(project) == [project / ".env"]
