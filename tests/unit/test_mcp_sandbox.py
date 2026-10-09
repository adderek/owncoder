"""MCP stdio server jail (agent/mcp/sandbox.py) + tool allow/deny filter."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.config.models import MCPServerConfig
from agent.mcp import manager, sandbox
from agent.mcp.client import MCPClient, MCPError

from agent.tests.unit.test_mcp import _FAKE_SERVER


def _srv(tmp_path, **kw) -> MCPServerConfig:
    kw.setdefault("sandbox_home", str(tmp_path / "home"))
    kw.setdefault("sandbox_seccomp", False)
    kw.setdefault("command", "true")
    return MCPServerConfig(name="t", sandbox="bwrap", **kw)


@pytest.fixture()
def has_bwrap(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/bwrap" if n == "bwrap" else None)


# ── argv construction (no bwrap needed) ─────────────────────────────────────

def test_no_bwrap_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda n: None)
    with pytest.raises(MCPError, match="refusing to run unsandboxed"):
        sandbox.wrap_argv(_srv(tmp_path), ["x"], {})


def test_argv_isolates_by_default(tmp_path, has_bwrap):
    env = {"XDG_RUNTIME_DIR": "/run/user/1000"}
    argv, fd = sandbox.wrap_argv(_srv(tmp_path), ["node", "srv.js"], env)
    assert fd is None
    assert "--unshare-net" in argv and "--unshare-user" in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[-3:] == ["--", "node", "srv.js"]
    home = str((tmp_path / "home").resolve())
    assert env["HOME"] == home and "XDG_RUNTIME_DIR" not in env
    # no host home / project bound — only the private home
    binds = [argv[i + 1] for i, a in enumerate(argv) if a in ("--bind", "--ro-bind")]
    assert binds == ["/usr", home]


def test_network_opt_in(tmp_path, has_bwrap):
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_network=True), ["x"], {})
    assert "--unshare-net" not in argv


def test_ro_rw_binds(tmp_path, has_bwrap):
    ro, rw = tmp_path / "ro", tmp_path / "rw"
    ro.mkdir(), rw.mkdir()
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(ro)], sandbox_rw=[str(rw)]), ["x"], {})
    s = " ".join(argv)
    assert f"--ro-bind {ro} {ro}" in s and f"--bind {rw} {rw}" in s


@pytest.mark.parametrize("bad", ["/", "~"])
def test_refuses_whole_home(tmp_path, has_bwrap, bad):
    with pytest.raises(MCPError, match="exposes all of"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[bad]), ["x"], {})


def test_missing_bind_refused(tmp_path, has_bwrap):
    with pytest.raises(MCPError, match="does not exist"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(tmp_path / "nope")]), ["x"], {})


def test_secret_dir_masked(tmp_path, has_bwrap, monkeypatch):
    fake_home = tmp_path / "h"
    (fake_home / "proj" / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    # binding a dir that contains ~/.ssh is refused (it's $HOME); a sub-dir is not
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(fake_home / "proj")]), ["x"], {})
    assert f"--tmpfs {fake_home / '.ssh'}" not in " ".join(argv)
    monkeypatch.setattr(sandbox, "_SECRET_DIRS", ("proj/.ssh",))
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(fake_home / "proj")]), ["x"], {})
    assert f"--tmpfs {fake_home / 'proj' / '.ssh'}" in " ".join(argv)


def test_cwd_outside_binds_refused(tmp_path, has_bwrap):
    with pytest.raises(MCPError, match="not inside"):
        sandbox.wrap_argv(_srv(tmp_path, cwd=str(tmp_path)), ["x"], {})


def test_unknown_mode_rejected(tmp_path):
    c = MCPClient(MCPServerConfig(name="t", command="true", sandbox="docker"))
    with pytest.raises(MCPError, match="unknown sandbox"):
        c.start()


# ── tool filter ─────────────────────────────────────────────────────────────

def test_tool_filter():
    s = MCPServerConfig(tools_allow=["inspect_*", "capture_web"], tools_deny=["capture_*"])
    assert manager._tool_allowed(s, "inspect_plist")
    assert not manager._tool_allowed(s, "capture_web")       # deny wins
    assert not manager._tool_allowed(s, "observe_x")
    assert manager._tool_allowed(MCPServerConfig(), "anything")


# ── live jail (needs working unprivileged bwrap) ────────────────────────────

def _bwrap_works() -> bool:
    if not shutil.which("bwrap"):
        return False
    r = subprocess.run(["bwrap", "--unshare-user", "--ro-bind", "/usr", "/usr",
                        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                        "--symlink", "usr/bin", "/bin", "--", "/bin/true"], capture_output=True)
    return r.returncode == 0


live = pytest.mark.skipif(not _bwrap_works(), reason="unprivileged bwrap unavailable")


@live
def test_live_jail_hides_host(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("s3cret")
    s = _srv(tmp_path, sandbox_seccomp=True)
    env = {"PATH": "/usr/bin"}
    argv, fd = sandbox.wrap_argv(s, ["sh", "-c", f"cat {secret}; ls {Path.home()}/.ssh; echo ok"], env)
    try:
        r = subprocess.run(argv, env=env, pass_fds=(fd,) if fd else (), capture_output=True, text=True, timeout=20)
    finally:
        if fd:
            os.close(fd)
    assert "s3cret" not in r.stdout and r.stdout.strip().endswith("ok")


@live
def test_live_mcp_roundtrip_in_jail(tmp_path):
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    (srv_dir / "fake_mcp.py").write_text(_FAKE_SERVER)
    py = Path(sys.executable).resolve()
    ro = [str(srv_dir)]
    if not str(py).startswith("/usr/"):
        ro.append(str(Path(sys.base_prefix).resolve()))  # e.g. uv-managed interpreter
    s = _srv(tmp_path, command=str(py), args=["-I", str(srv_dir / "fake_mcp.py")],
             sandbox_ro=ro, sandbox_seccomp=True)
    c = MCPClient(s)
    try:
        c.start()
        assert c.call_tool("echo", {"text": "hi"}) == "echo: hi"
    finally:
        c.close()


# ── {project} bind + project secret masks ───────────────────────────────────

@pytest.fixture()
def project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / ".agent" / "credpool").mkdir(parents=True)
    (root / ".agent" / "credpool" / "credpool.key").write_text("KEY")
    (root / ".coord").mkdir()
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("url = https://tok@host/r")
    (root / ".env").write_text("SECRET=1")
    (root / "src").mkdir()
    (root / "src" / "app.js").write_text("ok")
    (root / "sub" / ".git").mkdir(parents=True)               # nested repo
    (root / "sub" / ".git" / "config").write_text("url = https://tok2@host/s")
    monkeypatch.setattr(sandbox, "_project", lambda: (root.resolve(), (root / ".agent").resolve()))
    from agent.security import runner
    monkeypatch.setattr(runner, "_secret_mask_paths", lambda r: [r / ".env"])
    return root.resolve()


def test_project_token_expands_and_masks(tmp_path, has_bwrap, project):
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=["{project}"]), ["x"], {})
    s = " ".join(argv)
    assert f"--ro-bind {project} {project}" in s
    assert f"--tmpfs {project / '.agent'}" in s and f"--tmpfs {project / '.coord'}" in s
    assert f"--ro-bind /dev/null {project / '.env'}" in s
    assert f"--ro-bind /dev/null {project / '.git' / 'config'}" in s
    assert f"--ro-bind /dev/null {project / 'sub' / '.git' / 'config'}" in s
    # masks come after the bind they cover
    assert s.index(f"--tmpfs {project / '.agent'}") > s.index(f"--ro-bind {project} {project}")


def test_subdir_bind_masks_only_visible(tmp_path, has_bwrap, project):
    argv, _ = sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(project / "src")]), ["x"], {})
    s = " ".join(argv)
    assert ".env" not in s and ".agent" not in s


def test_bind_containing_project_refused(tmp_path, has_bwrap, project):
    with pytest.raises(MCPError, match="contains the project root"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(project.parent)]), ["x"], {})


def test_unrelated_bind_no_scan(tmp_path, has_bwrap, project, monkeypatch):
    from agent.security import runner
    monkeypatch.setattr(runner, "_secret_mask_paths", lambda r: pytest.fail("scanned"))
    other = tmp_path / "other"
    other.mkdir()
    sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=[str(other)]), ["x"], {})


def test_incomplete_scan_fails_closed(tmp_path, has_bwrap, project, monkeypatch):
    from agent.security import runner

    def boom(r):
        raise runner.SandboxMaskIncomplete("too many files")
    monkeypatch.setattr(runner, "_secret_mask_paths", boom)
    with pytest.raises(MCPError, match="not exposing project"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=["{project}"]), ["x"], {})


def test_token_without_policy_refused(tmp_path, has_bwrap, monkeypatch):
    monkeypatch.setattr(sandbox, "_project", lambda: None)
    with pytest.raises(MCPError, match="before the security policy"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_ro=["{project}"]), ["x"], {})


@live
def test_live_project_masks(tmp_path, project):
    s = _srv(tmp_path, sandbox_ro=["{project}"], sandbox_seccomp=True)
    env = {"PATH": "/usr/bin"}
    sh = (f"cat {project}/src/app.js; echo; cat {project}/.env; ls -A {project}/.agent; "
          f"cat {project}/.git/config {project}/sub/.git/config; echo END")
    argv, fd = sandbox.wrap_argv(s, ["sh", "-c", sh], env)
    try:
        r = subprocess.run(argv, env=env, pass_fds=(fd,) if fd else (), capture_output=True, text=True, timeout=20)
    finally:
        if fd:
            os.close(fd)
    out = r.stdout
    assert out.startswith("ok") and out.rstrip().endswith("END")
    assert "SECRET" not in out and "credpool" not in out and "tok" not in out


def test_project_rw_created_and_bound_after_ro(tmp_path, has_bwrap, project):
    argv, _ = sandbox.wrap_argv(
        _srv(tmp_path, sandbox_ro=["{project}"], sandbox_rw=["{project}/.rea-out"]), ["x"], {})
    out = project / ".rea-out"
    assert out.is_dir()
    s = " ".join(argv)
    assert s.index(f"--bind {out} {out}") > s.index(f"--ro-bind {project} {project}")


def test_project_symlink_escape_refused(tmp_path, has_bwrap, project):
    target = tmp_path / "outside"
    target.mkdir()
    (project / ".rea-out").symlink_to(target)
    with pytest.raises(MCPError, match="outside the project"):
        sandbox.wrap_argv(_srv(tmp_path, sandbox_rw=["{project}/.rea-out"]), ["x"], {})
