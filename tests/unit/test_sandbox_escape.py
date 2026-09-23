"""Regression tests: ways a compromised agent could reach host execution."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from agent.config import Config
from agent.security import fs as sec_fs
from agent.security import policy as sec_policy
from agent.security import runner as sec_runner


@pytest.fixture
def project(tmp_path, monkeypatch):
    sec_runner._BACKEND = None
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    root = tmp_path / "p"
    root.mkdir()
    cfg = Config()
    cfg.tools.working_dir = str(root)
    cfg.tools.agent_dir = str(root / ".agent")
    cfg.security.require_sandbox = False
    sec_policy.setup(cfg)
    sec_fs.init_root_pin()
    yield root
    sec_policy._policy = None
    sec_runner._BACKEND = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None


def _sh(cmd: str, root: Path):
    return sec_runner.run(["sh", "-c", cmd], cwd=str(root), timeout=10)


class TestProtectedNamesCannotBeCreated:
    """`--ro-bind-try` skips a missing path, so the shell could create it."""

    @pytest.mark.parametrize("name", [
        "agent.toml", "agent.yaml", "agent.yml", ".agent.config",
        ".agent.approve", ".agent.sandbox",
    ])
    def test_file_is_quarantined(self, project, name):
        r = _sh(f"echo planted > {name}", project)
        assert not (project / name).exists()
        assert r.returncode != 0
        assert "protected path" in r.stderr
        moved = list((project / ".agent" / "quarantine" / "sandbox").iterdir())
        assert len(moved) == 1 and moved[0].read_text() == "planted\n"

    def test_dir_is_quarantined(self, project):
        r = _sh("mkdir -p .claude && echo '{}' > .claude/settings.json", project)
        assert not (project / ".claude").exists()
        assert r.returncode != 0

    def test_existing_file_is_left_alone(self, project):
        (project / "agent.toml").write_text("# user\n")
        _sh("true", project)
        assert (project / "agent.toml").read_text() == "# user\n"

    def test_ordinary_files_unaffected(self, project):
        assert _sh("echo ok > notes.txt", project).returncode == 0
        assert (project / "notes.txt").exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
class TestNestedGitIsReadOnly:
    """Only the root `.git` used to be bound: `sub/.git/config` was writable,
    and the user's `git status` in sub/ runs whatever core.fsmonitor says."""

    def test_nested_git_config_readonly(self, project):
        (project / "sub" / ".git").mkdir(parents=True)
        (project / "sub" / ".git" / "config").write_text("[core]\n")
        assert _sh("echo '[core] fsmonitor = evil' >> sub/.git/config", project).returncode != 0
        assert _sh("echo x > sub/.git/hooks-new", project).returncode != 0
        assert (project / "sub" / ".git" / "config").read_text() == "[core]\n"

    def test_gitfile_readonly(self, project):
        (project / "sub").mkdir()
        (project / "sub" / ".git").write_text("gitdir: ../.git/modules/sub\n")
        assert _sh("echo 'gitdir: /tmp/x' > sub/.git", project).returncode != 0

    def test_rest_of_subdir_writable(self, project):
        (project / "sub" / ".git").mkdir(parents=True)
        assert _sh("echo ok > sub/file.txt", project).returncode == 0


class TestFsGate:
    @pytest.mark.parametrize("rel", ["agent.yaml", "agent.yml", "sub/agent.yaml", ".git",
                                     ".gemini/settings.json"])
    def test_write_denied(self, project, rel):
        assert sec_fs._is_write_protected(project, project / rel)

    def test_safe_open_refuses_swapped_directory(self, project, tmp_path):
        """A directory checked by safe_resolve then swapped for a symlink must
        not redirect the host-side open."""
        outside = tmp_path / "outside"
        outside.mkdir()
        d = project / "d"
        d.mkdir()
        real = sec_fs.safe_resolve("d/f.txt")
        d.rmdir()
        os.symlink(outside, d)
        from agent.security import path_grants as pg
        grant = pg.grant_for(real)
        with pytest.raises(sec_fs.SymlinkDenied):
            sec_fs._open_beneath(grant.path, real, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        assert not (outside / "f.txt").exists()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
class TestBwrapArgv:
    def test_own_interpreter_under_root_is_readonly(self, project, monkeypatch):
        venv = project / ".venv"
        venv.mkdir()
        monkeypatch.setattr(sys, "prefix", str(venv))
        argv = sec_runner._bwrap_argv(["true"], cwd=project, network=False)
        i = argv.index(str(venv))
        assert argv[i - 1] == "--ro-bind"


class TestGitRepoGuard:
    def test_nested_git_dir_refused(self, project):
        from agent.tools.git import main as g
        (project / "sub" / ".git").mkdir(parents=True)
        assert g._untrusted_git_dir(project / "sub", project)

    def test_gitfile_into_tree_refused(self, project):
        from agent.tools.git import main as g
        (project / "sub").mkdir()
        (project / "sub" / ".git").write_text("gitdir: ../.agent/tmp/g\n")
        assert g._untrusted_git_dir(project / "sub", project)

    def test_submodule_gitdir_allowed(self, project):
        from agent.tools.git import main as g
        (project / ".git" / "modules" / "sub").mkdir(parents=True)
        (project / "sub").mkdir()
        (project / "sub" / ".git").write_text("gitdir: ../.git/modules/sub\n")
        assert not g._untrusted_git_dir(project / "sub", project)
