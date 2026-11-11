"""TODO-4 item 2a: host git must not run config the sandbox could plant.

PoC: inside the sandbox `mv mod mod_old && mkdir mod && git init mod`, then
`core.fsmonitor = <cmd>` in mod/.git/config. A plain host `git status` at the
superproject descends into the submodule and runs <cmd>.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from agent.security import host_git

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def planted(tmp_path, monkeypatch):
    for k, v in {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
                 "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    sub, sup = tmp_path / "sub", tmp_path / "super"
    sub.mkdir()
    _git("init", "-q", cwd=sub)
    (sub / "a").write_text("a")
    _git("add", "a", cwd=sub)
    _git("commit", "-qm", "a", cwd=sub)
    sup.mkdir()
    _git("init", "-q", cwd=sup)
    _git("-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "mod", cwd=sup)
    _git("commit", "-qm", "s", cwd=sup)
    # what the sandbox could do
    os.rename(sup / "mod", sup / "mod_old")
    (sup / "mod").mkdir()
    mod = sup / "mod"
    _git("init", "-q", cwd=mod)
    (mod / "b").write_text("b")
    _git("add", "b", cwd=mod)
    _git("commit", "-qm", "b", cwd=mod)
    marker = tmp_path / "PWNED"
    _git("config", "core.fsmonitor", f"touch {marker}; false", cwd=mod)
    _git("config", "log.showSignature", "true", cwd=mod)
    _git("config", "gpg.program", f"touch {marker}", cwd=mod)
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text(f"#!/bin/sh\ntouch {marker}\n")
    (hooks / "pre-commit").chmod(0o755)
    _git("config", "core.hooksPath", str(hooks), cwd=mod)
    return sup, marker


def test_plain_git_is_exploitable(planted):
    """Guards the PoC itself: if this stops firing, the test below proves nothing."""
    sup, marker = planted
    subprocess.run(["git", "status", "--short"], cwd=sup, capture_output=True)
    assert marker.exists()


@pytest.mark.parametrize("args", [
    ("status", "--short"), ("diff",), ("diff", "--cached"), ("diff", "--name-only", "HEAD"),
    ("-C", "mod", "log", "-1"), ("-C", "mod", "status"), ("-C", "mod", "diff"),
])
def test_hardened_git_runs_nothing(planted, args):
    sup, marker = planted
    r = host_git.run(args, cwd=sup)
    assert r.returncode == 0, r.stderr
    assert not marker.exists()


def test_status_still_shows_the_gitlink_change(planted):
    sup, _ = planted
    assert " M mod" in host_git.run(("status", "--short"), cwd=sup).stdout


def test_hooks_disabled_unless_asked(planted):
    sup, marker = planted
    mod = sup / "mod"
    (mod / "c").write_text("c")
    host_git.run(("add", "c"), cwd=mod)    # plain `git add` would fire fsmonitor
    r = host_git.run(("commit", "-qm", "c"), cwd=mod)
    assert r.returncode == 0, r.stderr
    assert not marker.exists()
    (mod / "d").write_text("d")
    host_git.run(("add", "d"), cwd=mod)
    host_git.run(("commit", "-qm", "d"), cwd=mod, hooks=True)
    assert marker.exists()          # hooks=True: the user's own commit keeps hooks
