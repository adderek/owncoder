"""File tools see /tmp as the scratch (.agent/tmp), like the bwrap shell does.

Nothing the model writes may land in the host /tmp: it is shared, and a place
to leave setuid binaries or symlinks planted for other users.
"""
from __future__ import annotations

import os

import pytest

from agent.config.models import Config, ToolsConfig
from agent.security import fs as sec_fs
from agent.security import policy as sec_policy
from agent.tools import files
from agent.tools.files.paths import PathAccessDenied


@pytest.fixture
def env(tmp_path, monkeypatch):
    # Real projects don't live under /tmp; pytest's do. Stand in a fake host
    # tmp next to the project so the mapping is active.
    host_tmp = tmp_path / "hosttmp"
    host_tmp.mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(sec_policy, "HOST_TMP", host_tmp)
    monkeypatch.setattr(sec_fs, "_root_dev", None)
    monkeypatch.setattr(sec_fs, "_root_ino", None)
    cfg = Config(tools=ToolsConfig(working_dir=str(proj), agent_dir=str(proj / ".agent")))
    cfg.security.require_sandbox = False
    files.setup(cfg)
    yield proj, host_tmp, cfg
    sec_policy._policy = None
    sec_fs._root_dev = None
    sec_fs._root_ino = None


def test_write_to_tmp_lands_in_scratch(env):
    proj, host_tmp, _ = env
    out = files.write_file(path=str(host_tmp / "score_test.py"), content="print(1)\n")
    assert "error" not in out
    scratch = sec_policy.get().scratch_dir()
    assert (scratch / "score_test.py").read_text() == "print(1)\n"
    assert not (host_tmp / "score_test.py").exists()


def test_read_back_via_tmp_path(env):
    _, host_tmp, _ = env
    files.write_file(path=str(host_tmp / "a" / "b.txt"), content="hello\n")
    out = files.read_file(path=str(host_tmp / "a" / "b.txt"))
    assert "hello" in out["content"]


def test_dotdot_out_of_tmp_not_mapped(env):
    _, host_tmp, _ = env
    with pytest.raises(PathAccessDenied):
        files._resolve(str(host_tmp / ".." / "elsewhere.py"))


def test_symlink_in_scratch_cannot_escape(env):
    proj, host_tmp, _ = env
    outside = proj.parent / "outside.txt"
    outside.write_text("secret\n")
    scratch = sec_policy.get().ensure_scratch()
    os.symlink(outside, scratch / "link.txt")
    with pytest.raises(ValueError):
        files._resolve(str(host_tmp / "link.txt"))


def test_disabled_by_scratch_bind_tmp(env):
    _, host_tmp, cfg = env
    cfg.security.scratch_bind_tmp = False
    with pytest.raises(PathAccessDenied):
        files._resolve(str(host_tmp / "x.py"))


def test_project_under_tmp_not_mapped(env, monkeypatch):
    proj, _, _ = env
    monkeypatch.setattr(sec_policy, "HOST_TMP", proj.parent)
    assert sec_policy.get().map_tmp(proj.parent / "x.py") == proj.parent / "x.py"
