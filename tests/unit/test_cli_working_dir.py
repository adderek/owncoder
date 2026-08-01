"""s7: `--working-dir` picks the project root *and* the agent.toml that loads.

The flag is sugar over `config.tools.working_dir`, but it has to run before
the project root search — otherwise the config of the cwd's project would be
loaded while the tools worked in another directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.cli.main import _resolve_project, build_parser


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep the real ~/.config/agent out of it — those layers load *after* the
    project config and would mask what these tests assert."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


class _Args:
    def __init__(self, command="chat", config=None, working_dir=None):
        self.command = command
        self.config = config
        self.working_dir = working_dir


def _make_project(root: Path, marker: int | None = None) -> Path:
    """A project dir, optionally with an agent.toml carrying a marker value.

    `tools.shell_timeout` is just a field nothing else in this module writes,
    so it identifies which config layer won.
    """
    (root / ".agent").mkdir(parents=True, exist_ok=True)
    if marker:
        (root / "agent.toml").write_text(f"[tools]\nshell_timeout = {marker}\n")
    return root


def test_flag_is_in_the_parser():
    args = build_parser().parse_args(["--working-dir", "/tmp/x", "chat"])
    assert args.working_dir == "/tmp/x"


def test_flag_default_is_none():
    assert build_parser().parse_args(["chat"]).working_dir is None


def test_flag_sets_working_dir(tmp_path, monkeypatch):
    proj = _make_project(tmp_path / "proj")
    monkeypatch.chdir(tmp_path)
    root, config = _resolve_project(_Args(working_dir=str(proj)))
    assert root == proj
    assert config.tools.working_dir == str(proj)


def test_path_is_canonicalized(tmp_path, monkeypatch):
    proj = _make_project(tmp_path / "proj")
    link = tmp_path / "link"
    link.symlink_to(proj)
    monkeypatch.chdir(tmp_path)
    _, config = _resolve_project(_Args(working_dir=str(link)))
    assert config.tools.working_dir == os.path.realpath(str(proj))


def test_user_home_is_expanded(tmp_path, monkeypatch, _isolated_home):
    proj = _make_project(_isolated_home / "proj")
    monkeypatch.chdir(tmp_path)
    _, config = _resolve_project(_Args(working_dir="~/proj"))
    assert config.tools.working_dir == os.path.realpath(str(proj))


def test_flag_selects_that_projects_config(tmp_path, monkeypatch):
    here = _make_project(tmp_path / "here", marker=11)
    there = _make_project(tmp_path / "there", marker=22)
    monkeypatch.chdir(here)
    _, config = _resolve_project(_Args(working_dir=str(there)))
    assert config.tools.shell_timeout == 22


def test_without_flag_config_comes_from_cwd(tmp_path, monkeypatch):
    here = _make_project(tmp_path / "here", marker=11)
    _make_project(tmp_path / "there", marker=22)
    monkeypatch.chdir(here)
    root, config = _resolve_project(_Args())
    assert root == here
    assert config.tools.shell_timeout == 11
    assert config.tools.working_dir == str(here)


def test_explicit_config_flag_still_wins(tmp_path, monkeypatch):
    proj = _make_project(tmp_path / "proj", marker=33)
    other = tmp_path / "other.toml"
    other.write_text("[tools]\nshell_timeout = 55\n")
    monkeypatch.chdir(tmp_path)
    _, config = _resolve_project(_Args(config=str(other), working_dir=str(proj)))
    assert config.tools.shell_timeout == 55
    # ...but the working dir still comes from the flag.
    assert config.tools.working_dir == str(proj)


def test_parents_are_searched_from_the_flag_dir(tmp_path, monkeypatch):
    root = _make_project(tmp_path / "proj", marker=44)
    sub = root / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    found, config = _resolve_project(_Args(working_dir=str(sub)))
    assert found == root
    assert config.tools.shell_timeout == 44


def test_non_project_dir_exits(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _resolve_project(_Args(working_dir=str(plain)))


def test_missing_dir_exits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        _resolve_project(_Args(working_dir=str(tmp_path / "nope")))


def test_init_has_no_root_but_honours_the_flag(tmp_path, monkeypatch):
    target = tmp_path / "fresh"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    root, config = _resolve_project(_Args(command="init", working_dir=str(target)))
    assert root is None
    assert config.tools.working_dir == str(target)
