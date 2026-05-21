"""Unit tests for agent.skills.SkillLoader."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.skills import SkillLoader, _parse_skill_file


# ---------------------------------------------------------------------------
# _parse_skill_file
# ---------------------------------------------------------------------------

class TestParseSkillFile:
    def test_yaml_frontmatter(self, tmp_path):
        p = tmp_path / "s.md"
        p.write_text("---\ndescription: My skill\n---\nbody text\n")
        desc, body = _parse_skill_file(p)
        assert desc == "My skill"
        assert body == "body text"

    def test_hash_heading(self, tmp_path):
        p = tmp_path / "s.md"
        p.write_text("# My heading\nbody text\n")
        desc, body = _parse_skill_file(p)
        assert desc == "My heading"
        assert body == "body text"

    def test_no_description_falls_back_to_stem(self, tmp_path):
        p = tmp_path / "mystem.md"
        p.write_text("just body\n")
        desc, body = _parse_skill_file(p)
        assert desc == "mystem"
        assert body == "just body"


# ---------------------------------------------------------------------------
# SkillLoader
# ---------------------------------------------------------------------------

@pytest.fixture()
def skill_dirs(tmp_path):
    project_dir = tmp_path / ".agent" / "skills"
    bundled_dir = tmp_path / "bundled"
    project_dir.mkdir(parents=True)
    bundled_dir.mkdir()
    return project_dir, bundled_dir


@pytest.fixture()
def loader(tmp_path, skill_dirs, monkeypatch):
    project_dir, bundled_dir = skill_dirs
    cfg = MagicMock()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    ldr = SkillLoader(cfg)
    monkeypatch.setattr(ldr, "_bundled_dir", bundled_dir)
    return ldr, project_dir, bundled_dir


class TestSkillLoaderAvailable:
    def test_empty_dirs(self, loader):
        ldr, _, _ = loader
        assert ldr.available() == []

    def test_bundled_skills_listed(self, loader):
        ldr, _, bundled = loader
        (bundled / "python.md").write_text("# Python\nbody")
        names = [n for n, _ in ldr.available()]
        assert "python" in names

    def test_project_overrides_bundled(self, loader):
        ldr, project, bundled = loader
        (bundled / "python.md").write_text("# Python bundled\nbody")
        (project / "python.md").write_text("# Python project\nbody")
        skills = dict(ldr.available())
        assert skills["python"] == "Python project"

    def test_deduplication(self, loader):
        ldr, project, bundled = loader
        (bundled / "python.md").write_text("# Python\nbody")
        (project / "python.md").write_text("# Python local\nbody")
        assert sum(1 for n, _ in ldr.available() if n == "python") == 1


class TestSkillLoaderLoad:
    def test_load_existing(self, loader):
        ldr, _, bundled = loader
        (bundled / "python.md").write_text("# Python\nbody content")
        result = ldr.load(["python"])
        assert "body content" in result

    def test_load_missing_skill(self, loader):
        ldr, _, _ = loader
        result = ldr.load(["nonexistent"])
        assert "not found" in result

    def test_load_multiple(self, loader):
        ldr, _, bundled = loader
        (bundled / "a.md").write_text("# A\nalpha")
        (bundled / "b.md").write_text("# B\nbeta")
        result = ldr.load(["a", "b"])
        assert "alpha" in result
        assert "beta" in result

    def test_project_wins_over_bundled(self, loader):
        ldr, project, bundled = loader
        (bundled / "skill.md").write_text("# S\nbundled body")
        (project / "skill.md").write_text("# S\nproject body")
        result = ldr.load(["skill"])
        assert "project body" in result
        assert "bundled body" not in result


class TestSkillLoaderIndexSummary:
    def test_empty(self, loader):
        ldr, _, _ = loader
        assert ldr.index_summary() == ""

    def test_lists_skills(self, loader):
        ldr, _, bundled = loader
        (bundled / "python.md").write_text("# Python guidelines\nbody")
        summary = ldr.index_summary()
        assert "python" in summary
        assert "Python guidelines" in summary
