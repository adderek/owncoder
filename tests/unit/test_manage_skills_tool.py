"""Unit tests for tools/manage_skills."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.tools.manage_skills.main import search_skills, load_skill, setup


@pytest.fixture()
def skill_env(tmp_path, monkeypatch):
    project_skills = tmp_path / ".agent" / "skills"
    project_skills.mkdir(parents=True)
    bundled_skills = tmp_path / "bundled"
    bundled_skills.mkdir()

    cfg = MagicMock()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"

    setup(cfg)

    import agent.tools.manage_skills.main as m
    from agent.skills import SkillLoader
    loader = SkillLoader(cfg)
    monkeypatch.setattr(loader, "_bundled_dir", bundled_skills)
    m._skill_loader = loader

    return project_skills, bundled_skills


class TestSearchSkills:
    def test_empty_returns_all(self, skill_env):
        _, bundled = skill_env
        (bundled / "python.md").write_text("# Python guidelines\nbody")
        result = search_skills()
        assert result["count"] >= 1
        names = [s["name"] for s in result["skills"]]
        assert "python" in names

    def test_keyword_filter(self, skill_env):
        _, bundled = skill_env
        (bundled / "python.md").write_text("# Python guidelines\nbody")
        (bundled / "testing.md").write_text("# Testing patterns\nbody")
        result = search_skills("python")
        assert result["count"] == 1
        assert result["skills"][0]["name"] == "python"

    def test_no_matches(self, skill_env):
        result = search_skills("zzznomatch")
        assert result["count"] == 0

    def test_project_skill_visible(self, skill_env):
        project, _ = skill_env
        (project / "custom.md").write_text("# Custom skill\nbody")
        result = search_skills("custom")
        assert result["count"] == 1


class TestLoadSkill:
    def test_load_existing(self, skill_env):
        _, bundled = skill_env
        (bundled / "python.md").write_text("# Python\nuse type hints always")
        result = load_skill("python")
        assert "use type hints always" in result["content"]
        assert "error" not in result

    def test_load_missing(self, skill_env):
        result = load_skill("nonexistent")
        assert "error" in result
        assert "available" in result

    def test_project_overrides_bundled(self, skill_env):
        project, bundled = skill_env
        (bundled / "python.md").write_text("# Python\nbundled body")
        (project / "python.md").write_text("# Python\nproject body")
        result = load_skill("python")
        assert "project body" in result["content"]
        assert "bundled body" not in result["content"]

    def test_header_stripped(self, skill_env):
        _, bundled = skill_env
        (bundled / "foo.md").write_text("# Foo\nfoo content")
        result = load_skill("foo")
        assert not result["content"].startswith("# Skill: foo")
        assert "foo content" in result["content"]
