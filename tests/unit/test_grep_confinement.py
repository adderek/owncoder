"""Confinement tests for grep_code — paths must stay within working_dir."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent.tools.search import grep as grep_mod
from agent.config.models import Config, ToolsConfig


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("SECRET_IN_PROJECT = 1\n")
    return tmp_path


@pytest.fixture
def grep_config(project_dir):
    cfg = Config(tools=ToolsConfig(working_dir=str(project_dir)))
    grep_mod.setup(cfg)
    return cfg


class TestGrepConfinement:
    def test_absolute_escape_blocked(self, grep_config, project_dir):
        result = grep_mod.grep_code(pattern="root", path="/etc")
        assert "error" in result
        assert "escapes" in result["error"].lower() or "project root" in result["error"].lower()

    def test_dotdot_escape_blocked(self, grep_config, project_dir):
        result = grep_mod.grep_code(pattern="x", path="../../outside")
        assert "error" in result

    def test_home_escape_blocked(self, grep_config, project_dir):
        result = grep_mod.grep_code(pattern="ssh", path="~/.ssh")
        assert "error" in result

    def test_in_root_search_works(self, grep_config, project_dir):
        result = grep_mod.grep_code(pattern="SECRET_IN_PROJECT")
        assert "error" not in result
        assert result["count"] >= 1
        assert any("main.py" in r["path"] for r in result["results"])

    def test_subdir_search_works(self, grep_config, project_dir):
        result = grep_mod.grep_code(pattern="SECRET_IN_PROJECT", path="src")
        assert "error" not in result
        assert result["count"] >= 1

    def test_absolute_within_root_allowed(self, grep_config, project_dir):
        result = grep_mod.grep_code(
            pattern="SECRET_IN_PROJECT",
            path=str(project_dir / "src"),
        )
        assert "error" not in result
        assert result["count"] >= 1


class TestGrepReadDeny:
    """grep must not surface secret files read_file refuses to open."""

    def test_env_file_not_surfaced(self, grep_config, project_dir):
        (project_dir / ".env").write_text("CUSTOM_HOST=secret.internal.example.com\n")
        result = grep_mod.grep_code(pattern="example.com", file_glob="*")
        assert "error" not in result
        assert all(not r["path"].endswith(".env") for r in result["results"]), result

    def test_pem_key_not_surfaced(self, grep_config, project_dir):
        (project_dir / "id.pem").write_text("-----BEGIN PRIVATE KEY-----\nMIIabc\n")
        result = grep_mod.grep_code(pattern="PRIVATE", file_glob="*")
        assert "error" not in result
        assert all(not r["path"].endswith(".pem") for r in result["results"]), result

    def test_normal_source_still_found(self, grep_config, project_dir):
        (project_dir / ".env").write_text("HOST=example.com\n")
        result = grep_mod.grep_code(pattern="SECRET_IN_PROJECT", file_glob="*")
        assert "error" not in result
        assert any("main.py" in r["path"] for r in result["results"])

    def test_is_read_protected_helper(self):
        deny = [".env", ".env.*", "*.pem", "**/.ssh/*"]
        assert grep_mod._is_read_protected(".env", deny)
        assert grep_mod._is_read_protected("sub/.env", deny)
        assert grep_mod._is_read_protected("id.pem", deny)
        assert not grep_mod._is_read_protected("src/main.py", deny)
