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


class TestGrepContextLines:
    def _project(self, tmp_path):
        (tmp_path / "mod.py").write_text(
            "\n".join(f"line{i}" for i in range(1, 6))
            + "\nNEEDLE = 1\n"
            + "\n".join(f"line{i}" for i in range(7, 12)) + "\n")
        return tmp_path

    def test_context_attached_and_marked(self, tmp_path, monkeypatch):
        import agent.tools.search.grep as g
        cfg = type("C", (), {"tools": type("T", (), {"working_dir": str(tmp_path)})()})()
        monkeypatch.setattr(g, "_config", cfg)
        self._project(tmp_path)
        r = g.grep_code("NEEDLE", context_lines=2)
        assert r["count"] == 1
        ctx = r["results"][0]["context"]
        assert "6> NEEDLE = 1" in ctx        # match line marked with '>'
        assert "4: line4" in ctx and "8: line8" in ctx
        assert "2: " not in ctx              # outside the window

    def test_no_context_by_default(self, tmp_path, monkeypatch):
        import agent.tools.search.grep as g
        cfg = type("C", (), {"tools": type("T", (), {"working_dir": str(tmp_path)})()})()
        monkeypatch.setattr(g, "_config", cfg)
        self._project(tmp_path)
        r = g.grep_code("NEEDLE")
        assert "context" not in r["results"][0]

    def test_context_lines_clamped(self, tmp_path, monkeypatch):
        import agent.tools.search.grep as g
        cfg = type("C", (), {"tools": type("T", (), {"working_dir": str(tmp_path)})()})()
        monkeypatch.setattr(g, "_config", cfg)
        self._project(tmp_path)
        r = g.grep_code("NEEDLE", context_lines=999)  # clamps to 10, must not error
        assert r["count"] == 1 and "context" in r["results"][0]


class TestAllTextFiles:
    """grep_code must cover non-source text files (.example, .template, no-ext)."""

    @pytest.fixture
    def rich_project(self, tmp_path):
        (tmp_path / "cfg.example").write_text("MAGIC_MARKER = 42\n")
        (tmp_path / "page.template").write_text("MAGIC_MARKER here too\n")
        (tmp_path / "Makefile").write_text("MAGIC_MARKER: all\n")
        (tmp_path / "img.png").write_bytes(b"\x89PNG\x00\x00MAGIC_MARKER\x00")
        (tmp_path / "code.py").write_text("MAGIC_MARKER = 'py'\n")
        cfg = Config(tools=ToolsConfig(working_dir=str(tmp_path)))
        grep_mod.setup(cfg)
        return tmp_path

    def test_nonstandard_extensions_found(self, rich_project):
        res = grep_mod.grep_code("MAGIC_MARKER")
        paths = {r["path"] for r in res["results"]}
        assert "cfg.example" in paths
        assert "page.template" in paths
        assert "Makefile" in paths
        assert "code.py" in paths

    def test_binary_not_matched(self, rich_project):
        res = grep_mod.grep_code("MAGIC_MARKER")
        paths = {r["path"] for r in res["results"]}
        assert "img.png" not in paths

    def test_file_glob_still_narrows(self, rich_project):
        res = grep_mod.grep_code("MAGIC_MARKER", file_glob="*.example")
        paths = {r["path"] for r in res["results"]}
        assert paths == {"cfg.example"}

    def test_source_field_names_tool(self, rich_project):
        res = grep_mod.grep_code("MAGIC_MARKER")
        assert res["source"] in ("ripgrep", "grep")

    def test_grep_fallback_covers_text_files(self, rich_project, monkeypatch):
        import shutil as _sh
        monkeypatch.setattr(grep_mod.shutil, "which", lambda name: None)
        res = grep_mod.grep_code("MAGIC_MARKER")
        assert res["source"] == "grep"
        paths = {r["path"] for r in res["results"]}
        assert "cfg.example" in paths
        assert "img.png" not in paths
