"""Unit tests for agent/tools/build_project/ — build + parsed compiler errors."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.tools.build_project.main import (
    build_project,
    detect_build_system,
    extract_errors,
    setup,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    import agent.tools.build_project.main as mod
    monkeypatch.setattr(mod, "_config", None)
    yield


def _make_config(root: Path):
    cfg = MagicMock()
    cfg.tools.working_dir = str(root)
    return cfg


class TestDetect:
    def test_makefile(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:\n\ttrue\n")
        assert detect_build_system(str(tmp_path))[0] == "make"

    def test_meson_without_builddir(self, tmp_path):
        (tmp_path / "meson.build").write_text("project('x', 'c')\n")
        label, argv = detect_build_system(str(tmp_path))
        assert label == "meson" and argv[:2] == ["meson", "setup"]

    def test_meson_with_builddir(self, tmp_path):
        (tmp_path / "meson.build").write_text("project('x', 'c')\n")
        (tmp_path / "build").mkdir()
        assert detect_build_system(str(tmp_path))[1] == ["ninja", "-C", "build"]

    def test_gradlew_preferred(self, tmp_path):
        (tmp_path / "gradlew").write_text("#!/bin/sh\n")
        (tmp_path / "build.gradle").write_text("")
        assert detect_build_system(str(tmp_path))[1][0] == "./gradlew"

    def test_nothing(self, tmp_path):
        assert detect_build_system(str(tmp_path)) is None


class TestExtractErrors:
    def test_gcc_style(self):
        out = "src/main.c:42:7: error: 'x' undeclared\nsrc/main.c:50:1: warning: unused\n"
        errs = extract_errors(out)
        assert errs[0] == {"file": "src/main.c", "line": 42, "kind": "error",
                           "message": "'x' undeclared"}
        assert errs[1]["kind"] == "warning"

    def test_errors_sorted_before_warnings(self):
        out = "a.c:1:1: warning: w\nb.c:2:2: error: e\n"
        errs = extract_errors(out)
        assert [e["kind"] for e in errs] == ["error", "warning"]

    def test_javac(self):
        out = "src/App.java:12: error: cannot find symbol\n"
        assert extract_errors(out)[0]["file"] == "src/App.java"

    def test_kotlin_daemon(self):
        out = "e: /app/src/Main.kt:8:5 unresolved reference: foo\n"
        e = extract_errors(out)[0]
        assert e["kind"] == "error" and e["line"] == 8

    def test_go(self):
        out = "pkg/x.go:3:10: undefined: Foo\n"
        assert extract_errors(out)[0]["file"] == "pkg/x.go"

    def test_dedup(self):
        line = "a.c:1:1: error: boom\n"
        assert len(extract_errors(line * 3)) == 1

    def test_no_matches(self):
        assert extract_errors("BUILD SUCCESSFUL in 2s\n") == []


class TestBuildProjectTool:
    def test_no_build_system(self, tmp_path):
        setup(_make_config(tmp_path))
        r = asyncio.run(build_project(path=str(tmp_path)))
        assert "error" in r

    def test_make_success(self, tmp_path):
        (tmp_path / "Makefile").write_text("all:\n\t@echo built\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(build_project(path=str(tmp_path)))
        assert r["ok"] is True and r["system"] == "make"
        assert r["error_count"] == 0

    def test_make_failure_extracts_errors(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "all:\n\t@echo \"src/x.c:7:1: error: bad thing\" >&2; exit 1\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(build_project(path=str(tmp_path)))
        assert r["ok"] is False
        assert r["errors"][0]["file"] == "src/x.c" and r["errors"][0]["line"] == 7

    def test_make_target(self, tmp_path):
        (tmp_path / "Makefile").write_text(
            "all:\n\t@exit 1\ncustom:\n\t@echo custom-ran\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(build_project(target="custom", path=str(tmp_path)))
        assert r["ok"] is True and "custom" in r["command"]

    def test_unsupported_target_combo(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(build_project(target="thing", path=str(tmp_path)))
        assert "error" in r
