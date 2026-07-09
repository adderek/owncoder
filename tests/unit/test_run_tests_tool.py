"""Unit tests for agent/tools/run_tests/ — test discovery + summarized runs."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.config.models import TestSuiteConfig
from agent.tools.run_tests.main import (
    _build_argv,
    _detect_convention,
    _guess_parser,
    _parse_cargo,
    _parse_go,
    _parse_pytest,
    _select_suites,
    detect_framework,
    run_tests,
    setup,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    import agent.tools.run_tests.main as mod
    monkeypatch.setattr(mod, "_config", None)
    yield


def _make_config(root: Path, verify_command: str = "", suites: list | None = None):
    cfg = MagicMock()
    cfg.tools.working_dir = str(root)
    cfg.verify.command = verify_command
    cfg.tests.suites = suites or []
    return cfg


class TestDetect:
    def test_pytest_ini(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        assert detect_framework(str(tmp_path)) == "pytest"

    def test_pyproject_pytest_section(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
        assert detect_framework(str(tmp_path)) == "pytest"

    def test_cargo(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\n")
        assert detect_framework(str(tmp_path)) == "cargo"

    def test_go(self, tmp_path):
        (tmp_path / "go.mod").write_text("module x\n")
        assert detect_framework(str(tmp_path)) == "go"

    def test_npm_with_test_script(self, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}')
        assert detect_framework(str(tmp_path)) == "npm"

    def test_npm_without_test_script_ignored(self, tmp_path):
        (tmp_path / "package.json").write_text('{"scripts": {}}')
        assert detect_framework(str(tmp_path)) is None

    def test_bare_test_files_use_pytest(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_a.py").write_text("def test_x(): pass\n")
        assert detect_framework(str(tmp_path)) == "pytest"

    def test_nothing_detected(self, tmp_path):
        assert detect_framework(str(tmp_path)) is None

    def test_skips_venv_dirs(self, tmp_path):
        d = tmp_path / ".venv" / "lib"
        d.mkdir(parents=True)
        (d / "test_vendored.py").write_text("")
        assert detect_framework(str(tmp_path)) is None


class TestBuildArgv:
    def test_pytest_path_pattern(self, tmp_path):
        argv = _build_argv("pytest", str(tmp_path), "tests/unit/test_x.py")
        assert argv[-1] == "tests/unit/test_x.py"
        assert "-k" not in argv

    def test_pytest_keyword_pattern(self, tmp_path):
        argv = _build_argv("pytest", str(tmp_path), "notes and not slow")
        assert argv[-2:] == ["-k", "notes and not slow"]

    def test_pytest_uses_project_venv(self, tmp_path):
        vpy = tmp_path / ".venv" / "bin" / "python"
        vpy.parent.mkdir(parents=True)
        vpy.write_text("#!/bin/sh\n")
        vpy.chmod(0o755)
        argv = _build_argv("pytest", str(tmp_path), "")
        assert argv[0] == str(vpy)

    def test_go_run_filter(self, tmp_path):
        argv = _build_argv("go", str(tmp_path), "TestFoo")
        assert argv[-2:] == ["-run", "TestFoo"]

    def test_unknown_framework_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _build_argv("mocha", str(tmp_path), "")


class TestParsers:
    def test_pytest_counts(self):
        out = "....\n3 failed, 10 passed, 2 skipped in 1.23s\nFAILED tests/test_a.py::test_x\n"
        s = _parse_pytest(out)
        assert (s["passed"], s["failed"], s["skipped"]) == (10, 3, 2)
        assert s["failures"] == ["tests/test_a.py::test_x"]

    def test_pytest_all_pass(self):
        s = _parse_pytest("21 passed in 0.48s\n")
        assert s["passed"] == 21 and s["failed"] == 0

    def test_go(self):
        out = "--- PASS: TestA\n--- FAIL: TestB\nFAIL\n"
        s = _parse_go(out)
        assert s["passed"] == 1 and s["failures"] == ["TestB"]

    def test_cargo(self):
        out = "test result: FAILED. 7 passed; 1 failed; 0 ignored; ...\n---- tests::boom stdout ----\n"
        s = _parse_cargo(out)
        assert s["passed"] == 7 and s["failed"] == 1
        assert s["failures"] == ["tests::boom"]


class TestSelectSuites:
    def _suites(self):
        return [
            TestSuiteConfig(name="agent-unit", command="x"),
            TestSuiteConfig(name="root-unit", command="y"),
            TestSuiteConfig(name="e2e", command="z", default=False),
        ]

    def test_empty_selector_returns_defaults_only(self):
        sel = _select_suites(self._suites(), "")
        assert [s.name for s in sel] == ["agent-unit", "root-unit"]

    def test_exact_name_wins(self):
        sel = _select_suites(self._suites(), "e2e")
        assert [s.name for s in sel] == ["e2e"]

    def test_substring_match(self):
        sel = _select_suites(self._suites(), "unit")
        assert [s.name for s in sel] == ["agent-unit", "root-unit"]

    def test_no_match(self):
        assert _select_suites(self._suites(), "android") == []


class TestGuessParser:
    def test_explicit_wins(self):
        assert _guess_parser(TestSuiteConfig(command="pytest", parser="none")) == "none"

    def test_pytest_from_command(self):
        assert _guess_parser(TestSuiteConfig(command=".venv/bin/python -m pytest -q")) == "pytest"

    def test_go_from_command(self):
        assert _guess_parser(TestSuiteConfig(command="go test ./...")) == "go"

    def test_unknown_command(self):
        assert _guess_parser(TestSuiteConfig(command="./gradlew test")) == "none"


class TestDetectConvention:
    def test_makefile_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\techo b\ntest:\n\techo t\n")
        assert _detect_convention(str(tmp_path)) == ("make", "make test")

    def test_makefile_without_test_target(self, tmp_path):
        (tmp_path / "Makefile").write_text("build:\n\techo b\n")
        assert _detect_convention(str(tmp_path)) is None

    def test_test_script(self, tmp_path):
        d = tmp_path / "scripts"
        d.mkdir()
        sh = d / "test.sh"
        sh.write_text("#!/bin/sh\nexit 0\n")
        sh.chmod(0o755)
        assert _detect_convention(str(tmp_path)) == ("script", "./scripts/test.sh")


class TestDeclaredSuites:
    def test_suites_run_and_aggregate(self, tmp_path):
        (tmp_path / "sub").mkdir()
        suites = [
            TestSuiteConfig(name="ok", dir=".", command="echo '3 passed in 0.1s'"),
            TestSuiteConfig(name="bad", dir="sub", command="echo '1 failed, 2 passed in 0.1s'; exit 1"),
        ]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["framework"] == "declared-suites"
        assert r["ok"] is False
        assert r["passed"] == 5 and r["failed"] == 1
        assert [s["suite"] for s in r["suites"]] == ["ok", "bad"]
        assert r["suites"][1]["ok"] is False

    def test_named_suite_only(self, tmp_path):
        suites = [
            TestSuiteConfig(name="a", command="echo '1 passed in 0s'"),
            TestSuiteConfig(name="b", command="exit 1"),
        ]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(suite="a", path=str(tmp_path)))
        assert r["ok"] is True and len(r["suites"]) == 1

    def test_unknown_suite_lists_available(self, tmp_path):
        suites = [TestSuiteConfig(name="a", command="true")]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(suite="nope", path=str(tmp_path)))
        assert "error" in r and r["available_suites"] == ["a"]

    def test_non_default_suite_skipped_unnamed(self, tmp_path):
        suites = [
            TestSuiteConfig(name="fast", command="echo '1 passed in 0s'"),
            TestSuiteConfig(name="slow", command="exit 1", default=False),
        ]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["ok"] is True and len(r["suites"]) == 1

    def test_suite_env_passed(self, tmp_path):
        suites = [TestSuiteConfig(name="env", command='test "$MARKER" = hello',
                                  env={"MARKER": "hello"})]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["ok"] is True

    def test_missing_suite_dir_reported(self, tmp_path):
        suites = [TestSuiteConfig(name="ghost", dir="nope", command="true")]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["ok"] is False
        assert "does not exist" in r["suites"][0]["error"]

    def test_pattern_falls_through_when_no_suite_matches(self, tmp_path):
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        (tmp_path / "test_s.py").write_text("def test_alpha():\n    assert True\n")
        suites = [TestSuiteConfig(name="other", command="exit 1")]
        setup(_make_config(tmp_path, suites=suites))
        r = asyncio.run(run_tests(pattern="alpha", path=str(tmp_path)))
        assert r["framework"] == "pytest" and r["ok"] is True


class TestConventionTier:
    def test_make_test_used_when_no_suites(self, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\t@echo '2 passed in 0.1s'\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["framework"] == "convention:make"
        assert r["ok"] is True and r["passed"] == 2

    def test_convention_skipped_for_pattern_runs(self, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\t@exit 1\n")
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        (tmp_path / "test_s.py").write_text("def test_alpha():\n    assert True\n")
        setup(_make_config(tmp_path))
        r = asyncio.run(run_tests(pattern="alpha", path=str(tmp_path)))
        assert r["framework"] == "pytest" and r["ok"] is True


class TestRunTestsTool:
    def _pytest_project(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "pytest.ini").write_text("[pytest]\n")
        (tmp_path / "test_sample.py").write_text(body)
        return tmp_path

    def test_end_to_end_pass(self, tmp_path):
        root = self._pytest_project(tmp_path, "def test_ok():\n    assert True\n")
        setup(_make_config(root))
        r = asyncio.run(run_tests(path=str(root)))
        assert r["framework"] == "pytest"
        assert r["ok"] is True
        assert r["passed"] == 1 and r["failed"] == 0

    def test_end_to_end_failure_reported(self, tmp_path):
        root = self._pytest_project(
            tmp_path, "def test_ok():\n    assert True\n\ndef test_bad():\n    assert False\n")
        setup(_make_config(root))
        r = asyncio.run(run_tests(path=str(root)))
        assert r["ok"] is False
        assert r["failed"] == 1 and r["passed"] == 1
        assert any("test_bad" in f for f in r["failures"])

    def test_pattern_subset(self, tmp_path):
        root = self._pytest_project(
            tmp_path, "def test_alpha():\n    assert True\n\ndef test_beta():\n    assert False\n")
        setup(_make_config(root))
        r = asyncio.run(run_tests(pattern="alpha", path=str(root)))
        assert r["ok"] is True and r["passed"] == 1

    def test_no_framework_error(self, tmp_path):
        setup(_make_config(tmp_path))
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert "error" in r

    def test_verify_command_wins_for_full_runs(self, tmp_path):
        cfg = _make_config(tmp_path, verify_command="echo '2 passed in 0.1s'")
        setup(cfg)
        r = asyncio.run(run_tests(path=str(tmp_path)))
        assert r["framework"] == "verify.command"
        assert r["ok"] is True

    def test_verify_command_skipped_when_pattern(self, tmp_path):
        root = self._pytest_project(tmp_path, "def test_ok():\n    assert True\n")
        cfg = _make_config(root, verify_command="echo should-not-run")
        setup(cfg)
        r = asyncio.run(run_tests(pattern="ok", path=str(root)))
        assert r["framework"] == "pytest"

    def test_not_a_directory(self, tmp_path):
        setup(_make_config(tmp_path))
        r = asyncio.run(run_tests(path=str(tmp_path / "nope")))
        assert "error" in r
