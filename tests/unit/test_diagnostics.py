"""Unit tests for core.diagnostics — per-edit, file-scoped checker feedback."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.config import Config
from agent.config.models import DiagnosticsCheckerConfig
import agent.core.diagnostics as diag


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    c.diagnostics.enabled = True
    c.diagnostics.timeout_s = 20.0
    # Declared checker beats auto-detection, so the test does not depend on
    # which linters happen to be installed on the machine running it.
    c.diagnostics.checkers = [DiagnosticsCheckerConfig(
        suffixes=[".py"],
        command=["python3", "-c",
                 "import sys,py_compile;py_compile.compile(sys.argv[1],doraise=True)",
                 "{file}"],
    )]
    return c


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


def _edit(path):
    return _Call("edit_file", json.dumps({"path": path}))


def _run(coro):
    return asyncio.run(coro)


class TestCheckPath:
    def test_clean_file_has_no_findings(self, cfg, tmp_path):
        (tmp_path / "ok.py").write_text("x = 1\n")
        assert _run(diag.check_path("ok.py", cfg)) == []

    def test_broken_file_reports_findings(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        findings = _run(diag.check_path("bad.py", cfg))
        assert findings, "syntax error should produce at least one finding"

    def test_disabled_returns_nothing(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        cfg.diagnostics.enabled = False
        assert _run(diag.check_path("bad.py", cfg)) == []

    def test_unmatched_suffix_is_skipped(self, cfg, tmp_path):
        (tmp_path / "notes.txt").write_text("def f(:\n")
        assert _run(diag.check_path("notes.txt", cfg)) == []

    def test_path_outside_working_dir_is_refused(self, cfg, tmp_path):
        outside = tmp_path.parent / "outside.py"
        outside.write_text("def f(:\n")
        assert _run(diag.check_path(str(outside), cfg)) == []

    def test_findings_are_capped(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        cfg.diagnostics.max_findings = 1
        assert len(_run(diag.check_path("bad.py", cfg))) <= 1

    def test_timeout_yields_no_findings(self, cfg, tmp_path):
        (tmp_path / "slow.py").write_text("x = 1\n")
        cfg.diagnostics.timeout_s = 0.05
        cfg.diagnostics.checkers = [DiagnosticsCheckerConfig(
            suffixes=[".py"], command=["python3", "-c", "import time;time.sleep(5)"],
        )]
        assert _run(diag.check_path("slow.py", cfg)) == []

    def test_missing_binary_yields_no_findings(self, cfg, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        cfg.diagnostics.checkers = [DiagnosticsCheckerConfig(
            suffixes=[".py"], command=["definitely-not-a-real-binary-xyz", "{file}"],
        )]
        assert _run(diag.check_path("a.py", cfg)) == []


class TestAnnotate:
    def test_broken_edit_gets_diagnostics(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        results = [json.dumps({"status": "ok"})]
        out = _run(diag.annotate([_edit("bad.py")], results, cfg))
        assert "_diagnostics" in json.loads(out[0])

    def test_clean_edit_is_untouched(self, cfg, tmp_path):
        (tmp_path / "ok.py").write_text("x = 1\n")
        results = [json.dumps({"status": "ok"})]
        assert _run(diag.annotate([_edit("ok.py")], results, cfg)) == results

    def test_failed_edit_is_not_checked(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        results = [json.dumps({"error": "anchor_not_found"})]
        assert _run(diag.annotate([_edit("bad.py")], results, cfg)) == results

    def test_non_json_result_passes_through(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        results = ["plain text result"]
        assert _run(diag.annotate([_edit("bad.py")], results, cfg)) == results

    def test_disabled_is_byte_identical(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        cfg.diagnostics.enabled = False
        results = [json.dumps({"status": "ok"})]
        assert _run(diag.annotate([_edit("bad.py")], results, cfg)) == results

    def test_same_path_twice_shares_findings(self, cfg, tmp_path):
        (tmp_path / "bad.py").write_text("def f(:\n")
        calls = [_edit("bad.py"), _edit("bad.py")]
        results = [json.dumps({"status": "ok"}), json.dumps({"status": "ok"})]
        out = _run(diag.annotate(calls, results, cfg))
        assert all("_diagnostics" in json.loads(r) for r in out)


class TestAutoCheckers:
    def test_auto_mode_picks_installed_only(self, cfg):
        cfg.diagnostics.checkers = []
        for suffixes, argv in diag._resolve_argv(cfg):
            assert suffixes and argv

    def test_declared_checkers_win(self, cfg):
        pairs = diag._resolve_argv(cfg)
        assert pairs == [((".py",), cfg.diagnostics.checkers[0].command)]
