"""Offline tests for evals/mine.py — recorded failures → ranked modes → scaffold.

No LLM, no agent binary: the input is a synthetic .agent/failures/ journal in a
tmp dir, exactly the shape failure_report.py writes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MINE_PY = REPO_ROOT / "evals" / "mine.py"
RUN_PY = REPO_ROOT / "evals" / "run.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mine = _load(MINE_PY, "evals_mine_under_test")
run = _load(RUN_PY, "evals_run_under_test_for_mine")


def _journal(project: Path, records: list[dict]) -> Path:
    directory = project / ".agent" / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "index.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return directory


@pytest.fixture()
def project(tmp_path):
    return tmp_path / "proj"


class TestNormalize:
    def test_paths_and_numbers_collapse(self):
        a = mine.normalize("anchor not found in /home/x/src/a.py line 42")
        b = mine.normalize("anchor not found in /srv/other/b.py line 7")
        assert a == b

    def test_different_errors_stay_different(self):
        assert mine.normalize("unknown tool: frobnicate") != mine.normalize("bad JSON arguments")

    def test_empty_input_is_safe(self):
        assert mine.normalize("") == ""


class TestClustering:
    def test_identical_shapes_group(self, project):
        _journal(project, [
            {"kind": "tool_exception", "tool": "edit_file", "reason": "anchor",
             "error": "anchor not found in /a/b.py line 3", "session_id": "s1", "ts": "2026-01-01"},
            {"kind": "tool_exception", "tool": "edit_file", "reason": "anchor",
             "error": "anchor not found in /c/d.py line 99", "session_id": "s2", "ts": "2026-01-02"},
        ])
        modes = mine.cluster(mine.load_records(project / ".agent" / "failures"))
        assert len(modes) == 1
        assert modes[0].count == 2
        assert len(modes[0].sessions) == 2

    def test_ranked_by_frequency(self, project):
        records = [{"kind": "invalid_tool_call", "tool": "rare", "reason": "x",
                    "error": "one off", "session_id": "s1", "ts": "2026-01-01"}]
        records += [{"kind": "invalid_tool_call", "tool": "common", "reason": "y",
                     "error": "same shape", "session_id": f"s{i}", "ts": "2026-01-02"}
                    for i in range(5)]
        _journal(project, records)
        modes = mine.cluster(mine.load_records(project / ".agent" / "failures"))
        assert modes[0].tool == "common"

    def test_first_and_last_seen_span_the_records(self, project):
        _journal(project, [
            {"kind": "k", "tool": "t", "error": "e", "ts": "2026-03-05"},
            {"kind": "k", "tool": "t", "error": "e", "ts": "2026-01-09"},
        ])
        mode = mine.cluster(mine.load_records(project / ".agent" / "failures"))[0]
        assert mode.first_seen == "2026-01-09"
        assert mode.last_seen == "2026-03-05"

    def test_detail_file_enriches_the_record(self, project):
        directory = _journal(project, [
            {"kind": "invalid_tool_call", "tool": "edit_file", "error": "boom",
             "ts": "2026-01-01", "file": "detail.json"},
        ])
        (directory / "detail.json").write_text(
            json.dumps({"arguments": {"path": "src/a.py", "anchor": "def foo"}}))
        mode = mine.cluster(mine.load_records(directory))[0]
        assert mode.samples[0]["arguments"]["anchor"] == "def foo"

    def test_malformed_lines_are_skipped(self, project):
        directory = project / ".agent" / "failures"
        directory.mkdir(parents=True)
        (directory / "index.jsonl").write_text(
            '{"kind": "k", "tool": "t", "error": "e"}\nnot json at all\n\n')
        assert len(mine.load_records(directory)) == 1

    def test_missing_journal_is_empty_not_an_error(self, tmp_path):
        assert mine.load_records(tmp_path / "nope") == []

    def test_multiple_projects_are_merged(self, tmp_path):
        for name in ("a", "b"):
            _journal(tmp_path / name, [{"kind": "k", "tool": "t", "error": "same"}])
        dirs = mine.failure_dirs([tmp_path / "a", tmp_path / "b"])
        records = [r for d in dirs for r in mine.load_records(d)]
        assert mine.cluster(records)[0].count == 2


class TestScaffold:
    def _mode(self, project):
        _journal(project, [
            {"kind": "invalid_tool_call", "tool": "edit_file", "reason": "anchor_not_found",
             "error": "anchor not found", "session_id": "s1", "ts": "2026-01-01"},
        ])
        return mine.cluster(mine.load_records(project / ".agent" / "failures"))[0]

    def test_writes_task_and_evidence(self, project, tmp_path):
        tasks, fixtures = tmp_path / "tasks", tmp_path / "fixtures"
        tasks.mkdir()
        task_path, fixture_dir = mine.scaffold(self._mode(project), tasks, fixtures)
        assert task_path.is_file()
        evidence = json.loads((fixture_dir / "FAILURE.json").read_text())
        assert evidence["tool"] == "edit_file"
        assert evidence["occurrences"] == 1

    def test_scaffold_is_a_draft(self, project, tmp_path):
        tasks, fixtures = tmp_path / "tasks", tmp_path / "fixtures"
        tasks.mkdir()
        task_path, _ = mine.scaffold(self._mode(project), tasks, fixtures)
        assert "draft: true" in task_path.read_text()

    def test_repeated_scaffolds_do_not_collide(self, project, tmp_path):
        tasks, fixtures = tmp_path / "tasks", tmp_path / "fixtures"
        tasks.mkdir()
        mode = self._mode(project)
        first, _ = mine.scaffold(mode, tasks, fixtures)
        second, _ = mine.scaffold(mode, tasks, fixtures)
        assert first != second

    def test_scaffolded_task_parses_and_is_skipped_by_the_runner(self, project, tmp_path):
        tasks, fixtures = tmp_path / "tasks", tmp_path / "fixtures"
        tasks.mkdir()
        mine.scaffold(self._mode(project), tasks, fixtures)
        loaded, errors = run.load_tasks(tasks)
        assert loaded == [], "a draft must not enter the graded suite"
        assert errors == [], "…and must not be reported as malformed either"

    def test_drafts_can_be_opted_into(self, project, tmp_path):
        tasks, fixtures = tmp_path / "tasks", tmp_path / "fixtures"
        tasks.mkdir()
        mine.scaffold(self._mode(project), tasks, fixtures)
        loaded, _ = run.load_tasks(tasks, include_drafts=True)
        assert len(loaded) == 1


class TestCli:
    def test_report_runs_on_empty_project(self, tmp_path, capsys):
        assert mine.main(["--project", str(tmp_path)]) == 0
        assert "Nothing to mine" in capsys.readouterr().out

    def test_json_output(self, project, tmp_path):
        _journal(project, [{"kind": "k", "tool": "t", "error": "e", "session_id": "s"}])
        out = tmp_path / "modes.json"
        mine.main(["--project", str(project), "--json", str(out)])
        assert json.loads(out.read_text())[0]["tool"] == "t"

    def test_min_count_filters(self, project, capsys):
        _journal(project, [{"kind": "k", "tool": "t", "error": "e"}])
        mine.main(["--project", str(project), "--min-count", "5"])
        assert "Nothing to mine" in capsys.readouterr().out

    def test_scaffold_index_out_of_range_is_an_error(self, project, capsys):
        _journal(project, [{"kind": "k", "tool": "t", "error": "e"}])
        assert mine.main(["--project", str(project), "--scaffold", "9"]) == 1


class TestFailUnderFloor:
    """--baseline only catches *movement*. A suite that was already failing half
    its tasks compares clean against a matching baseline and reports success;
    an absolute floor is what catches that."""

    def _exit(self, monkeypatch, tmp_path, statuses, extra_args):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        results = [run.TaskResult(id=f"t{i}", status=s) for i, s in enumerate(statuses)]
        pending = list(results)
        monkeypatch.setattr(run, "load_tasks",
                            lambda *a, **k: ([{"id": r.id, "prompt": "p", "fixture": "f"}
                                              for r in results], []))
        monkeypatch.setattr(run, "run_task", lambda *a, **k: pending.pop(0))
        return run.main(["--tasks-dir", str(tasks_dir), "--agent-cmd", "true", *extra_args])

    def _baseline(self, tmp_path, statuses) -> Path:
        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"results": [
            {"id": f"t{i}", "status": s, "checks": [], "judge": None}
            for i, s in enumerate(statuses)
        ]}))
        return path

    def test_matching_baseline_alone_reports_success(self, monkeypatch, tmp_path):
        statuses = ["pass", "fail", "fail", "fail"]
        baseline = self._baseline(tmp_path, statuses)
        assert self._exit(monkeypatch, tmp_path, statuses,
                          ["--baseline", str(baseline)]) == 0

    def test_floor_catches_what_the_baseline_misses(self, monkeypatch, tmp_path):
        statuses = ["pass", "fail", "fail", "fail"]
        baseline = self._baseline(tmp_path, statuses)
        assert self._exit(monkeypatch, tmp_path, statuses,
                          ["--baseline", str(baseline), "--fail-under", "0.75"]) == 1

    def test_rate_at_or_above_floor_passes(self, monkeypatch, tmp_path):
        statuses = ["pass", "pass", "pass", "fail"]
        baseline = self._baseline(tmp_path, statuses)
        assert self._exit(monkeypatch, tmp_path, statuses,
                          ["--baseline", str(baseline), "--fail-under", "0.75"]) == 0

    def test_floor_without_baseline_still_needs_every_task_to_pass(self, monkeypatch, tmp_path):
        assert self._exit(monkeypatch, tmp_path, ["pass", "pass", "pass", "fail"],
                          ["--fail-under", "0.5"]) == 1
