"""Offline tests for the eval harness runner (evals/run.py).

These tests never invoke a real LLM or the real `agent` binary: the agent
invocation is always replaced with a tiny fake script via --agent-cmd /
the equivalent programmatic call.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "evals" / "run.py"


def _load_run_module():
    spec = importlib.util.spec_from_file_location("evals_run_under_test", RUN_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses needs this to resolve __module__
    spec.loader.exec_module(module)
    return module


run = _load_run_module()


# --------------------------------------------------------------------------
# helpers to build a tiny synthetic task (NOT one of the seed tasks)
# --------------------------------------------------------------------------

def _write_task(tasks_dir: Path, task_dict: dict, filename: str = "greet-file.yaml") -> Path:
    tasks_dir.mkdir(parents=True, exist_ok=True)
    path = tasks_dir / filename
    if filename.endswith(".json") or run.yaml is None:
        path.write_text(json.dumps(task_dict))
    else:
        path.write_text(run.yaml.safe_dump(task_dict))
    return path


def _make_fixture(fixtures_dir: Path, fixture_id: str) -> Path:
    fixture_dir = fixtures_dir / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "seed.txt").write_text("seed marker\n")
    return fixture_dir


def _synthetic_task() -> dict:
    return {
        "id": "greet-file",
        "prompt": "Write the word hello into greet.txt in this directory.",
        "fixture": "greet-file",
        "timeout_s": 30,
        "checks": [
            {"type": "file_exists", "path": "greet.txt"},
            {"type": "file_contains", "path": "greet.txt", "text": "hello"},
            {"type": "file_contains", "path": "seed.txt", "text": "seed marker"},
        ],
    }


def _correct_fake_agent_script(tmp_path: Path) -> Path:
    """A fake agent that does the right thing for the synthetic task above."""
    script = tmp_path / "fake_agent_correct.py"
    script.write_text(textwrap.dedent("""\
        import pathlib
        pathlib.Path("greet.txt").write_text("hello there")
        """))
    return script


def _failing_fake_agent_script(tmp_path: Path) -> Path:
    """A fake agent that does nothing useful (leaves greet.txt missing)."""
    script = tmp_path / "fake_agent_noop.py"
    script.write_text("import sys\n")
    return script


# --------------------------------------------------------------------------
# load_tasks
# --------------------------------------------------------------------------

def test_load_tasks_reads_valid_task(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, _synthetic_task())

    tasks, errors = run.load_tasks(tasks_dir)

    assert errors == []
    assert len(tasks) == 1
    assert tasks[0]["id"] == "greet-file"
    assert tasks[0]["fixture"] == "greet-file"
    assert len(tasks[0]["checks"]) == 3


def test_load_tasks_reports_malformed_file_without_crashing(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True)
    # Missing required 'prompt' and 'fixture' fields.
    (tasks_dir / "broken.yaml").write_text("id: broken\n")
    # A perfectly valid task alongside it, to prove one bad file doesn't
    # take down the whole load.
    _write_task(tasks_dir, _synthetic_task())

    tasks, errors = run.load_tasks(tasks_dir)

    assert len(tasks) == 1
    assert tasks[0]["id"] == "greet-file"
    assert len(errors) == 1
    assert errors[0][0] == "broken"
    assert "missing" in errors[0][1].lower()


def test_load_tasks_missing_dir_returns_empty(tmp_path):
    tasks, errors = run.load_tasks(tmp_path / "does-not-exist")
    assert tasks == []
    assert errors == []


# --------------------------------------------------------------------------
# build_command
# --------------------------------------------------------------------------

def test_build_command_substitutes_placeholder():
    cmd = run.build_command("myagent run {prompt} --flag", "say hi")
    assert cmd == "myagent run 'say hi' --flag"


def test_build_command_appends_when_no_placeholder():
    cmd = run.build_command("myagent run", "say hi")
    assert cmd == "myagent run 'say hi'"


# --------------------------------------------------------------------------
# run_task: end-to-end against a fake agent (no real LLM / agent binary)
# --------------------------------------------------------------------------

def test_run_task_passes_with_correct_fake_agent(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    _make_fixture(fixtures_dir, "greet-file")
    script = _correct_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"

    result = run.run_task(_synthetic_task(), agent_cmd, keep=False, fixtures_dir=fixtures_dir)

    assert result.status == "pass"
    assert result.exit_code == 0
    assert not result.timed_out
    assert len(result.checks) == 3
    assert all(c["passed"] for c in result.checks)
    assert result.workspace is None  # not kept


def test_run_task_fails_with_noop_fake_agent(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    _make_fixture(fixtures_dir, "greet-file")
    script = _failing_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"

    result = run.run_task(_synthetic_task(), agent_cmd, keep=False, fixtures_dir=fixtures_dir)

    assert result.status == "fail"
    assert result.exit_code == 0  # the fake agent itself ran fine
    failed_types = {c["type"] for c in result.checks if not c["passed"]}
    assert "file_exists" in failed_types
    assert "file_contains" in failed_types


def test_run_task_reports_error_for_missing_fixture(tmp_path):
    fixtures_dir = tmp_path / "fixtures"  # deliberately never created
    agent_cmd = f"{sys.executable} -c pass {{prompt}}"

    result = run.run_task(_synthetic_task(), agent_cmd, keep=False, fixtures_dir=fixtures_dir)

    assert result.status == "error"
    assert "fixture not found" in result.message


def test_run_task_keep_preserves_workspace(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    _make_fixture(fixtures_dir, "greet-file")
    script = _correct_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"

    result = run.run_task(_synthetic_task(), agent_cmd, keep=True, fixtures_dir=fixtures_dir)

    assert result.status == "pass"
    assert result.workspace is not None
    workspace_path = Path(result.workspace)
    assert workspace_path.is_dir()
    assert (workspace_path / "greet.txt").read_text() == "hello there"
    assert (workspace_path / "seed.txt").exists()  # fixture was actually copied


def test_run_task_agent_timeout_fails_all_checks(tmp_path):
    fixtures_dir = tmp_path / "fixtures"
    _make_fixture(fixtures_dir, "greet-file")
    slow_script = tmp_path / "slow_agent.py"
    slow_script.write_text("import time\ntime.sleep(5)\n")
    agent_cmd = f"{sys.executable} {slow_script} {{prompt}}"

    task = dict(_synthetic_task())
    task["timeout_s"] = 1
    result = run.run_task(task, agent_cmd, keep=False, fixtures_dir=fixtures_dir)

    assert result.status == "fail"
    assert result.timed_out is True
    assert all(not c["passed"] for c in result.checks)


# --------------------------------------------------------------------------
# run_check unit coverage for each check type
# --------------------------------------------------------------------------

def test_run_check_command_pass_and_fail(tmp_path):
    ok = run.run_check({"type": "command", "run": "true"}, tmp_path)
    bad = run.run_check({"type": "command", "run": "false"}, tmp_path)
    assert ok["passed"] is True
    assert bad["passed"] is False


def test_run_check_file_not_contains(tmp_path):
    (tmp_path / "f.txt").write_text("old_name is here")
    passed = run.run_check({"type": "file_not_contains", "path": "f.txt", "text": "new_name"}, tmp_path)
    failed = run.run_check({"type": "file_not_contains", "path": "f.txt", "text": "old_name"}, tmp_path)
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_run_check_file_not_contains_missing_file_fails(tmp_path):
    result = run.run_check({"type": "file_not_contains", "path": "missing.txt", "text": "x"}, tmp_path)
    assert result["passed"] is False


def test_run_check_unknown_type(tmp_path):
    result = run.run_check({"type": "not_a_real_type"}, tmp_path)
    assert result["passed"] is False


# --------------------------------------------------------------------------
# main(): CLI wiring, --json output, exit codes, --list
# --------------------------------------------------------------------------

def test_main_end_to_end_pass_writes_json_and_exits_zero(tmp_path):
    tasks_dir = tmp_path / "tasks"
    fixtures_dir = tmp_path / "fixtures"
    _write_task(tasks_dir, _synthetic_task())
    _make_fixture(fixtures_dir, "greet-file")
    script = _correct_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"
    json_path = tmp_path / "results.json"

    exit_code = run.main([
        "--tasks-dir", str(tasks_dir),
        "--fixtures-dir", str(fixtures_dir),
        "--agent-cmd", agent_cmd,
        "--json", str(json_path),
    ])

    assert exit_code == 0
    payload = json.loads(json_path.read_text())
    assert payload["passed"] == 1
    assert payload["total"] == 1
    assert payload["results"][0]["id"] == "greet-file"
    assert payload["results"][0]["status"] == "pass"


def test_main_end_to_end_fail_exits_nonzero(tmp_path):
    tasks_dir = tmp_path / "tasks"
    fixtures_dir = tmp_path / "fixtures"
    _write_task(tasks_dir, _synthetic_task())
    _make_fixture(fixtures_dir, "greet-file")
    script = _failing_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"
    json_path = tmp_path / "results.json"

    exit_code = run.main([
        "--tasks-dir", str(tasks_dir),
        "--fixtures-dir", str(fixtures_dir),
        "--agent-cmd", agent_cmd,
        "--json", str(json_path),
    ])

    assert exit_code == 1
    payload = json.loads(json_path.read_text())
    assert payload["passed"] == 0
    assert payload["total"] == 1
    assert payload["results"][0]["status"] == "fail"


def test_main_list_does_not_run_agent(tmp_path, capsys):
    tasks_dir = tmp_path / "tasks"
    fixtures_dir = tmp_path / "fixtures"
    _write_task(tasks_dir, _synthetic_task())
    # Deliberately do NOT create the fixture dir; --list must not touch it.
    # An agent-cmd that would explode if ever executed, to prove --list never runs it.
    agent_cmd = f"{sys.executable} -c \"import sys; sys.exit(99)\" {{prompt}}"

    exit_code = run.main([
        "--tasks-dir", str(tasks_dir),
        "--fixtures-dir", str(fixtures_dir),
        "--agent-cmd", agent_cmd,
        "--list",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "greet-file" in out


def test_main_tasks_filter_runs_only_selected(tmp_path):
    tasks_dir = tmp_path / "tasks"
    fixtures_dir = tmp_path / "fixtures"
    _write_task(tasks_dir, _synthetic_task(), filename="greet-file.yaml")
    other_task = dict(_synthetic_task())
    other_task["id"] = "other-task"
    other_task["fixture"] = "other-task"
    _write_task(tasks_dir, other_task, filename="other-task.yaml")
    _make_fixture(fixtures_dir, "greet-file")
    # deliberately no fixture for other-task: if it were run it would error out
    script = _correct_fake_agent_script(tmp_path)
    agent_cmd = f"{sys.executable} {script} {{prompt}}"
    json_path = tmp_path / "results.json"

    exit_code = run.main([
        "--tasks-dir", str(tasks_dir),
        "--fixtures-dir", str(fixtures_dir),
        "--agent-cmd", agent_cmd,
        "--tasks", "greet-file",
        "--json", str(json_path),
    ])

    assert exit_code == 0
    payload = json.loads(json_path.read_text())
    assert payload["total"] == 1
    assert payload["results"][0]["id"] == "greet-file"


def test_main_reports_error_task_and_nonzero_exit(tmp_path):
    tasks_dir = tmp_path / "tasks"
    fixtures_dir = tmp_path / "fixtures"
    _write_task(tasks_dir, _synthetic_task())
    # No fixture created -> should surface as an "error" result, not a crash.
    agent_cmd = f"{sys.executable} -c pass {{prompt}}"
    json_path = tmp_path / "results.json"

    exit_code = run.main([
        "--tasks-dir", str(tasks_dir),
        "--fixtures-dir", str(fixtures_dir),
        "--agent-cmd", agent_cmd,
        "--json", str(json_path),
    ])

    assert exit_code == 1
    payload = json.loads(json_path.read_text())
    assert payload["results"][0]["status"] == "error"
