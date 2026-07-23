#!/usr/bin/env python3
"""Eval harness runner.

Runs the coding agent (or any --agent-cmd override) against a set of small,
repeatable coding tasks defined in evals/tasks/*.yaml, then checks the
resulting workspace against per-task success criteria.

See evals/README.md for usage and how to add a task.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised only when pyyaml is absent
    yaml = None

EVALS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVALS_DIR.parent
TASKS_DIR = EVALS_DIR / "tasks"
FIXTURES_DIR = EVALS_DIR / "fixtures"

DEFAULT_TIMEOUT_S = 300
CHECK_TIMEOUT_S = 60


@dataclass
class TaskResult:
    id: str
    status: str  # "pass", "fail", or "error"
    elapsed: float = 0.0
    exit_code: int | None = None
    timed_out: bool = False
    checks: list = field(default_factory=list)
    workspace: str | None = None
    message: str = ""
    judge: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _task_file_glob() -> str:
    return "*.yaml" if yaml is not None else "*.json"


def _load_task_file(path: Path) -> dict:
    text = path.read_text()
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"task file {path.name} did not contain a mapping")
    return data


def load_tasks(tasks_dir: Path = TASKS_DIR) -> tuple[list[dict], list[tuple[str, str]]]:
    """Load all task definitions. Returns (tasks, errors).

    A malformed task file is reported in `errors` (id derived from filename)
    rather than raising, so one bad file doesn't crash the whole run.
    """
    tasks: list[dict] = []
    errors: list[tuple[str, str]] = []
    if not tasks_dir.is_dir():
        return tasks, errors
    for path in sorted(tasks_dir.glob(_task_file_glob())):
        task_id = path.stem
        try:
            data = _load_task_file(path)
            if "id" not in data:
                raise ValueError("missing required 'id' field")
            if "prompt" not in data:
                raise ValueError("missing required 'prompt' field")
            if "fixture" not in data:
                raise ValueError("missing required 'fixture' field")
            tasks.append(data)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, reported per-task
            errors.append((task_id, f"malformed task file: {exc}"))
    return tasks, errors


def default_agent_cmd(repo_root: Path = REPO_ROOT) -> str:
    agent_bin = repo_root / ".venv" / "bin" / "agent"
    return f"{shlex.quote(str(agent_bin))} run {{prompt}}"


def _subprocess_env() -> dict:
    """Environment for spawned subprocesses.

    Puts the running interpreter's own bin directory (typically
    `<repo>/.venv/bin`) first on PATH, so a bare `python` in a check's
    `run:` command (or in an --agent-cmd override) resolves to the same
    interpreter running the harness -- and its installed packages (e.g.
    pytest) -- regardless of whether the venv was actually activated in
    the calling shell.
    """
    env = os.environ.copy()
    # Deliberately do NOT resolve symlinks: venvs typically symlink
    # bin/python -> a base interpreter, and resolving would walk right past
    # the venv's own bin/ (and its site-packages) to the system one.
    interpreter_dir = str(Path(sys.executable).parent)
    env["PATH"] = interpreter_dir + os.pathsep + env.get("PATH", "")
    return env


def build_command(template: str, prompt: str) -> str:
    """Fill in the agent command template with the task prompt.

    If the template contains the literal placeholder ``{prompt}`` it is
    substituted (shell-quoted). Otherwise the prompt is appended as the
    final argument (also shell-quoted).
    """
    quoted = shlex.quote(prompt)
    if "{prompt}" in template:
        return template.replace("{prompt}", quoted)
    return f"{template} {quoted}"


def run_check(check: dict, workspace: Path) -> dict[str, Any]:
    ctype = check.get("type")
    try:
        if ctype == "command":
            run = check.get("run", "")
            proc = subprocess.run(
                run,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=CHECK_TIMEOUT_S,
                env=_subprocess_env(),
            )
            passed = proc.returncode == 0
            detail = "" if passed else f"`{run}` exited {proc.returncode}: {proc.stderr.strip()[-300:]}"
            return {"type": ctype, "passed": passed, "detail": detail}

        if ctype == "file_exists":
            path = check.get("path", "")
            passed = (workspace / path).is_file()
            return {"type": ctype, "passed": passed, "detail": "" if passed else f"{path} does not exist"}

        if ctype == "file_contains":
            path = check.get("path", "")
            text = check.get("text", "")
            target = workspace / path
            if not target.is_file():
                return {"type": ctype, "passed": False, "detail": f"{path} does not exist"}
            content = target.read_text(errors="replace")
            passed = text in content
            return {"type": ctype, "passed": passed, "detail": "" if passed else f"{path} does not contain {text!r}"}

        if ctype == "file_not_contains":
            path = check.get("path", "")
            text = check.get("text", "")
            target = workspace / path
            if not target.is_file():
                return {"type": ctype, "passed": False, "detail": f"{path} does not exist"}
            content = target.read_text(errors="replace")
            passed = text not in content
            return {"type": ctype, "passed": passed, "detail": "" if passed else f"{path} still contains {text!r}"}

        return {"type": ctype, "passed": False, "detail": f"unknown check type: {ctype!r}"}
    except subprocess.TimeoutExpired:
        return {"type": ctype, "passed": False, "detail": "check command timed out"}
    except Exception as exc:  # noqa: BLE001
        return {"type": ctype, "passed": False, "detail": str(exc)}


def _judge_module():
    """Import evals/judge.py whether run.py was launched as a script or a
    module."""
    try:
        from evals import judge as judge_mod  # type: ignore
    except ImportError:
        if str(EVALS_DIR) not in sys.path:
            sys.path.insert(0, str(EVALS_DIR))
        import judge as judge_mod  # type: ignore
    return judge_mod


def _judge_workspace(task: dict, fixture_dir: Path, workspace: Path,
                     judge_config) -> dict:
    """Diff the workspace against its fixture and score it. Never raises —
    a judge failure is reported in the returned dict, not fatal."""
    judge_mod = _judge_module()
    try:
        diff = judge_mod.compute_diff(fixture_dir, workspace)
    except Exception as exc:  # noqa: BLE001
        return {"score": None, "error": f"diff failed: {exc}"}
    import asyncio
    return asyncio.run(judge_mod.judge_task(judge_config, task, diff))


def run_task(task: dict, agent_cmd_template: str, keep: bool,
             fixtures_dir: Path = FIXTURES_DIR,
             judge_config=None) -> TaskResult:
    task_id = task.get("id", "<unknown>")
    fixture_name = task.get("fixture", "")
    fixture_dir = fixtures_dir / fixture_name
    if not fixture_dir.is_dir():
        return TaskResult(id=task_id, status="error",
                           message=f"fixture not found: {fixture_dir}")

    prompt = task.get("prompt", "")
    timeout_s = task.get("timeout_s", DEFAULT_TIMEOUT_S)
    checks = task.get("checks", [])

    workspace = Path(tempfile.mkdtemp(prefix=f"eval-{task_id}-"))
    shutil.copytree(fixture_dir, workspace, dirs_exist_ok=True)
    # `agent run` refuses to start outside an initialized project; the marker
    # is a bare .agent/ directory (full `agent init` would try to index/embed).
    (workspace / ".agent").mkdir(exist_ok=True)

    command = build_command(agent_cmd_template, prompt)

    start = time.monotonic()
    exit_code: int | None = None
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_subprocess_env(),
        )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed = time.monotonic() - start

    if timed_out:
        check_results = [
            {"type": c.get("type"), "passed": False, "detail": "agent timed out"}
            for c in checks
        ]
    else:
        check_results = [run_check(c, workspace) for c in checks]

    passed = (not timed_out) and all(c["passed"] for c in check_results)
    status = "pass" if passed else "fail"

    judge_result: dict | None = None
    if judge_config is not None and not timed_out:
        judge_result = _judge_workspace(task, fixture_dir, workspace, judge_config)

    workspace_path: str | None = None
    if keep:
        workspace_path = str(workspace)
    else:
        shutil.rmtree(workspace, ignore_errors=True)

    return TaskResult(
        id=task_id,
        status=status,
        elapsed=elapsed,
        exit_code=exit_code,
        timed_out=timed_out,
        checks=check_results,
        workspace=workspace_path,
        judge=judge_result,
    )


def print_summary(results: list[TaskResult]) -> None:
    judged_any = any(r.judge is not None for r in results)
    header = f"{'TASK':<24} {'RESULT':<7} {'TIME':>8}"
    if judged_any:
        header += f" {'JUDGE':>6}"
    header += "  FAILED CHECKS"
    print(header)
    print("-" * len(header))
    for r in results:
        time_str = f"{r.elapsed:.1f}s" if r.elapsed else "-"
        if r.status == "error":
            detail = r.message
        else:
            failed = [c["type"] for c in r.checks if not c["passed"]]
            detail = ", ".join(failed) if failed else "-"
        line = f"{r.id:<24} {r.status:<7} {time_str:>8}"
        if judged_any:
            if r.judge is None:
                jstr = "-"
            elif r.judge.get("score") is None:
                jstr = "ERR"
            else:
                jstr = str(r.judge["score"])
            line += f" {jstr:>6}"
        print(f"{line}  {detail}")
    print()
    for r in results:
        if r.workspace:
            print(f"kept workspace for {r.id}: {r.workspace}")
        if r.judge and r.judge.get("error"):
            print(f"judge error for {r.id}: {r.judge['error']}")
    passed = sum(1 for r in results if r.status == "pass")
    total = len(results)
    print(f"\nScore: {passed}/{total} passed")
    scores = [r.judge["score"] for r in results
              if r.judge and r.judge.get("score") is not None]
    if scores:
        print(f"Judged: mean {sum(scores) / len(scores):.1f}/10 "
              f"over {len(scores)} task(s)")


def write_json(results: list[TaskResult], path: Path) -> None:
    payload = {
        "results": [r.as_dict() for r in results],
        "passed": sum(1 for r in results if r.status == "pass"),
        "total": len(results),
    }
    path.write_text(json.dumps(payload, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent coding evals.")
    parser.add_argument("--list", action="store_true",
                         help="list available tasks and exit")
    parser.add_argument("--tasks", default=None,
                         help="comma-separated list of task ids to run (default: all)")
    parser.add_argument("--agent-cmd", default=None,
                         help="override the agent invocation command; "
                              "use {prompt} as a placeholder for the task prompt "
                              "(default: <repo>/.venv/bin/agent run {prompt})")
    parser.add_argument("--keep", action="store_true",
                         help="keep temp workspaces and print their paths")
    parser.add_argument("--json", dest="json_path", default=None,
                         help="write machine-readable results to PATH")
    parser.add_argument("--judge", action="store_true",
                         help="score each task's diff 0-10 with the LLM judge "
                              "(role 'judge'; requires an agent config). "
                              "Mechanical pass/fail stays primary.")
    parser.add_argument("--baseline", default=None,
                         help="previous --json results file to compare against; "
                              "a mechanical pass->fail flip or a judged drop "
                              ">2 points is a regression and fails the run")
    parser.add_argument("--tasks-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fixtures-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else TASKS_DIR
    fixtures_dir = Path(args.fixtures_dir) if args.fixtures_dir else FIXTURES_DIR

    tasks, load_errors = load_tasks(tasks_dir)

    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        tasks = [t for t in tasks if t.get("id") in wanted]
        load_errors = [(tid, err) for tid, err in load_errors if tid in wanted]

    if args.list:
        for t in tasks:
            print(f"{t['id']:<20} fixture={t.get('fixture', '?'):<20} {t.get('prompt', '')[:60]}")
        for tid, err in load_errors:
            print(f"{tid:<20} ERROR: {err}")
        return 0

    agent_cmd_template = args.agent_cmd or default_agent_cmd(REPO_ROOT)

    judge_config = None
    if args.judge:
        try:
            judge_config = _judge_module().load_agent_config()
        except Exception as exc:  # noqa: BLE001
            print(f"--judge: failed to load agent config: {exc}", file=sys.stderr)
            return 1

    results: list[TaskResult] = [
        TaskResult(id=tid, status="error", message=err) for tid, err in load_errors
    ]
    for task in tasks:
        results.append(run_task(task, agent_cmd_template, args.keep,
                                fixtures_dir, judge_config=judge_config))

    print_summary(results)

    if args.json_path:
        write_json(results, Path(args.json_path))

    regressions: list[str] = []
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            print(f"--baseline: no such file: {baseline_path}", file=sys.stderr)
            return 1
        baseline = json.loads(baseline_path.read_text())
        current = {"results": [r.as_dict() for r in results]}
        regressions = _judge_module().compare_to_baseline(current, baseline)
        if regressions:
            print("\nRegressions vs baseline:")
            for reg in regressions:
                print(f"  {reg}")
        else:
            print("\nNo regressions vs baseline.")
        return 1 if regressions else 0

    all_passed = all(r.status == "pass" for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
