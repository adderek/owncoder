"""File churn analysis: git history + agent-tracked edit events.

Combines git commit frequency/size with the persistent `.agent/edit_stats.jsonl`
log to identify refactoring candidates — large files edited repeatedly.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools._common import working_dir

if TYPE_CHECKING:
    from agent.config import Config

_config = None


def setup(config) -> None:
    global _config
    _config = config


def _working_dir() -> str:
    return working_dir(_config)


def _run_git(*args: str, cwd: str | None = None) -> tuple[str, int]:
    cwd = cwd or _working_dir()
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.returncode


def _count_lines_fast(fpath: str, base_dir: str) -> int:
    """Count lines without reading full file into memory."""
    full = Path(base_dir) / fpath
    if not full.is_file():
        return 0
    try:
        with open(full, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _format_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.0f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.0f}TB"


def _read_agent_edit_log(wd: str) -> list[dict]:
    """Read .agent/edit_stats.jsonl records. Returns [] if missing or unreadable."""
    if _config:
        agent_dir = Path(_config.tools.agent_dir)
        if not agent_dir.is_absolute():
            agent_dir = Path(wd) / agent_dir
    else:
        agent_dir = Path(wd) / ".agent"
    log_path = agent_dir / "edit_stats.jsonl"
    if not log_path.is_file():
        return []
    records = []
    try:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return records


def _agent_tracked_stats(wd: str, scope: str, max_files: int) -> dict:
    """Aggregate agent edit-log into per-file stats and refactor candidates."""
    import fnmatch

    records = _read_agent_edit_log(wd)
    if not records:
        return {"available": False, "reason": "no edit_stats.jsonl found"}

    per_file: dict[str, dict] = {}
    for rec in records:
        path = rec.get("path", "")
        if not path or path == "<multi>":
            continue
        if not fnmatch.fnmatch(path, scope):
            continue
        outcome = rec.get("outcome", "")
        if path not in per_file:
            per_file[path] = {"path": path, "agent_edits": 0, "size_bytes": 0, "lines": 0, "last_edit_ts": 0.0}
        entry = per_file[path]
        if outcome == "ok":
            entry["agent_edits"] += 1
            ts = rec.get("ts", 0.0)
            if ts > entry["last_edit_ts"]:
                entry["last_edit_ts"] = ts
                if "size_bytes" in rec:
                    entry["size_bytes"] = rec["size_bytes"]
                if "lines" in rec:
                    entry["lines"] = rec["lines"]

    if not per_file:
        return {"available": True, "files": [], "refactoring_candidates": []}

    # For files with no size snapshot yet (edited before this feature), read live
    for entry in per_file.values():
        if entry["size_bytes"] == 0:
            fpath = Path(wd) / entry["path"]
            if fpath.is_file():
                entry["size_bytes"] = fpath.stat().st_size
                try:
                    with open(fpath, "rb") as f:
                        entry["lines"] = sum(1 for _ in f)
                except OSError:
                    pass

    files_list = sorted(per_file.values(), key=lambda e: -e["agent_edits"])[:max_files]
    for entry in files_list:
        entry["size_str"] = _format_size(entry["size_bytes"])
        entry["days_since_edit"] = round(max(0.0, (time() - entry["last_edit_ts"]) / 86400), 1) if entry["last_edit_ts"] else None

    refactor = [
        e for e in per_file.values()
        if e["lines"] >= 300 and e["agent_edits"] >= 3
    ]
    refactor.sort(key=lambda e: -(e["agent_edits"] * max(e["lines"], 1)))
    for entry in refactor[:max_files]:
        entry.setdefault("size_str", _format_size(entry["size_bytes"]))

    return {
        "available": True,
        "total_edit_events": sum(e["agent_edits"] for e in per_file.values()),
        "files": files_list,
        "refactoring_candidates": refactor[:max_files],
    }


@register(
    "project_file_stats",
    {
        "description": (
            "Analyze file-level statistics for refactoring decisions: git commit frequency/churn "
            "and agent-tracked edit events from .agent/edit_stats.jsonl. "
            "Returns files sorted by churn (most-modified first), recently-reworked files, "
            "refactoring candidates (large + frequently modified), and an agent_tracked section "
            "showing files the agent has edited most — including size/line count snapshots and "
            "candidates not yet visible in git history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "File glob pattern to scope analysis (default: '**/*.py' for Python projects)",
                    "default": "**/*.py",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Max files to report per section (default: 20)",
                    "default": 20,
                },
                "commits": {
                    "type": "integer",
                    "description": "Number of recent commits to scan (default: 200)",
                    "default": 200,
                },
                "large_file_threshold": {
                    "type": "integer",
                    "description": "Lines threshold for 'large file' classification (default: 500)",
                    "default": 500,
                },
            },
            "required": [],
        },
    },
)
def project_file_stats(
    scope: str = "**/*.py",
    max_files: int = 20,
    commits: int = 200,
    large_file_threshold: int = 500,
) -> dict:
    wd = _working_dir()
    result: dict[str, object] = {}

    # ── Step 1: get file commit counts (git log --name-only) ──────────
    out, rc = _run_git("log", f"-{commits}", "--format=COMMIT:%H%nTS:%ct", "--name-only", cwd=wd)
    if rc != 0:
        return {"error": "Not a git repository or git failed"}

    commit_count: dict[str, int] = {}
    commit_timestamps: dict[str, list[int]] = defaultdict(list)
    current_commit = ""
    current_ts = 0
    in_commit_block = False

    for line in out.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("COMMIT:"):
            current_commit = line_stripped[7:]
            current_ts = 0
            in_commit_block = False
            continue
        if line_stripped.startswith("TS:"):
            try:
                current_ts = int(line_stripped[3:])
            except ValueError:
                current_ts = 0
            in_commit_block = True
            continue
        # Blank line separates commits, reset file tracking
        if not line_stripped:
            in_commit_block = False
            continue
        # File name
        if in_commit_block and line_stripped:
            commit_count[line_stripped] = commit_count.get(line_stripped, 0) + 1
            if current_ts:
                commit_timestamps[line_stripped].append(current_ts)

    if not commit_count:
        return {"error": "No file history found in recent commits"}

    # ── Step 2: get per-file diff stat for churn ──────────────────────
    diff_out, rc2 = _run_git(
        "log", f"-{commits}", "--format=", "--numstat", f"--diff-filter=AM",
        cwd=wd,
    )
    diff_churn: dict[str, int] = {}
    if rc2 == 0:
        for line in diff_out.splitlines():
            parts = line.strip().split("\t", 2)
            if len(parts) == 3:
                added_str, deleted_str, fpath = parts
                added = 0
                deleted = 0
                try:
                    added = int(added_str) if added_str != "-" else 0
                    deleted = int(deleted_str) if deleted_str != "-" else 0
                except ValueError:
                    pass
                if fpath:
                    diff_churn[fpath] = diff_churn.get(fpath, 0) + added + deleted

    # ── Step 3: filter by scope, get live metadata ────────────────────
    import fnmatch

    candidates: list[dict] = []
    for fpath, count in commit_count.items():
        if not fnmatch.fnmatch(fpath, scope):
            continue

        lines_count = _count_lines_fast(fpath, wd)
        abs_path = Path(wd) / fpath
        size_bytes = abs_path.stat().st_size if abs_path.is_file() else 0
        churn = diff_churn.get(fpath, 0)

        timestamps = commit_timestamps.get(fpath, [])
        timestamps.sort(reverse=True)
        days_since_modified = 0
        if timestamps:
            days_since_modified = max(0.0, (time() - timestamps[0]) / 86400)

        candidates.append({
            "path": fpath,
            "commits": count,
            "churn": churn,
            "lines": lines_count,
            "size": size_bytes,
            "size_str": _format_size(size_bytes),
            "days_since_modified": round(days_since_modified, 1),
        })

    # ── Step 4: sort and categorize ────────────────────────────────────
    # High-churn / high-modification files (sorted by commit count)
    by_churn = sorted(candidates, key=lambda c: -c["commits"])[:max_files]

    # Recently reworked (modified within last 7 days, sorted by churn)
    recent = [c for c in candidates if c["days_since_modified"] <= 7]
    recent.sort(key=lambda c: -c["churn"])
    recent = recent[:max_files]

    # Refactoring candidates: large AND frequently modified
    refactor_candidates = [
        c for c in candidates
        if c["lines"] >= large_file_threshold and c["commits"] >= 3
    ]
    refactor_candidates.sort(key=lambda c: -(c["commits"] * c["lines"]))
    refactor_candidates = refactor_candidates[:max_files]

    result = {
        "total_files_analyzed": len(candidates),
        "commit_window": commits,
        "by_frequency": by_churn,
        "recently_reworked": recent,
        "refactoring_candidates": refactor_candidates,
        "summary": {
            "most_modified": len(by_churn) > 0 and by_churn[0]["path"] or "",
            "most_churned": len(by_churn) > 0 and by_churn[0]["path"] or "",
        },
        "agent_tracked": _agent_tracked_stats(wd, scope, max_files),
    }
    return result
