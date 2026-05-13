"""Git-based file churn analysis: modification frequency, recently reworked
files, and refactoring candidates (large + frequently modified)."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from time import time
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

_config = None


def setup(config) -> None:
    global _config
    _config = config


def _working_dir() -> str:
    if _config:
        return _config.tools.working_dir
    return "."


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


@register(
    "project_file_stats",
    {
        "description": (
            "Analyze git commit history for file-level statistics: modification frequency, "
            "churn rate, recently-reworked files, and refactoring candidates. "
            "Returns files sorted by churn (most-modified first), plus a refactoring-candidates "
            "section identifying large files that are also frequently modified. "
            "This helps decide what to refactor, split, or simplify."
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
    }
    return result
