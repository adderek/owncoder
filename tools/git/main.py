from __future__ import annotations

import subprocess
from pathlib import Path
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


def _run_git(*args: str, cwd: str | None = None) -> tuple[str, str, int]:
    cwd = cwd or _working_dir()
    result = subprocess.run(
        ["git"] + list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


@register(
    "git_diff",
    {
        "description": "Show git diff of changes. staged=true for staged changes, false for unstaged.",
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes (default: false)",
                },
                "path": {
                    "type": "string",
                    "description": "Limit diff to this file path",
                },
            },
            "required": [],
        },
    },
)
def git_diff(staged: bool = False, path: str | None = None) -> dict:
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args += ["--", path]
    stdout, stderr, rc = _run_git(*args)
    if rc != 0 and stderr:
        return {"error": stderr}
    return {"diff": stdout, "staged": staged}


@register(
    "git_log",
    {
        "description": "Show recent git commits, optionally filtered to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filter to commits touching this file",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of commits (default: 10)",
                },
                "format": {
                    "type": "string",
                    "description": "Log format: oneline, short, medium (default: oneline)",
                },
            },
            "required": [],
        },
    },
)
def git_log(path: str | None = None, n: int = 10, format: str = "oneline") -> dict:
    _allowed_formats = {"oneline", "short", "medium", "full", "fuller"}
    safe_format = format if format in _allowed_formats else "oneline"
    args = [
        "log",
        "--oneline" if safe_format == "oneline" else f"--format={safe_format}",
        f"-{n}",
    ]
    if path:
        args += ["--", path]
    stdout, stderr, rc = _run_git(*args)
    if rc != 0:
        return {"error": stderr or "git log failed"}
    return {"log": stdout, "n": n}


@register(
    "git_blame",
    {
        "description": "Show who changed each line in a file between start_line and end_line.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File to blame"},
                "start_line": {"type": "integer", "description": "First line"},
                "end_line": {"type": "integer", "description": "Last line"},
            },
            "required": ["path"],
        },
    },
)
def git_blame(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> dict:
    args = ["blame", "--porcelain"]
    if start_line and end_line:
        args += [f"-L{start_line},{end_line}"]
    elif start_line:
        args += [f"-L{start_line},+50"]
    args += ["--", path]
    stdout, stderr, rc = _run_git(*args)
    if rc != 0:
        return {"error": stderr or "git blame failed"}

    # Parse porcelain output into structured form.
    # Each record starts with a 40-hex-char commit hash followed by line numbers.
    import re as _re

    _hash_re = _re.compile(r"^([0-9a-f]{40}) \d+ (\d+)")
    entries = []
    current: dict = {}
    for line in stdout.splitlines():
        m = _hash_re.match(line)
        if m:
            current = {"hash": m.group(1), "lineno": int(m.group(2))}
        elif line.startswith("author "):
            current["author"] = line[7:]
        elif line.startswith("author-time "):
            current["timestamp"] = int(line[12:])
        elif line.startswith("summary "):
            current["summary"] = line[8:]
        elif line.startswith("\t"):
            current["content"] = line[1:]
            entries.append(current)
            current = {}

    return {"blame": entries, "path": path}


@register(
    "git_status",
    {
        "description": "Show current git status: branch, staged, unstaged, untracked files.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
)
def git_status() -> dict:
    stdout, stderr, rc = _run_git("status", "--porcelain=v1", "-b")
    if rc != 0:
        return {"error": stderr or "git status failed"}

    lines = stdout.splitlines()
    branch = ""
    staged = []
    unstaged = []
    untracked = []

    for line in lines:
        if line.startswith("## "):
            branch_info = line[3:]
            branch = branch_info.split("...")[0]
            continue
        if len(line) < 2:
            continue
        x, y = line[0], line[1]
        fname = line[3:]
        if x != " " and x != "?":
            staged.append({"status": x, "file": fname})
        if y != " " and y != "?":
            unstaged.append({"status": y, "file": fname})
        if x == "?" and y == "?":
            untracked.append(fname)

    return {
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }


@register(
    "git_related_files",
    {
        "description": "Find files most often committed together with the given file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path to find related files for",
                },
            },
            "required": ["path"],
        },
    },
)
def git_related_files(path: str) -> dict:
    out, _, rc = _run_git("log", "-n", "50", "--format=", "--name-only", "--", path)
    if rc != 0 or not out.strip():
        return {"related": [], "path": path}

    counts: dict[str, int] = {}
    for f in out.splitlines():
        f = f.strip()
        if f and f != path:
            counts[f] = counts.get(f, 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"related": [{"file": f, "count": c} for f, c in ranked], "path": path}
