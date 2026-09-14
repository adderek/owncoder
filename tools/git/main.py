from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools._common import working_dir, is_read_protected, is_path_allowed

if TYPE_CHECKING:
    from agent.config import Config

_config = None

# Tools that surface file *contents* must refuse secret files, matching
# read_file / grep_code — otherwise the model could read a tracked .env via
# git blame/diff.
_READ_PROTECTED_ERR = "path is a protected secret file; refusing to expose its contents"


def setup(config) -> None:
    global _config
    _config = config


def _working_dir() -> str:
    return working_dir(_config)


def _repo_dir(repo: str | None) -> tuple[str | None, str | None]:
    """Resolve *repo* (a subdirectory such as a submodule) under the project
    root. Returns (cwd, error). None means the project root itself."""
    if not repo:
        return None, None
    root = Path(_working_dir()).resolve()
    target = (root / repo).resolve()
    if not is_path_allowed(target, root):
        return None, f"repo {repo!r} is outside the project root"
    if not target.is_dir():
        return None, f"repo {repo!r} is not a directory"
    return str(target), None


# Read-only git must not execute anything configured by the repository:
# fsmonitor hooks run on status/diff, external diff drivers and textconv
# filters run on diff/blame. The agent cannot write .git/** (security/fs.py),
# but a checkout received with its .git dir can carry such config.
_GIT_PREFIX = ("--no-pager", "-c", "core.fsmonitor=false")
_DIFF_SAFE = ("--no-ext-diff", "--no-textconv")


def _run_git(*args: str, cwd: str | None = None, timeout: float = 30.0) -> tuple[str, str, int]:
    cwd = cwd or _working_dir()
    # GIT_TERMINAL_PROMPT=0 stops git from blocking on an interactive credential
    # prompt; the timeout bounds index.lock contention and pathological repos so
    # an LLM-invoked git call can never hang the agent turn indefinitely.
    # GIT_OPTIONAL_LOCKS=0 keeps status from taking index.lock at all, so it
    # never collides with the user's own git in the same tree.
    import os
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
    try:
        result = subprocess.run(
            ["git", *_GIT_PREFIX, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "", f"git timed out after {timeout:.0f}s: git {' '.join(args)}", 124
    return result.stdout, result.stderr, result.returncode


_REPO_PARAM = {
    "type": "string",
    "description": "Subdirectory repo to run in, e.g. a submodule path (default: project root)",
}


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
                "repo": _REPO_PARAM,
            },
            "required": [],
        },
    },
)
def git_diff(staged: bool = False, path: str | None = None, repo: str | None = None) -> dict:
    if path and is_read_protected(path):
        return {"error": _READ_PROTECTED_ERR, "path": path}
    cwd, err = _repo_dir(repo)
    if err:
        return {"error": err}
    base = ["diff", *_DIFF_SAFE, "--no-renames"]
    if staged:
        base.append("--cached")
    scope = ["--", path] if path else []
    # A diff without a path (or with a directory path) can still include a
    # secret file. List the changed files first and exclude protected ones by
    # literal pathspec; --no-renames keeps a renamed secret under its own name.
    names, stderr, rc = _run_git(*base, "--name-only", "-z", *scope, cwd=cwd)
    if rc != 0:
        return {"error": stderr or "git diff failed"}
    withheld = [n for n in names.split("\0") if n and is_read_protected(n)]
    if withheld:
        scope = scope or ["--"]
        if len(scope) == 1:
            scope.append(".")
        scope += [f":(exclude,literal){n}" for n in withheld]
    stdout, stderr, rc = _run_git(*base, *scope, cwd=cwd)
    if rc != 0 and stderr:
        return {"error": stderr}
    result = {"diff": stdout, "staged": staged}
    if withheld:
        result["withheld"] = withheld
    return result


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
                "repo": _REPO_PARAM,
            },
            "required": [],
        },
    },
)
def git_log(path: str | None = None, n: int = 10, format: str = "oneline",
            repo: str | None = None) -> dict:
    cwd, err = _repo_dir(repo)
    if err:
        return {"error": err}
    _allowed_formats = {"oneline", "short", "medium", "full", "fuller"}
    safe_format = format if format in _allowed_formats else "oneline"
    args = [
        "log",
        "--oneline" if safe_format == "oneline" else f"--format={safe_format}",
        f"-{n}",
    ]
    if path:
        args += ["--", path]
    stdout, stderr, rc = _run_git(*args, cwd=cwd)
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
                "repo": _REPO_PARAM,
            },
            "required": ["path"],
        },
    },
)
def git_blame(
    path: str, start_line: int | None = None, end_line: int | None = None,
    repo: str | None = None,
) -> dict:
    if is_read_protected(path):
        return {"error": _READ_PROTECTED_ERR, "path": path}
    cwd, err = _repo_dir(repo)
    if err:
        return {"error": err}
    args = ["blame", "--porcelain", "--no-textconv"]
    if start_line and end_line:
        args += [f"-L{start_line},{end_line}"]
    elif start_line:
        args += [f"-L{start_line},+50"]
    args += ["--", path]
    stdout, stderr, rc = _run_git(*args, cwd=cwd)
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
            "properties": {"repo": _REPO_PARAM},
            "required": [],
        },
    },
)
def git_status(repo: str | None = None) -> dict:
    cwd, err = _repo_dir(repo)
    if err:
        return {"error": err}
    stdout, stderr, rc = _run_git("status", "--porcelain=v1", "-b", cwd=cwd)
    if rc != 0:
        return {"error": stderr or "git status failed"}

    lines = stdout.splitlines()
    branch = None
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

    if branch is None:
        return {"error": "failed to parse git status: no branch line found"}

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
