"""Git-based snapshot/revert for incremental plan step execution.

Discovers all git repos under working_dir (including submodules), commits
a snapshot before a step begins, and reverts on failure.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RepoSnapshot:
    repo: str       # absolute path to repo root
    sha: str        # HEAD SHA after snapshot (or current HEAD if clean)
    was_dirty: bool # True = we made a snapshot commit; False = tree was clean

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RepoSnapshot":
        return cls(repo=d["repo"], sha=d["sha"], was_dirty=d.get("was_dirty", False))


def find_git_repos(working_dir: str, max_depth: int = 4) -> list[str]:
    """Return sorted list of git repo roots under working_dir (inclusive).

    Handles normal repos (.git dir) and submodules (.git file).
    """
    base = Path(working_dir).resolve()
    found: set[str] = set()

    def _walk(path: Path, depth: int) -> None:
        if (path / ".git").exists():
            found.add(str(path))
        if depth <= 0:
            return
        try:
            for child in sorted(path.iterdir()):
                if child.is_dir() and child.name != ".git":
                    _walk(child, depth - 1)
        except PermissionError:
            pass

    _walk(base, max_depth)
    return sorted(found)


def _git(*args: str, cwd: str, timeout: float = 30.0) -> tuple[str, str, int]:
    # Bound the call and disable interactive credential prompts so a snapshot
    # step can never hang the plan on a stuck git process.
    import os
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        r = subprocess.run(
            ["git"] + list(args), cwd=cwd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return "", f"git timed out after {timeout:.0f}s: git {' '.join(args)}", 124
    return r.stdout, r.stderr, r.returncode


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def snapshot_step(plan_id: str, step_id: str, working_dir: str) -> list[RepoSnapshot]:
    """Snapshot all git repos under working_dir. Returns one RepoSnapshot per repo.

    Dirty repos get a commit; clean repos record current HEAD (no commit made).
    Returns [] if git unavailable or no repos found.
    """
    if not _git_available():
        return []
    repos = find_git_repos(working_dir)
    if not repos:
        return []

    msg = f"owncoder-snap: {plan_id}:{step_id}"
    result: list[RepoSnapshot] = []

    for repo in repos:
        try:
            status_out, _, _ = _git("status", "--porcelain", cwd=repo)
            is_dirty = bool(status_out.strip())

            if is_dirty:
                _git("add", "-A", cwd=repo)
                _git("commit", "-m", msg, cwd=repo)

            sha_out, _, rc = _git("rev-parse", "HEAD", cwd=repo)
            sha = sha_out.strip() if rc == 0 else ""
            if sha:
                result.append(RepoSnapshot(repo=repo, sha=sha, was_dirty=is_dirty))
        except Exception:
            pass

    return result


def revert_to_snapshots(snapshots: list[RepoSnapshot]) -> list[tuple[str, bool, str]]:
    """Revert repos that were snapshotted (was_dirty=True) to their snapshot SHA.

    Returns list of (repo_path, ok, message).
    """
    results: list[tuple[str, bool, str]] = []
    for snap in snapshots:
        if not snap.was_dirty:
            continue
        try:
            out1, err1, rc1 = _git("reset", "--hard", snap.sha, cwd=snap.repo)
            out2, err2, rc2 = _git("clean", "-fd", cwd=snap.repo)
            ok = rc1 == 0 and rc2 == 0
            msg = " | ".join(filter(None, [out1.strip(), err1.strip(), out2.strip(), err2.strip()]))
            results.append((snap.repo, ok, msg))
        except Exception as exc:
            results.append((snap.repo, False, str(exc)))
    return results


def squash_snapshot(repo: str, message: str) -> tuple[bool, str]:
    """Squash the snapshot commit: git reset --soft HEAD~1 + git commit -m message."""
    try:
        _, err1, rc1 = _git("reset", "--soft", "HEAD~1", cwd=repo)
        if rc1 != 0:
            return False, err1.strip()
        out2, err2, rc2 = _git("commit", "-m", message, cwd=repo)
        return rc2 == 0, (out2 + err2).strip()
    except Exception as exc:
        return False, str(exc)
