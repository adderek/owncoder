"""Tests for planning/increment.py and tools/increment_tools.py."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.planning import plan as plan_mod
from agent.planning.increment import (
    RepoSnapshot,
    find_git_repos,
    snapshot_step,
    revert_to_snapshots,
    squash_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(path), capture_output=True, check=True,
    )


def _git_initial_commit(path: Path) -> None:
    """Create an initial commit so HEAD exists."""
    (path / "README").write_text("init")
    subprocess.run(["git", "add", "-A"], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(path), capture_output=True, check=True,
    )


def _head_sha(path: Path) -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(path), capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def _is_git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


pytestmark = pytest.mark.skipif(
    not _is_git_available(), reason="git not available"
)


# ---------------------------------------------------------------------------
# find_git_repos
# ---------------------------------------------------------------------------

class TestFindGitRepos:
    def test_no_git(self, tmp_path):
        assert find_git_repos(str(tmp_path)) == []

    def test_single_repo_at_root(self, tmp_path):
        _git_init(tmp_path)
        repos = find_git_repos(str(tmp_path))
        assert repos == [str(tmp_path)]

    def test_nested_repos(self, tmp_path):
        _git_init(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        _git_init(sub)
        repos = find_git_repos(str(tmp_path))
        assert str(tmp_path) in repos
        assert str(sub) in repos
        assert len(repos) == 2

    def test_git_file_submodule_included(self, tmp_path):
        sub = tmp_path / "submod"
        sub.mkdir()
        # Simulate a submodule: .git as a file
        (sub / ".git").write_text("gitdir: ../.git/modules/submod")
        repos = find_git_repos(str(tmp_path))
        assert str(sub) in repos

    def test_no_git_dir_itself_in_results(self, tmp_path):
        _git_init(tmp_path)
        repos = find_git_repos(str(tmp_path))
        for r in repos:
            assert not r.endswith("/.git")


# ---------------------------------------------------------------------------
# snapshot_step
# ---------------------------------------------------------------------------

class TestSnapshotStep:
    def test_no_git_repo(self, tmp_path):
        snaps = snapshot_step("plan1", "s1", str(tmp_path))
        assert snaps == []

    def test_clean_repo_records_head(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        snaps = snapshot_step("plan1", "s1", str(tmp_path))
        assert len(snaps) == 1
        assert snaps[0].was_dirty is False
        assert snaps[0].sha == _head_sha(tmp_path)

    def test_dirty_repo_creates_commit(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        head_before = _head_sha(tmp_path)
        (tmp_path / "new_file.txt").write_text("hello")
        snaps = snapshot_step("plan1", "s1", str(tmp_path))
        assert len(snaps) == 1
        assert snaps[0].was_dirty is True
        assert snaps[0].sha != head_before
        assert snaps[0].sha == _head_sha(tmp_path)

    def test_dirty_commit_message(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        (tmp_path / "x.txt").write_text("x")
        snapshot_step("myplan", "step2", str(tmp_path))
        log = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=str(tmp_path), capture_output=True, text=True,
        ).stdout.strip()
        assert log == "owncoder-snap: myplan:step2"

    def test_multi_repo(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        _git_init(sub)
        _git_initial_commit(sub)
        (sub / "change.txt").write_text("changed")
        snaps = snapshot_step("p", "s", str(tmp_path))
        assert len(snaps) == 2
        sub_snap = next(s for s in snaps if s.repo == str(sub))
        assert sub_snap.was_dirty is True


# ---------------------------------------------------------------------------
# revert_to_snapshots
# ---------------------------------------------------------------------------

class TestRevertToSnapshots:
    def test_reverts_dirty_repo(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        # Make repo dirty so snapshot commits (was_dirty=True)
        (tmp_path / "pre_snap.txt").write_text("pre-snap work")
        snaps = snapshot_step("p", "s1", str(tmp_path))
        assert snaps[0].was_dirty is True
        # Changes after snapshot should disappear on revert
        (tmp_path / "after_snap.txt").write_text("should disappear")
        results = revert_to_snapshots(snaps)
        assert len(results) == 1
        repo, ok, _ = results[0]
        assert ok
        # pre_snap.txt was committed into snap so it still exists
        assert (tmp_path / "pre_snap.txt").exists()
        assert not (tmp_path / "after_snap.txt").exists()

    def test_clean_repos_skipped(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        snaps = snapshot_step("p", "s1", str(tmp_path))
        assert snaps[0].was_dirty is False
        results = revert_to_snapshots(snaps)
        assert results == []  # nothing to revert

    def test_removes_untracked_files(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        (tmp_path / "new.txt").write_text("new")
        snaps = snapshot_step("p", "s", str(tmp_path))
        (tmp_path / "untracked_after.txt").write_text("gone")
        revert_to_snapshots(snaps)
        assert not (tmp_path / "untracked_after.txt").exists()


# ---------------------------------------------------------------------------
# squash_snapshot
# ---------------------------------------------------------------------------

class TestSquashSnapshot:
    def test_squash_replaces_not_removes_commit(self, tmp_path):
        """squash replaces the snap commit in-place; count stays the same."""
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        (tmp_path / "work.txt").write_text("work")
        snapshot_step("p", "s", str(tmp_path))
        count_before = int(subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(tmp_path), capture_output=True, text=True,
        ).stdout.strip())
        ok, _ = squash_snapshot(str(tmp_path), "feat: real message")
        assert ok
        count_after = int(subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=str(tmp_path), capture_output=True, text=True,
        ).stdout.strip())
        assert count_after == count_before  # replaced in-place

    def test_squash_sets_message(self, tmp_path):
        _git_init(tmp_path)
        _git_initial_commit(tmp_path)
        (tmp_path / "work.txt").write_text("work")
        snapshot_step("p", "s", str(tmp_path))
        squash_snapshot(str(tmp_path), "feat: real message")
        log = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=str(tmp_path), capture_output=True, text=True,
        ).stdout.strip()
        assert log == "feat: real message"


# ---------------------------------------------------------------------------
# RepoSnapshot serialization
# ---------------------------------------------------------------------------

class TestRepoSnapshotSerialization:
    def test_round_trip(self):
        snap = RepoSnapshot(repo="/some/path", sha="abc123", was_dirty=True)
        assert RepoSnapshot.from_dict(snap.to_dict()) == snap

    def test_from_dict_missing_was_dirty_defaults_false(self):
        snap = RepoSnapshot.from_dict({"repo": "/x", "sha": "abc"})
        assert snap.was_dirty is False


# ---------------------------------------------------------------------------
# Step new fields persist and load
# ---------------------------------------------------------------------------

class TestStepNewFields:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        plan_mod.configure(str(tmp_path), ".agent")

    def test_snapshot_refs_persist(self):
        plan = plan_mod.create_plan("goal", steps=["do something"])
        step = plan.steps[0]
        refs = [{"repo": "/r", "sha": "abc", "was_dirty": True}]
        plan_mod.update_step(plan, step.id, snapshot_refs=refs, retry_count=1)
        loaded = plan_mod.load_plan(plan.id)
        assert loaded.steps[0].snapshot_refs == refs
        assert loaded.steps[0].retry_count == 1

    def test_old_json_loads_with_defaults(self, tmp_path):
        """JSON without new fields loads fine with defaults."""
        import json
        plan = plan_mod.create_plan("goal", steps=["s"])
        path = tmp_path / ".agent" / "plans" / f"{plan.id}.json"
        data = json.loads(path.read_text())
        # Remove new fields to simulate old JSON
        for step_data in data["steps"]:
            step_data.pop("snapshot_refs", None)
            step_data.pop("retry_count", None)
            step_data.pop("max_retries", None)
            step_data.pop("skills", None)
        path.write_text(json.dumps(data))
        loaded = plan_mod.load_plan(plan.id)
        assert loaded.steps[0].snapshot_refs == []
        assert loaded.steps[0].retry_count == 0
        assert loaded.steps[0].max_retries == 3
        assert loaded.steps[0].skills == []

    def test_skills_persist_and_load(self):
        plan = plan_mod.create_plan("goal", steps=[{
            "id": "s1", "description": "do thing", "skills": ["python", "testing"]
        }])
        loaded = plan_mod.load_plan(plan.id)
        assert loaded.steps[0].skills == ["python", "testing"]
