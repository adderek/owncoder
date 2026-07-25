"""Checkpoint persistence — the point is surviving a process restart.

A restart is simulated by calling `ckpt.setup(config)` again after clearing the
in-memory state, which is exactly what the tools layer does on session start.
"""
from __future__ import annotations

import json

import pytest

from agent.config import Config
from agent.tools.files import setup as files_setup, write_file, _undo_stack
import agent.core.checkpoint as ckpt
import agent.core.checkpoint_store as store


@pytest.fixture()
def workdir(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.checkpoints.persist = True
    files_setup(cfg)          # calls ckpt.setup(cfg)
    _undo_stack.clear()
    yield tmp_path, cfg
    ckpt.reset()


def _restart(cfg):
    """Drop in-memory state and reload from disk, as a new process would."""
    ckpt.reset()
    ckpt.setup(cfg)


class TestSurvivesRestart:
    def test_checkpoint_and_journal_are_restored(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        cp = ckpt.create_checkpoint("before refactor")
        write_file("a.txt", "v2")

        _restart(cfg)

        ids = [c.id for c in ckpt.list_checkpoints()]
        assert cp.id in ids
        assert ckpt.list_checkpoints()[0].label == "before refactor"

    def test_rollback_works_after_restart(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        cp = ckpt.create_checkpoint("cp")
        write_file("a.txt", "v2")
        write_file("new.txt", "created")

        _restart(cfg)
        res = ckpt.rollback_to(cp.id)

        assert res["ok"] is True
        assert (tmp / "a.txt").read_text() == "v1"
        assert not (tmp / "new.txt").exists()

    def test_rollback_after_restart_is_itself_persisted(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        cp = ckpt.create_checkpoint("cp")
        write_file("a.txt", "v2")
        ckpt.rollback_to(cp.id)

        _restart(cfg)
        # The rolled-back edit is gone from the journal, so rolling back again
        # is a no-op rather than a second revert of stale content.
        res = ckpt.rollback_to(cp.id)
        assert res["reverted_edits"] == 0
        assert (tmp / "a.txt").read_text() == "v1"

    def test_generated_ids_do_not_collide_with_restored_ones(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        first = ckpt.create_checkpoint("one")

        _restart(cfg)
        second = ckpt.create_checkpoint("two")

        assert second.id != first.id
        assert len(ckpt.list_checkpoints()) == 2


class TestOptOut:
    def test_persist_off_keeps_memory_only(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        cfg.checkpoints.persist = False
        files_setup(cfg)
        _undo_stack.clear()

        (tmp_path / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        ckpt.create_checkpoint("cp")

        assert not store.root(cfg).exists()
        _restart(cfg)
        assert ckpt.list_checkpoints() == []


class TestStore:
    def test_identical_pre_images_share_one_blob(self, workdir):
        tmp, cfg = workdir
        directory = store.root(cfg)
        first = store.write_blob(directory, "same content")
        second = store.write_blob(directory, "same content")
        assert first == second
        blobs = [p for p in (directory / "blobs").rglob("*") if p.is_file()]
        assert len(blobs) == 1

    def test_malformed_journal_line_is_skipped(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        journal_path = store.root(cfg) / "journal.jsonl"
        with journal_path.open("a", encoding="utf-8") as f:
            f.write("this is not json\n")
        journal, _ = store.load(cfg)
        assert len(journal) == 1

    def test_entry_with_missing_blob_is_dropped(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        for blob in (store.root(cfg) / "blobs").rglob("*"):
            if blob.is_file():
                blob.unlink()
        journal, _ = store.load(cfg)
        assert journal == [], "an unrestorable entry must not pretend it can roll back"

    def test_unreadable_checkpoint_file_is_ignored(self, workdir):
        tmp, cfg = workdir
        directory = store.root(cfg)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "checkpoints.json").write_text("{ broken", encoding="utf-8")
        _, checkpoints = store.load(cfg)
        assert checkpoints == []

    def test_prune_drops_old_entries_and_orphan_blobs(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")

        # Age the single journal entry past the cap.
        journal_path = store.root(cfg) / "journal.jsonl"
        record = json.loads(journal_path.read_text().splitlines()[0])
        record["ts"] = 0
        journal_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        cfg.checkpoints.max_age_days = 1
        res = store.prune(cfg)
        assert res["dropped"] == 1
        assert res["blobs_deleted"] == 1
        assert store.load(cfg)[0] == []

    def test_prune_keeps_fresh_entries(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        cfg.checkpoints.max_age_days = 30
        assert store.prune(cfg)["dropped"] == 0
        assert len(store.load(cfg)[0]) == 1

    def test_max_age_zero_keeps_everything(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        journal_path = store.root(cfg) / "journal.jsonl"
        record = json.loads(journal_path.read_text().splitlines()[0])
        record["ts"] = 0
        journal_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        cfg.checkpoints.max_age_days = 0
        assert store.prune(cfg)["dropped"] == 0


class TestWriteProtection:
    def test_store_dir_is_write_denied_to_agent_tools(self):
        from agent.security.fs import _DEFAULT_WRITE_DENY_GLOBS
        assert ".agent/checkpoints/**" in _DEFAULT_WRITE_DENY_GLOBS

    def test_agent_cannot_edit_the_journal(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        out = write_file(".agent/checkpoints/journal.jsonl", "[]")
        assert "error" in out


class TestCommand:
    def test_prune_subcommand_reports(self, workdir):
        tmp, cfg = workdir
        (tmp / "a.txt").write_text("v0")
        write_file("a.txt", "v1")
        assert "Pruned" in ckpt.run_checkpoint_command("prune")

    def test_prune_reports_when_persistence_is_off(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        cfg.checkpoints.persist = False
        files_setup(cfg)
        assert "persistence is off" in ckpt.run_checkpoint_command("prune")
