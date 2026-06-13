"""Unit tests for core.checkpoint — session-wide edit checkpoint/rollback."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.tools.files import setup as files_setup, write_file, _undo_stack
import agent.core.checkpoint as ckpt


@pytest.fixture()
def workdir(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)        # also resets the checkpoint journal
    _undo_stack.clear()
    ckpt.reset()
    return tmp_path, cfg


def _write(rel, content):
    return write_file(rel, content)


class TestJournalAndRollback:
    def test_rollback_restores_modified_file(self, workdir):
        tmp, _ = workdir
        (tmp / "a.txt").write_text("v0")
        _write("a.txt", "v1")  # journaled: before=v0
        cp = ckpt.create_checkpoint("before v2")
        _write("a.txt", "v2")
        assert (tmp / "a.txt").read_text() == "v2"
        res = ckpt.rollback_to(cp.id)
        assert res["ok"]
        assert (tmp / "a.txt").read_text() == "v1"
        assert "a.txt" in res["restored"]

    def test_rollback_deletes_created_file(self, workdir):
        tmp, _ = workdir
        cp = ckpt.create_checkpoint("clean")
        _write("new.txt", "hello")  # created after checkpoint, before=None
        assert (tmp / "new.txt").exists()
        res = ckpt.rollback_to(cp.id)
        assert not (tmp / "new.txt").exists()
        assert "new.txt" in res["deleted"]

    def test_multi_edit_reverts_to_checkpoint_state(self, workdir):
        tmp, _ = workdir
        (tmp / "f.txt").write_text("0")
        _write("f.txt", "1")
        cp = ckpt.create_checkpoint("at 1")
        _write("f.txt", "2")
        _write("f.txt", "3")
        ckpt.rollback_to(cp.id)
        assert (tmp / "f.txt").read_text() == "1"

    def test_checkpoint_before_changes_is_noop(self, workdir):
        tmp, _ = workdir
        (tmp / "f.txt").write_text("orig")
        cp = ckpt.create_checkpoint("start")
        res = ckpt.rollback_to(cp.id)
        assert res["reverted_edits"] == 0
        assert (tmp / "f.txt").read_text() == "orig"

    def test_unknown_checkpoint(self, workdir):
        assert "error" in ckpt.rollback_to("nope")

    def test_rollback_drops_later_checkpoints(self, workdir):
        tmp, _ = workdir
        cp1 = ckpt.create_checkpoint("one")
        _write("a.txt", "a")
        cp2 = ckpt.create_checkpoint("two")
        ckpt.rollback_to(cp1.id)
        ids = [c.id for c in ckpt.list_checkpoints()]
        assert cp1.id in ids and cp2.id not in ids

    def test_journal_trimmed_after_rollback(self, workdir):
        tmp, _ = workdir
        cp = ckpt.create_checkpoint("c")
        _write("x.txt", "1")
        ckpt.rollback_to(cp.id)
        # editing again then rolling back to same checkpoint id fails (it was kept)
        _write("x.txt", "2")
        res = ckpt.rollback_to(cp.id)
        assert res["ok"]
        assert not (tmp / "x.txt").exists()


class TestCommandHandler:
    def test_list_empty_then_new_then_rollback(self, workdir):
        tmp, _ = workdir
        assert "No checkpoints" in ckpt.run_checkpoint_command("list")
        out = ckpt.run_checkpoint_command("new my-point")
        assert "Created checkpoint" in out
        assert "my-point" in ckpt.run_checkpoint_command("list")
        _write("z.txt", "data")
        cid = ckpt.list_checkpoints()[0].id
        res = ckpt.run_checkpoint_command(f"rollback {cid}")
        assert "Rolled back" in res
        assert not (tmp / "z.txt").exists()

    def test_rollback_requires_id(self, workdir):
        assert "Usage" in ckpt.run_checkpoint_command("rollback")

    def test_unknown_sub(self, workdir):
        assert "Unknown subcommand" in ckpt.run_checkpoint_command("frob")
