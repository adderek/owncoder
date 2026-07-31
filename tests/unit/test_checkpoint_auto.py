"""Automatic checkpoints and turn-failure handling — core/checkpoint.py (P1).

Manual checkpoints only help when the user remembered to make one before the
risky change. These cover the every-N-edits timer and what happens to the edits
a failed turn leaves behind.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.config import Config
from agent.core.agent import Agent
from agent.tools.files import setup as files_setup, write_file, _undo_stack
import agent.core.checkpoint as ckpt


@pytest.fixture()
def workdir(tmp_path):
    """Project root with the checkpoint config attached but persistence off."""
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.checkpoints.persist = False
    cfg.checkpoints.auto_interval = 3
    files_setup(cfg)
    _undo_stack.clear()
    ckpt.setup(cfg)          # attaches config; no store because persist is off
    return tmp_path, cfg


def _write(rel, content):
    return write_file(rel, content)


def _autos():
    return [c for c in ckpt.list_checkpoints() if c.auto]


# ── the every-N-edits timer ───────────────────────────────────────────────────

class TestAutoInterval:
    def test_checkpoint_appears_every_n_edits(self, workdir):
        for i in range(3):
            _write(f"f{i}.txt", "x")
        assert len(_autos()) == 1

    def test_no_checkpoint_before_the_interval(self, workdir):
        _write("a.txt", "x")
        _write("b.txt", "x")
        assert _autos() == []

    def test_counter_restarts_after_each_auto(self, workdir):
        for i in range(6):
            _write(f"f{i}.txt", "x")
        assert len(_autos()) == 2

    def test_interval_zero_disables(self, workdir):
        _tmp, cfg = workdir
        cfg.checkpoints.auto_interval = 0
        for i in range(10):
            _write(f"f{i}.txt", "x")
        assert _autos() == []

    def test_no_config_means_no_auto_checkpoints(self, tmp_path):
        """A bare reset() (what most tests do) must not start auto-checkpointing."""
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        files_setup(cfg)
        ckpt.reset()
        for i in range(10):
            write_file(f"f{i}.txt", "x")
        assert ckpt.list_checkpoints() == []

    def test_auto_checkpoint_is_tagged_in_the_listing(self, workdir):
        for i in range(3):
            _write(f"f{i}.txt", "x")
        assert "[auto]" in ckpt.run_checkpoint_command("list")

    def test_manual_checkpoints_are_not_tagged_auto(self, workdir):
        cp = ckpt.create_checkpoint("by hand")
        assert not cp.auto
        assert _autos() == []


# ── locating the rollback point ───────────────────────────────────────────────

class TestLatestAuto:
    def test_none_when_no_auto_checkpoint_exists(self, workdir):
        ckpt.create_checkpoint("manual")
        assert ckpt.latest_auto_checkpoint() is None

    def test_none_when_nothing_changed_since(self, workdir):
        for i in range(3):
            _write(f"f{i}.txt", "x")
        assert ckpt.latest_auto_checkpoint() is None, "no edits after it to undo"

    def test_returns_newest_auto_with_pending_edits(self, workdir):
        for i in range(6):
            _write(f"f{i}.txt", "x")
        _write("after.txt", "x")
        cp = ckpt.latest_auto_checkpoint()
        assert cp is not None and cp.id == _autos()[-1].id

    def test_edits_since_counts_journal_entries(self, workdir):
        for i in range(3):
            _write(f"f{i}.txt", "x")
        cp = _autos()[0]
        _write("x.txt", "1")
        _write("y.txt", "1")
        assert ckpt.edits_since(cp.id) == 2

    def test_edits_since_unknown_checkpoint_is_zero(self, workdir):
        assert ckpt.edits_since("nope") == 0


class TestRollbackLastAuto:
    def test_reverts_edits_made_after_the_checkpoint(self, workdir):
        tmp, _ = workdir
        for i in range(3):
            _write(f"f{i}.txt", "keep")
        _write("f0.txt", "clobbered")
        _write("new.txt", "created after")

        res = ckpt.rollback_last_auto()
        assert res["ok"]
        assert (tmp / "f0.txt").read_text() == "keep"
        assert not (tmp / "new.txt").exists()

    def test_returns_none_with_nothing_to_undo(self, workdir):
        assert ckpt.rollback_last_auto() is None

    def test_rollback_resets_the_edit_counter(self, workdir):
        for i in range(3):
            _write(f"f{i}.txt", "x")
        _write("pending.txt", "x")       # 1 edit toward the next auto
        ckpt.rollback_last_auto()
        _write("a.txt", "x")
        _write("b.txt", "x")
        assert len(_autos()) == 1, "the reverted edit must not count toward the next auto"


class TestPersistence:
    def test_auto_flag_survives_a_reload(self, tmp_path):
        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        cfg.checkpoints.persist = True
        cfg.checkpoints.auto_interval = 2
        files_setup(cfg)
        _undo_stack.clear()
        ckpt.setup(cfg)

        write_file("a.txt", "x")
        write_file("b.txt", "x")
        assert len(_autos()) == 1

        ckpt.setup(cfg)                  # simulate a restart
        restored = _autos()
        assert len(restored) == 1 and restored[0].auto


# ── what a failed turn leaves behind ──────────────────────────────────────────

class _StubAgent:
    """Just enough Agent for the note helper, which only reads self.config."""

    def __init__(self, config):
        self.config = config

    note_for = Agent._checkpoint_note_for_failed_turn


class TestFailedTurnNote:
    def _agent_with_pending_edits(self, cfg):
        for i in range(3):
            _write(f"f{i}.txt", "keep")
        _write("f0.txt", "half-finished")
        return _StubAgent(cfg)

    def test_cancellation_is_never_treated_as_failure(self, workdir):
        _tmp, cfg = workdir
        a = self._agent_with_pending_edits(cfg)
        assert a.note_for(KeyboardInterrupt()) == ""
        assert a.note_for(asyncio.CancelledError()) == ""

    def test_ctrl_c_does_not_revert_files(self, workdir):
        tmp, cfg = workdir
        cfg.checkpoints.auto_rollback_on_error = True
        a = self._agent_with_pending_edits(cfg)
        a.note_for(KeyboardInterrupt())
        assert (tmp / "f0.txt").read_text() == "half-finished"

    def test_no_note_without_pending_edits(self, workdir):
        _tmp, cfg = workdir
        for i in range(3):
            _write(f"f{i}.txt", "x")
        assert _StubAgent(cfg).note_for(RuntimeError("boom")) == ""

    def test_default_names_the_checkpoint_and_keeps_files(self, workdir):
        tmp, cfg = workdir
        a = self._agent_with_pending_edits(cfg)
        note = a.note_for(RuntimeError("boom"))
        cp_id = _autos()[-1].id
        assert f"/checkpoint rollback {cp_id}" in note
        assert (tmp / "f0.txt").read_text() == "half-finished", "default must not revert"

    def test_opt_in_rolls_back_and_reports(self, workdir):
        tmp, cfg = workdir
        cfg.checkpoints.auto_rollback_on_error = True
        a = self._agent_with_pending_edits(cfg)
        note = a.note_for(RuntimeError("boom"))
        assert "auto-rolled back" in note
        assert (tmp / "f0.txt").read_text() == "keep"

    def test_note_is_empty_when_no_auto_checkpoint_exists(self, workdir):
        _tmp, cfg = workdir
        _write("only.txt", "x")          # below the interval; no auto checkpoint
        assert _StubAgent(cfg).note_for(RuntimeError("boom")) == ""
