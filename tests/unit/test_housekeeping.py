"""agent/housekeeping.py — level-1 state-directory housekeeping."""
from __future__ import annotations

import fcntl
import gzip
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

import pytest

from agent import housekeeping as hk


def _cfg(**kw):
    house = SimpleNamespace(enabled=True, interval_hours=24.0, keep_index_backups=1)
    for k, v in kw.items():
        setattr(house, k, v)
    return SimpleNamespace(housekeeping=house, logs=SimpleNamespace(backup_count=5))


def _names(d):
    return sorted(p.name for p in d.iterdir() if p.name != hk.LOCK_NAME)


def test_dry_run_changes_nothing(tmp_path):
    (tmp_path / "agent.log").write_text("live")
    (tmp_path / "agent.log.1").write_text("old" * 1000)
    (tmp_path / "index.db").write_text("x")
    (tmp_path / "index.db.bak-20260101-000000").write_text("a")
    (tmp_path / "index.db.bak-20260201-000000").write_text("b")
    before = _names(tmp_path)

    report = hk.run(tmp_path, _cfg(), do_apply=False)

    assert not report.applied
    assert {a.kind for a in report.actions} == {"compress", "delete"}
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_legacy_logs_are_gzipped_in_order(tmp_path):
    (tmp_path / "agent.log").write_text("live")
    (tmp_path / "agent.log.1").write_text("newer\n" * 500)
    (tmp_path / "agent.log.3").write_text("older\n" * 500)
    os.utime(tmp_path / "agent.log.3", (1_000_000, 1_000_000))

    report = hk.run(tmp_path, _cfg(), do_apply=True)

    assert not report.errors
    assert report.freed > 0
    assert _names(tmp_path) == ["agent.log", "agent.log.1.gz", "agent.log.2.gz"]
    assert gzip.open(tmp_path / "agent.log.1.gz", "rt").read() == "newer\n" * 500
    assert gzip.open(tmp_path / "agent.log.2.gz", "rt").read() == "older\n" * 500
    assert (tmp_path / "agent.log.2.gz").stat().st_mtime == 1_000_000
    assert (tmp_path / "agent.log").read_text() == "live"


def test_legacy_logs_queue_behind_existing_gz(tmp_path):
    with gzip.open(tmp_path / "agent.log.1.gz", "wt") as fh:
        fh.write("rotated by the new handler")
    (tmp_path / "agent.log.1").write_text("legacy")

    hk.run(tmp_path, _cfg(), do_apply=True)

    assert gzip.open(tmp_path / "agent.log.1.gz", "rt").read() == "rotated by the new handler"
    assert gzip.open(tmp_path / "agent.log.2.gz", "rt").read() == "legacy"
    assert not (tmp_path / "agent.log.1").exists()


def test_logs_beyond_backup_count_are_dropped(tmp_path):
    cfg = _cfg()
    cfg.logs.backup_count = 2
    for i in (1, 2, 3):
        (tmp_path / f"agent.log.{i}").write_text(str(i))

    hk.run(tmp_path, cfg, do_apply=True)

    assert _names(tmp_path) == ["agent.log.1.gz", "agent.log.2.gz"]


def test_index_backups_trimmed_to_newest(tmp_path):
    (tmp_path / "index.db").write_text("live")
    for stamp in ("20260921-155309", "20261108-211052", "20260301-000000"):
        (tmp_path / f"index.db.bak-{stamp}").write_text(stamp)
        (tmp_path / f"index.db.bak-{stamp}-shm").write_text("s")

    report = hk.run(tmp_path, _cfg(), do_apply=True)

    assert not report.errors
    assert _names(tmp_path) == [
        "index.db", "index.db.bak-20261108-211052", "index.db.bak-20261108-211052-shm",
    ]


def test_index_backups_kept_without_live_index(tmp_path):
    (tmp_path / "index.db.bak-20260101-000000").write_text("a")
    (tmp_path / "index.db.bak-20260201-000000").write_text("b")
    assert hk.plan(tmp_path) == []
    (tmp_path / "index.db").write_text("")          # empty = unusable
    assert hk.plan(tmp_path) == []


def test_keep_all_index_backups(tmp_path):
    (tmp_path / "index.db").write_text("live")
    (tmp_path / "index.db.bak-20260101-000000").write_text("a")
    (tmp_path / "index.db.bak-20260201-000000").write_text("b")
    assert hk.run(tmp_path, _cfg(keep_index_backups=-1), do_apply=False).actions == []


def test_sessions_and_audit_untouched(tmp_path):
    sess = tmp_path / "2026" / "05" / "03" / "20260503T203418.510Z_c11c"
    sess.mkdir(parents=True)
    (sess / "session.json").write_text("{}")
    (sess / "running").write_text("")
    (tmp_path / "audit.jsonl").write_text("{}\n")
    (tmp_path / "audit.jsonl.1").write_text("{}\n")
    assert hk.plan(tmp_path) == []


def test_run_returns_none_when_locked(tmp_path):
    (tmp_path / "agent.log.1").write_text("x")
    with open(tmp_path / hk.LOCK_NAME, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert hk.run(tmp_path, _cfg(), do_apply=True) is None
    assert (tmp_path / "agent.log.1").exists()


def test_background_runs_once_per_interval(tmp_path):
    (tmp_path / "agent.log.1").write_text("x" * 100)
    thread = hk.maybe_start_background(tmp_path, _cfg())
    assert thread is not None
    thread.join(10)
    assert (tmp_path / "agent.log.1.gz").exists()

    (tmp_path / "agent.log.1").write_text("y" * 100)
    assert hk.maybe_start_background(tmp_path, _cfg()) is None

    old = time.time() - 25 * 3600
    os.utime(tmp_path / hk.LOCK_NAME, (old, old))
    thread = hk.maybe_start_background(tmp_path, _cfg())
    assert thread is not None
    thread.join(10)
    assert (tmp_path / "agent.log.2.gz").exists()


def test_background_disabled(tmp_path):
    (tmp_path / "agent.log.1").write_text("x")
    assert hk.maybe_start_background(tmp_path, _cfg(enabled=False)) is None


def test_handler_rotates_straight_to_gz(tmp_path):
    handler = RotatingFileHandler(tmp_path / "agent.log", maxBytes=200, backupCount=2,
                                  encoding="utf-8")
    handler.namer = hk.gz_namer
    handler.rotator = hk.gz_rotator
    log = logging.getLogger("test_housekeeping.rotation")
    log.propagate = False
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    try:
        for i in range(40):
            log.info("line %03d %s", i, "x" * 40)
    finally:
        log.removeHandler(handler)
        handler.close()

    assert _names(tmp_path) == ["agent.log", "agent.log.1.gz", "agent.log.2.gz"]
    assert "line" in gzip.open(tmp_path / "agent.log.1.gz", "rt").read()
    assert hk.plan(tmp_path, backup_count=2) == []
