"""SQLite concurrency pragmas: WAL + busy_timeout + synchronous=NORMAL."""
from __future__ import annotations

import sqlite3
import threading

from agent.core.sqlite_util import apply_concurrency_pragmas


def _open(path):
    c = sqlite3.connect(str(path), timeout=30)
    apply_concurrency_pragmas(c)
    return c


def test_pragmas_applied(tmp_path):
    c = _open(tmp_path / "x.db")
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
    assert c.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    c.close()


def test_custom_busy_timeout(tmp_path):
    c = _open(tmp_path / "y.db")
    apply_concurrency_pragmas(c, busy_ms=1234)
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
    c.close()


def test_open_succeeds_while_another_connection_holds_a_write_lock(tmp_path):
    """Opening while a writer holds the write lock must not raise.

    The journal_mode switch takes a short exclusive lock and does not wait on
    the busy handler, so a naive `PRAGMA journal_mode=WAL` raises "database is
    locked" here. Callers open lazily per thread and several swallow connection
    errors, so that surfaced as silently lost writes, not as a failure.
    """
    # The database must NOT be in WAL yet: it is the *transition* that needs the
    # exclusive lock. Re-setting the mode on a database already in WAL is a
    # no-op that never blocks, which is why this only bit on first contact.
    db = tmp_path / "contended.db"
    w = sqlite3.connect(str(db), timeout=30, check_same_thread=False)
    w.execute("CREATE TABLE t(x INTEGER)")
    w.commit()
    assert w.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal"
    w.execute("BEGIN IMMEDIATE")          # hold the write lock
    w.execute("INSERT INTO t VALUES (1)")

    def _release():
        w.commit()

    # Let the writer go a moment later, from another thread, so the retry loop
    # has something to succeed on rather than exhausting its attempts.
    timer = threading.Timer(0.05, _release)
    timer.start()
    try:
        second = _open(db)                # must not raise
        assert second.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        second.close()
    finally:
        timer.join()
        w.close()


def test_wal_switch_failure_degrades_instead_of_raising(tmp_path, monkeypatch, caplog):
    """If WAL can never be taken, the open still succeeds with a warning —
    every store here is correct (just less concurrent) on a rollback journal."""
    import agent.core.sqlite_util as su

    monkeypatch.setattr(su, "_set_wal", lambda conn: False)
    with caplog.at_level("WARNING"):
        c = sqlite3.connect(str(tmp_path / "nowal.db"), timeout=30)
        su.apply_concurrency_pragmas(c)
    assert "could not switch database to WAL" in caplog.text
    assert c.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000
    c.close()


def test_wal_reader_not_blocked_by_writer(tmp_path):
    """Under WAL a reader sees the last committed snapshot while a writer holds
    an open write transaction — it does not block."""
    db = tmp_path / "z.db"
    w = _open(db)
    w.execute("CREATE TABLE t(x INTEGER)")
    w.execute("INSERT INTO t VALUES (1)")
    w.commit()

    r = _open(db)
    # writer opens an uncommitted write transaction
    w.execute("BEGIN")
    w.execute("INSERT INTO t VALUES (2)")
    # reader still reads the committed snapshot without blocking
    assert r.execute("SELECT count(*) FROM t").fetchone()[0] == 1
    w.commit()
    assert r.execute("SELECT count(*) FROM t").fetchone()[0] == 2
    w.close()
    r.close()
