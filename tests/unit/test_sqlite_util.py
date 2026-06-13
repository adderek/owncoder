"""SQLite concurrency pragmas: WAL + busy_timeout + synchronous=NORMAL."""
from __future__ import annotations

import sqlite3

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
