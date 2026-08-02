"""Shared SQLite connection tuning for concurrent (multi-process) access.

Several agents may open the same database (code index, memory, archive) at
once. Applied per connection:

- journal_mode=WAL: readers never block the single writer and vice-versa
  (persisted on the db file; idempotent to re-set).
- busy_timeout: on write contention, block and retry internally for up to this
  long instead of immediately raising "database is locked". Matches the
  connect(timeout=) but is explicit and independent of the driver default.
- synchronous=NORMAL: the recommended durability level under WAL — safe across
  application crashes; only an OS crash / power loss can lose the last commits,
  which is acceptable for a regenerable index and tolerable for memory.
"""
from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)

DEFAULT_BUSY_MS = 30_000


JOURNAL_MODE_ATTEMPTS = 5
_JOURNAL_RETRY_S = 0.02


def apply_concurrency_pragmas(conn: sqlite3.Connection, busy_ms: int = DEFAULT_BUSY_MS) -> None:
    """Set busy_timeout + WAL + synchronous=NORMAL on *conn*.

    Order and retries here are load-bearing:

    - busy_timeout goes first so it is already in effect for everything after it.
    - The journal_mode switch takes a short exclusive lock and, unlike ordinary
      statements, does **not** wait on the busy handler: on a database another
      connection is mid-write on, it raises "database is locked" immediately.
      Since every caller opens lazily per thread, that lands as a *lost write*
      inside whatever store swallows connection errors, not as a visible
      failure — so it is retried, and skipped when the file already says WAL
      (the mode is persisted on the database, so re-setting it buys nothing).
    """
    conn.execute(f"PRAGMA busy_timeout={int(busy_ms)}")
    if not _set_wal(conn):
        # Every store here tolerates rollback-journal mode (correct, just less
        # concurrent), so degrade with a warning rather than failing the open.
        logger.warning("could not switch database to WAL (locked by another "
                       "connection); continuing in its current journal mode")
    conn.execute("PRAGMA synchronous=NORMAL")


def _set_wal(conn: sqlite3.Connection) -> bool:
    """Put *conn*'s database in WAL mode. True if it is in WAL when we return."""
    for attempt in range(JOURNAL_MODE_ATTEMPTS):
        try:
            current = conn.execute("PRAGMA journal_mode").fetchone()
            if current and str(current[0]).lower() == "wal":
                return True
            row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            if row and str(row[0]).lower() == "wal":
                return True
        except sqlite3.OperationalError:
            pass  # locked by a concurrent writer — back off and retry
        if attempt < JOURNAL_MODE_ATTEMPTS - 1:
            time.sleep(_JOURNAL_RETRY_S * (attempt + 1))
    return False


def open_threadlocal_conn(
    db_path: str,
    *,
    load_vec: bool = False,
    foreign_keys: bool = False,
    busy_ms: int = DEFAULT_BUSY_MS,
    uri: bool = False,
) -> sqlite3.Connection:
    """Open a per-thread SQLite connection with the shared tuning.

    Centralizes the connect + row_factory + concurrency-pragma boilerplate the
    stores all repeat. ``load_vec`` registers the sqlite-vec extension; if it is
    unavailable the connection is still returned and vector search degrades to
    FTS-only (logged) rather than crashing — a uniform policy across all stores.
    Store-specific DDL is the caller's job (run it after caching the conn).
    ``uri`` passes *db_path* through as a SQLite URI, which is how an in-memory
    database gets a shared cache — without it every thread would silently open
    its own empty one (see agent/security/vault.py).
    """
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False, uri=uri)
    conn.row_factory = sqlite3.Row
    apply_concurrency_pragmas(conn, busy_ms)
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    if load_vec:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except Exception as e:
            logger.warning("sqlite-vec load failed (%s); vector search degraded to FTS-only", e)
        finally:
            conn.enable_load_extension(False)
    return conn
