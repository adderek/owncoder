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

logger = logging.getLogger(__name__)

DEFAULT_BUSY_MS = 30_000


def apply_concurrency_pragmas(conn: sqlite3.Connection, busy_ms: int = DEFAULT_BUSY_MS) -> None:
    """Set WAL + busy_timeout + synchronous=NORMAL on *conn*."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_ms)}")
    conn.execute("PRAGMA synchronous=NORMAL")


def open_threadlocal_conn(
    db_path: str,
    *,
    load_vec: bool = False,
    foreign_keys: bool = False,
    busy_ms: int = DEFAULT_BUSY_MS,
) -> sqlite3.Connection:
    """Open a per-thread SQLite connection with the shared tuning.

    Centralizes the connect + row_factory + concurrency-pragma boilerplate the
    stores all repeat. ``load_vec`` registers the sqlite-vec extension; if it is
    unavailable the connection is still returned and vector search degrades to
    FTS-only (logged) rather than crashing — a uniform policy across all stores.
    Store-specific DDL is the caller's job (run it after caching the conn).
    """
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
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
