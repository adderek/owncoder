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

import sqlite3

DEFAULT_BUSY_MS = 30_000


def apply_concurrency_pragmas(conn: sqlite3.Connection, busy_ms: int = DEFAULT_BUSY_MS) -> None:
    """Set WAL + busy_timeout + synchronous=NORMAL on *conn*."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_ms)}")
    conn.execute("PRAGMA synchronous=NORMAL")
