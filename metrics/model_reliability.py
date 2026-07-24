"""Global, cross-project per-model call outcome tracking.

Records success/failure/rate_limited per model entry into a shared SQLite db
(WAL mode — safe under multiple concurrent agent processes, unlike
atomic-replace JSON which can lose concurrent writes). Global rather than
per-project: model reliability is a property of the endpoint, not the
project calling it.
"""
from __future__ import annotations

import random
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path.home() / ".config" / "agent" / "metrics" / "model_reliability.db"
_PRUNE_AFTER_DAYS = 30
_PRUNE_SAMPLE_RATE = 200  # ~1-in-N writes triggers a prune sweep

_local = threading.local()
_schema_lock = threading.Lock()
_schema_ready = False

OUTCOMES = ("success", "failure", "rate_limited")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # DDL races if run per-thread on separate connections (concurrent
    # "CREATE TABLE IF NOT EXISTS" can still hit SQLITE_BUSY) — guard with a
    # process-wide lock so it only actually runs once, matching how
    # MemoryStore.__init__ sets up its schema a single time before any
    # per-thread connections are opened.
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                entry_name TEXT NOT NULL,
                ts         INTEGER NOT NULL,
                outcome    TEXT NOT NULL,
                role       TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_calls_entry_ts ON calls(entry_name, ts);
        """)
        _schema_ready = True


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn"):
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        from agent.core.sqlite_util import open_threadlocal_conn
        conn = open_threadlocal_conn(str(_DB_PATH))
        _ensure_schema(conn)
        _local.conn = conn
    return _local.conn


def record_outcome(entry_name: str, outcome: str, role: str = "") -> None:
    """Record one call outcome for *entry_name*. Best-effort — never raises."""
    if not entry_name or outcome not in OUTCOMES:
        return
    try:
        conn = _conn()
        now = int(time.time())
        conn.execute(
            "INSERT INTO calls (entry_name, ts, outcome, role) VALUES (?, ?, ?, ?)",
            (entry_name, now, outcome, role),
        )
        conn.commit()
        if random.randint(1, _PRUNE_SAMPLE_RATE) == 1:
            cutoff = now - _PRUNE_AFTER_DAYS * 86400
            conn.execute("DELETE FROM calls WHERE ts < ?", (cutoff,))
            conn.commit()
    except Exception:
        pass


def reliability_summary(entry_name: str, window_hours: int = 24) -> dict:
    """{success, failure, rate_limited, total, success_rate} over the window."""
    result = {"success": 0, "failure": 0, "rate_limited": 0, "total": 0, "success_rate": None}
    try:
        conn = _conn()
        cutoff = int(time.time()) - window_hours * 3600
        rows = conn.execute(
            "SELECT outcome, COUNT(*) FROM calls WHERE entry_name = ? AND ts >= ? GROUP BY outcome",
            (entry_name, cutoff),
        ).fetchall()
        for outcome, count in rows:
            if outcome in result:
                result[outcome] = count
        result["total"] = result["success"] + result["failure"] + result["rate_limited"]
        # rate-limited calls reflect throttling, not model unreliability —
        # excluded from the success-rate denominator so a busy free model
        # doesn't read as "fragile".
        judged = result["success"] + result["failure"]
        if judged:
            result["success_rate"] = round(result["success"] / judged, 3)
    except Exception:
        pass
    return result
