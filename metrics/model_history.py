"""Long-term per-model throughput history — one row per LLM call.

``model_stats.json`` keeps only the current EWMA and lifetime totals, which
answers "how fast is this model *now*" but not "was it slower last week" or
"does this endpoint fluctuate". This module appends the raw samples to a
shared SQLite db (WAL, same pattern as ``model_reliability``: global rather
than per-project, because throughput is a property of the endpoint) so trends
and graphs can be computed after the fact.

Rows are pruned after ``_PRUNE_AFTER_DAYS``. At a few thousand calls a day the
table stays in the low megabytes.
"""
from __future__ import annotations

import random
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path.home() / ".config" / "agent" / "metrics" / "model_throughput.db"
_PRUNE_AFTER_DAYS = 180
_PRUNE_SAMPLE_RATE = 200  # ~1-in-N writes triggers a prune sweep

_local = threading.local()
_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema(conn: sqlite3.Connection) -> None:
    # Process-wide guard: concurrent "CREATE TABLE IF NOT EXISTS" on separate
    # per-thread connections can still hit SQLITE_BUSY.
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS samples (
                entry_name TEXT NOT NULL,
                ts         INTEGER NOT NULL,
                in_tokens  INTEGER NOT NULL DEFAULT 0,
                out_tokens INTEGER NOT NULL DEFAULT 0,
                gen_sec    REAL    NOT NULL DEFAULT 0,
                ttft       REAL    NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_samples_entry_ts ON samples(entry_name, ts);
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


def record_sample(entry_name: str, out_tokens: int = 0, gen_sec: float = 0.0,
                  in_tokens: int = 0, ttft: float = 0.0) -> None:
    """Append one call's raw throughput measurement. Never raises.

    *in_tokens* is the uncached prompt (what the endpoint actually had to
    prefill) — see ``model_stats`` for why cached tokens are excluded.
    """
    if not entry_name:
        return
    if not ((out_tokens and gen_sec > 0) or (in_tokens and ttft > 0)):
        return
    try:
        conn = _conn()
        now = int(time.time())
        conn.execute(
            "INSERT INTO samples (entry_name, ts, in_tokens, out_tokens, gen_sec, ttft) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (entry_name, now, int(in_tokens or 0), int(out_tokens or 0),
             float(gen_sec or 0.0), float(ttft or 0.0)),
        )
        conn.commit()
        if random.randint(1, _PRUNE_SAMPLE_RATE) == 1:
            conn.execute("DELETE FROM samples WHERE ts < ?",
                         (now - _PRUNE_AFTER_DAYS * 86400,))
            conn.commit()
    except Exception:
        pass


def series(entry_name: str, hours: int = 168, buckets: int = 24) -> list[dict]:
    """Bucketed throughput history for *entry_name*, oldest bucket first.

    Each bucket is ``{"t": <unix start>, "in_tps": float|None,
    "out_tps": float|None, "calls": int}``. Rates are token-weighted within
    the bucket (Σtokens / Σseconds), not a mean of per-call rates, so one
    three-token reply cannot dominate a bucket. Empty buckets are kept with
    ``None`` rates so the x-axis stays linear in time.
    """
    if buckets < 1 or hours < 1:
        return []
    now = int(time.time())
    start = now - hours * 3600
    width = max(1, (now - start) // buckets)
    out = [{"t": start + i * width, "in_tps": None, "out_tps": None, "calls": 0}
           for i in range(buckets)]
    try:
        rows = _conn().execute(
            "SELECT ts, in_tokens, out_tokens, gen_sec, ttft FROM samples "
            "WHERE entry_name = ? AND ts >= ? ORDER BY ts",
            (entry_name, start),
        ).fetchall()
    except Exception:
        return out
    acc = [[0, 0.0, 0, 0.0, 0] for _ in range(buckets)]  # in_tok, ttft, out_tok, gen, calls
    for ts, in_tok, out_tok, gen, ttft in rows:
        i = min(buckets - 1, max(0, (ts - start) // width))
        a = acc[i]
        if in_tok and ttft > 0:
            a[0] += in_tok
            a[1] += ttft
        if out_tok and gen > 0:
            a[2] += out_tok
            a[3] += gen
        a[4] += 1
    for i, (in_tok, ttft, out_tok, gen, calls) in enumerate(acc):
        out[i]["calls"] = calls
        if ttft > 0:
            out[i]["in_tps"] = round(in_tok / ttft, 1)
        if gen > 0:
            out[i]["out_tps"] = round(out_tok / gen, 1)
    return out


def window_summary(entry_name: str, hours: int = 24) -> dict:
    """Token-weighted average rates over the window, plus spread.

    ``{"calls", "in_tps", "out_tps", "out_tps_min", "out_tps_max"}`` — the
    min/max are per-call decode rates over samples big enough to be
    meaningful, so a wide gap flags a fluctuating endpoint.
    """
    res: dict = {"calls": 0, "in_tps": None, "out_tps": None,
                 "out_tps_min": None, "out_tps_max": None}
    try:
        row = _conn().execute(
            "SELECT COUNT(*), SUM(in_tokens), SUM(ttft), SUM(out_tokens), SUM(gen_sec), "
            "       MIN(CASE WHEN out_tokens >= 20 AND gen_sec > 0 THEN out_tokens / gen_sec END), "
            "       MAX(CASE WHEN out_tokens >= 20 AND gen_sec > 0 THEN out_tokens / gen_sec END) "
            "FROM samples WHERE entry_name = ? AND ts >= ?",
            (entry_name, int(time.time()) - hours * 3600),
        ).fetchone()
    except Exception:
        return res
    if not row:
        return res
    calls, in_tok, ttft, out_tok, gen, lo, hi = row
    res["calls"] = calls or 0
    if in_tok and ttft:
        res["in_tps"] = round(in_tok / ttft, 1)
    if out_tok and gen:
        res["out_tps"] = round(out_tok / gen, 1)
    if lo is not None:
        res["out_tps_min"] = round(lo, 1)
        res["out_tps_max"] = round(hi, 1)
    return res


def ttft_samples(entry_name: str, days: int = 30) -> list[tuple[int, float]]:
    """``[(in_tokens, ttft), …]`` prefill measurements for *entry_name*.

    Feeds ``ttft_expect``: only rows that measured an actual prefill (both
    fields set) are useful for predicting how long the next one will take.
    """
    try:
        cutoff = int(time.time()) - max(1, days) * 86400
        rows = _conn().execute(
            "SELECT in_tokens, ttft FROM samples "
            "WHERE entry_name = ? AND ts >= ? AND in_tokens > 0 AND ttft > 0",
            (entry_name, cutoff),
        ).fetchall()
    except Exception:
        return []
    return [(int(n), float(t)) for n, t in rows]


def known_entries(days: int = 30) -> list[str]:
    """Entry names with at least one sample in the window."""
    try:
        cutoff = int(time.time()) - days * 86400
        return [r[0] for r in _conn().execute(
            "SELECT entry_name, COUNT(*) c FROM samples WHERE ts >= ? "
            "GROUP BY entry_name ORDER BY c DESC", (cutoff,)).fetchall()]
    except Exception:
        return []
