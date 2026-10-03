"""Diagnostics log for token_watch: evidence to judge it later.

Every evaluated model call, every action outcome, every calibration merge and
regime change is appended to ~/.config/agent/token_watch/diag/YYYY-MM.jsonl
(user-global, next to the calibration it explains). Records:

- call     metrics (ppl, window entropy/p extremes, ctx tokens), thresholds
           used + which layer they came from, events, learned or why not.
- outcome  what followed an action: the retried call's / next call's events —
           did a retry clear the derail, did a claim note make the model
           verify or hedge.
- merge / regime_change   calibration layer changes.

`report(days)` (/tokwatch diag) condenses it: event rates per model, retry and
note success, threshold trajectory per fingerprint. Raw lines stay for a
refactor-time replay. Capped at diag_max_mb per month file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
SCHEMA = 1
max_mb: float = 50.0
enabled: bool = True


def diag_dir() -> Path:
    from .token_watch_calib import base_dir
    return base_dir() / "diag"


def _file(ts: float | None = None) -> Path:
    return diag_dir() / time.strftime("%Y-%m.jsonl", time.localtime(ts or time.time()))


def configure(cfg) -> None:
    global max_mb, enabled
    max_mb = float(getattr(cfg, "diag_max_mb", max_mb))
    enabled = bool(getattr(cfg, "diag", enabled))


def log(record: dict) -> None:
    """Append one record; never raises."""
    if not enabled:
        return
    try:
        rec = {"v": SCHEMA, "ts": round(time.time(), 3), **record}
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        f = _file()
        with _lock:
            f.parent.mkdir(parents=True, exist_ok=True)
            if f.exists() and f.stat().st_size > max_mb * 1024 * 1024:
                return
            fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))   # O_APPEND: one write = one whole line
            finally:
                os.close(fd)
    except Exception:
        logger.debug("token_watch diag: write failed", exc_info=True)


def call_record(*, ctx, model: str | None, stats: dict, eff, events: list[dict],
                learned: dict, tripped: str | None, summary: dict | None) -> dict:
    cal = getattr(eff, "calibration", {}) or {}
    return {
        "type": "call", "session": getattr(ctx, "session", None), "turn": getattr(ctx, "turn", None),
        "model": model, "fingerprint": cal.get("fingerprint"),
        "ctx_tokens": getattr(ctx, "ctx_tokens", None),
        "metrics": {"n": stats.get("n"), "windows": stats.get("windows"),
                    "ppl": _r(stats.get("ppl")), "h_max": _r(stats.get("h_max")),
                    "p_min": _r(stats.get("p_min")),
                    "mean_H": (summary or {}).get("mean_H"), "min_p": (summary or {}).get("min_p"),
                    "low_p": (summary or {}).get("low_p"), "tool_ppl": (summary or {}).get("tool_ppl")},
        "thresholds": {**cal.get("values", {}), "source": cal.get("source"),
                       "prior": cal.get("prior"), "slow_weight": cal.get("slow_weight"),
                       "slow_sessions": cal.get("slow_sessions"), "fast_calls": cal.get("fast_calls"),
                       "loosened": cal.get("loosened") or None,
                       "baseline_ppl": _r(cal.get("baseline_ppl"))},
        "events": [{k: ev.get(k) for k in ("kind", "action", "value", "threshold", "cut", "text")}
                   for ev in events],
        "tripped": tripped, "learned": learned.get("learned"), "skip": learned.get("reason") or None,
    }


def _r(v, n: int = 4):
    return None if v is None else round(float(v), n)


# ── report ────────────────────────────────────────────────────────────────

def _records(days: int):
    cutoff = time.time() - days * 86400
    d = diag_dir()
    if not d.exists():
        return
    for f in sorted(d.glob("*.jsonl")):
        try:
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts", 0) >= cutoff:
                        yield rec
        except OSError:
            continue


def report(days: int = 30) -> str:
    calls: Counter = Counter()
    events: dict = defaultdict(Counter)
    actions: Counter = Counter()
    outcomes: dict = defaultdict(Counter)
    learned: Counter = Counter()
    sources: Counter = Counter()
    traj: dict = defaultdict(list)
    regimes: list = []
    merges = 0
    for rec in _records(days):
        t = rec.get("type")
        if t == "call":
            m = rec.get("model") or "?"
            calls[m] += 1
            learned["yes" if rec.get("learned") else (rec.get("skip") or "no")] += 1
            th = rec.get("thresholds") or {}
            sources[th.get("source") or "?"] += 1
            fp = rec.get("fingerprint") or "?"
            traj[(m, fp)].append((th.get("derail_entropy"), th.get("derail_p")))
            for ev in rec.get("events") or []:
                events[m][ev.get("kind")] += 1
                actions[f"{ev.get('kind')}→{ev.get('action')}"] += 1
        elif t == "outcome":
            outcomes[f"{rec.get('kind')}→{rec.get('action')}"][
                "cleared" if rec.get("cleared") else "persisted"] += 1
        elif t == "merge":
            merges += 1
        elif t == "regime_change":
            regimes.append(rec)
    if not calls:
        return f"token_watch diag: no records in the last {days} days ({diag_dir()})"
    lines = [f"token_watch diag — last {days} days ({diag_dir()})"]
    for m, n in calls.most_common():
        ev = ", ".join(f"{k} {c} ({c / n:.1%})" for k, c in events[m].most_common()) or "no events"
        lines.append(f"  {m}: {n} calls · {ev}")
    if actions:
        lines.append("  actions: " + ", ".join(f"{k} {c}" for k, c in actions.most_common()))
    for k, c in outcomes.items():
        tot = sum(c.values())
        lines.append(f"  outcome {k}: {c['cleared']}/{tot} cleared")
    lines.append("  thresholds from: " + ", ".join(f"{k} {c}" for k, c in sources.most_common()))
    lines.append("  learned: " + ", ".join(f"{k} {c}" for k, c in learned.most_common()))
    for (m, fp), pts in sorted(traj.items()):
        pts = [p for p in pts if p[0] is not None]
        if pts:
            lines.append(f"  {m} [{fp}] derail_entropy {pts[0][0]}→{pts[-1][0]}, "
                         f"derail_p {pts[0][1]}→{pts[-1][1]} over {len(pts)} calls")
    lines.append(f"  slow merges: {merges} · regime changes: {len(regimes)}")
    for r in regimes[-5:]:
        lines.append(f"    {time.strftime('%Y-%m-%d %H:%M', time.localtime(r['ts']))} "
                     f"{r.get('model')}: {', '.join(r.get('changed', {}))}")
    return "\n".join(lines)
