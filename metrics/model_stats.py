"""Per-model throughput statistics — EWMA over observed tok/s."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_ALPHA = 0.3      # EWMA weight for new sample
_MIN_TOKENS = 20  # ignore tiny samples (tokenizer overhead dominates)


def _stats_path() -> Path:
    return Path.home() / ".config" / "agent" / "metrics" / "model_stats.json"


def load_stats() -> dict:
    p = _stats_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def update_stats(entry_name: str, tokens: int, elapsed_sec: float) -> None:
    """Record a new tok/s sample for *entry_name* using EWMA."""
    if tokens < _MIN_TOKENS or elapsed_sec <= 0:
        return
    tps = tokens / elapsed_sec
    stats = load_stats()
    rec = stats.get(entry_name, {})
    prev = rec.get("tps_ewma", 0.0)
    new_ewma = tps if prev == 0.0 else _ALPHA * tps + (1 - _ALPHA) * prev
    stats[entry_name] = {
        "tps_ewma": round(new_ewma, 1),
        "tps_last": round(tps, 1),
        "samples": rec.get("samples", 0) + 1,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    p = _stats_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2))
    os.replace(tmp, p)


def get_tps(entry_name: str) -> float:
    """Return EWMA tok/s for *entry_name*, or 0.0 if no data."""
    return load_stats().get(entry_name, {}).get("tps_ewma", 0.0)
