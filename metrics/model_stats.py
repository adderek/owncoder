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


def resolve_entry_name(config) -> str:
    """Best-effort registry entry name for the agent's active LLM endpoint.

    Matches config.llm (base_url, model) against the configured model_entries so
    the daily-chat path persists tps under the SAME key the registry/commit path
    uses (e.g. "gpu-gemma4"). Falls back to the raw model name when no entry
    matches — so stats are still recorded for unregistered endpoints.
    """
    try:
        llm = config.llm
        entries = getattr(config, "model_entries", {}) or {}
        for name, entry in entries.items():
            if getattr(entry, "base_url", None) == llm.base_url and getattr(entry, "model", None) == llm.model:
                return name
        return llm.model or "default"
    except Exception:
        return "default"
