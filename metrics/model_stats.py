"""Per-model throughput statistics — EWMA over observed tok/s.

Persisted per registry *entry name* (``resolve_entry_name``) in
``~/.config/agent/metrics/model_stats.json``. Two throughput numbers are
tracked:

* **out tok/s** — completion tokens ÷ generation seconds (time from the first
  token to the last). The classic decode rate.
* **in tok/s** — *uncached* prompt tokens ÷ TTFT. This is prefill throughput
  only when the prompt actually had to be processed; tokens the endpoint
  served from its cache are subtracted from the numerator by the caller
  (``Agent._record_usage``) so a cache hit does not inflate the rate. TTFT
  still includes queueing/network, so treat it as an end-to-end prefill rate,
  not a raw GPU number.

Alongside the EWMAs the file keeps cumulative counters (calls, tokens,
seconds) so a true long-run average can be shown next to the EWMA.

Writes are load-modify-atomic-replace: concurrent writers from background
roles can lose the odd sample. Accepted — these are display metrics, and the
lossiness is bounded by the EWMA anyway.
"""
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


def _ewma(prev: float, sample: float) -> float:
    return sample if not prev else _ALPHA * sample + (1 - _ALPHA) * prev


def update_stats(entry_name: str, tokens: int, elapsed_sec: float,
                 in_tokens: int = 0, ttft: float = 0.0) -> None:
    """Record a new throughput sample for *entry_name* using EWMA.

    *tokens*/*elapsed_sec* are the completion tokens and generation seconds
    (the decode rate). *in_tokens*/*ttft* are the uncached prompt tokens and
    time-to-first-token — supplied by callers that know them, and gated
    independently of the output sample so a two-word reply still contributes a
    prefill measurement.
    """
    out_ok = tokens >= _MIN_TOKENS and elapsed_sec > 0
    in_ok = in_tokens > 0 and ttft and ttft > 0
    if not out_ok and not in_ok:
        return
    # Raw sample → long-term history db, so throughput over time (trends,
    # fluctuation, graphs) survives the EWMA collapsing everything into one
    # number. Best-effort: never let metrics break a turn.
    try:
        from agent.metrics.model_history import record_sample
        record_sample(entry_name, out_tokens=tokens if out_ok else 0,
                      gen_sec=elapsed_sec if out_ok else 0.0,
                      in_tokens=in_tokens if in_ok else 0,
                      ttft=float(ttft) if in_ok else 0.0)
    except Exception:
        pass
    stats = load_stats()
    rec = dict(stats.get(entry_name, {}))
    if out_ok:
        tps = tokens / elapsed_sec
        rec["tps_ewma"] = round(_ewma(rec.get("tps_ewma", 0.0), tps), 1)
        rec["tps_last"] = round(tps, 1)
        rec["samples"] = rec.get("samples", 0) + 1
        rec["tokens_out"] = rec.get("tokens_out", 0) + int(tokens)
        rec["gen_seconds"] = round(rec.get("gen_seconds", 0.0) + elapsed_sec, 2)
    if in_ok:
        itps = in_tokens / ttft
        rec["in_tps_ewma"] = round(_ewma(rec.get("in_tps_ewma", 0.0), itps), 1)
        rec["in_tps_last"] = round(itps, 1)
        rec["in_samples"] = rec.get("in_samples", 0) + 1
        rec["tokens_in"] = rec.get("tokens_in", 0) + int(in_tokens)
        rec["ttft_ewma"] = round(_ewma(rec.get("ttft_ewma", 0.0), float(ttft)), 3)
        rec["ttft_last"] = round(float(ttft), 3)
        rec["ttft_seconds"] = round(rec.get("ttft_seconds", 0.0) + float(ttft), 2)
    rec["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stats[entry_name] = rec
    p = _stats_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2))
    os.replace(tmp, p)


def get_tps(entry_name: str) -> float:
    """Return EWMA tok/s for *entry_name*, or 0.0 if no data."""
    return load_stats().get(entry_name, {}).get("tps_ewma", 0.0)


def stats_for(entry_name: str, stats: dict | None = None) -> dict:
    """Display-ready throughput record for *entry_name* (empty dict if none).

    Pass *stats* (a `load_stats()` snapshot) when summarising many entries at
    once so the JSON is read a single time. Adds the cumulative averages
    (``tps_avg``/``in_tps_avg``) derived from the counters.
    """
    rec = (load_stats() if stats is None else stats).get(entry_name) or {}
    if not rec:
        return {}
    out = dict(rec)
    gen = rec.get("gen_seconds", 0.0)
    if gen > 0 and rec.get("tokens_out"):
        out["tps_avg"] = round(rec["tokens_out"] / gen, 1)
    ttft = rec.get("ttft_seconds", 0.0)
    if ttft > 0 and rec.get("tokens_in"):
        out["in_tps_avg"] = round(rec["tokens_in"] / ttft, 1)
    return out


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
