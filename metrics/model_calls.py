"""Per-tier model-call counters — how many LLM calls hit each cost tier.

A process-wide *session* counter (cumulative for the life of the process) plus a
*round* counter that is reset at the start of every agent turn. Tiers mirror
``config/registry.py`` ``entry_tier``: local / free / bundled / paid.

Every LLM dispatch — the main agent turn and every background/role call
(summarizer, namer, compaction, security review/triage/verify/evolve, commit,
promoter, reflector, skill distiller) — records one call against the cost tier
of the model entry it used. The UI prints the round breakdown after each turn;
``/modelcalls`` shows the session totals.
"""
from __future__ import annotations

from collections import Counter

# Display order. Unknown tiers (should not happen) are appended after these.
TIERS = ("local", "free", "bundled", "paid")

_session: Counter = Counter()
_round: Counter = Counter()


def record(tier: str | None) -> None:
    """Record one LLM call against *tier* (empty/None → ``local``)."""
    t = tier or "local"
    _session[t] += 1
    _round[t] += 1


def record_entry(entry) -> None:
    """Record one call against the cost tier of a registry ``ModelEntry``."""
    try:
        from agent.config.registry import entry_tier
        record(entry_tier(entry))
    except Exception:
        record("local")


def record_main(config) -> None:
    """Record one call against the tier of the main ``config.llm`` endpoint."""
    try:
        from agent.config.registry import entry_tier
        from agent.metrics.model_stats import resolve_entry_name
        name = resolve_entry_name(config)
        entry = (getattr(config, "model_entries", {}) or {}).get(name)
        record(entry_tier(entry) if entry is not None else "local")
    except Exception:
        record("local")


def record_entry_name(config, name: str) -> None:
    """Record one call against the tier of the model entry called *name*."""
    try:
        from agent.config.registry import entry_tier
        entry = (getattr(config, "model_entries", {}) or {}).get(name)
        record(entry_tier(entry) if entry is not None else "local")
    except Exception:
        record("local")


def reset_round() -> None:
    """Clear the per-round counter — called at the start of each agent turn."""
    _round.clear()


def round_counts() -> dict:
    return dict(_round)


def session_counts() -> dict:
    return dict(_session)


def _ordered(counts: dict) -> list[str]:
    extra = [t for t in counts if t not in TIERS]
    return [t for t in TIERS if counts.get(t)] + [t for t in extra if counts.get(t)]


def format_line(counts: dict, label: str = "models") -> str:
    """One-line breakdown, e.g. ``models: 4 calls (local=3 paid=1)``. Empty if none."""
    total = sum(counts.values())
    if not total:
        return ""
    parts = " ".join(f"{t}={counts[t]}" for t in _ordered(counts))
    return f"{label}: {total} call{'s' if total != 1 else ''} ({parts})"


def run_modelcalls_command(arg: str = "") -> str:
    """Slash handler — session totals (``reset`` clears them)."""
    if arg.strip().lower() == "reset":
        _session.clear()
        return "model-call counters reset."
    line = format_line(session_counts(), label="model calls this session")
    return line or "model calls this session: none yet."
