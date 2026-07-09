"""Per-tier model-call counters — how many LLM calls hit each cost tier.

A process-wide *session* counter (cumulative for the life of the process) plus a
*round* counter that is reset at the start of every agent turn. Tiers mirror
``config/registry.py`` ``entry_tier``: local / free / bundled / paid.

Every LLM dispatch — the main agent turn and every background/role call
(summarizer, namer, compaction, security review/triage/verify/evolve, commit,
promoter, reflector, skill distiller) — records one call against the cost tier
of the model entry it used. The UI prints the round breakdown after each turn;
``/modelcalls`` shows the session totals.

Alongside the tier counters, every call is also recorded with its *role*
(which subsystem dispatched it) and *model* (entry/model identifier). The
per-round detail backs the clickable round line in the Textual UI;
``/modelcalls detail`` shows the session-wide role × model table.
"""
from __future__ import annotations

import time
from collections import Counter

# Display order. Unknown tiers (should not happen) are appended after these.
TIERS = ("local", "free", "bundled", "paid")

_session: Counter = Counter()
_round: Counter = Counter()
# Per-call detail: list of {"role", "model", "tier", "t"} dicts for the
# current round ("t" = seconds since round start), and a session-wide
# Counter keyed by (role, model, tier).
_round_detail: list[dict] = []
_session_detail: Counter = Counter()
_round_started: float = time.monotonic()


def record(tier: str | None, role: str = "", model: str = "") -> None:
    """Record one LLM call against *tier* (empty/None → ``local``).

    *role* names the dispatching subsystem ("main", "summarizer", …) and
    *model* the model/entry identifier; both default to "?" when unknown.
    """
    t = tier or "local"
    _session[t] += 1
    _round[t] += 1
    r = role or "?"
    m = model or "?"
    _round_detail.append({"role": r, "model": m, "tier": t,
                          "t": time.monotonic() - _round_started})
    _session_detail[(r, m, t)] += 1


def record_entry(entry, role: str = "") -> None:
    """Record one call against the cost tier of a registry ``ModelEntry``."""
    try:
        from agent.config.registry import entry_tier
        record(entry_tier(entry), role=role, model=getattr(entry, "model", ""))
    except Exception:
        record("local", role=role)


def record_main(config, role: str = "main") -> None:
    """Record one call against the tier of the main ``config.llm`` endpoint."""
    try:
        from agent.config.registry import entry_tier
        from agent.metrics.model_stats import resolve_entry_name
        name = resolve_entry_name(config)
        entry = (getattr(config, "model_entries", {}) or {}).get(name)
        model = getattr(entry, "model", "") or name or getattr(
            getattr(config, "llm", None), "model", "")
        record(entry_tier(entry) if entry is not None else "local",
               role=role, model=model)
    except Exception:
        record("local", role=role)


def record_entry_name(config, name: str, role: str = "") -> None:
    """Record one call against the tier of the model entry called *name*."""
    try:
        from agent.config.registry import entry_tier
        entry = (getattr(config, "model_entries", {}) or {}).get(name)
        model = getattr(entry, "model", "") or name
        record(entry_tier(entry) if entry is not None else "local",
               role=role, model=model)
    except Exception:
        record("local", role=role)


def reset_round() -> None:
    """Clear the per-round counters — called at the start of each agent turn."""
    global _round_started
    _round.clear()
    _round_detail.clear()
    _round_started = time.monotonic()


def round_duration() -> float:
    """Seconds elapsed since the current round started."""
    return time.monotonic() - _round_started


def round_counts() -> dict:
    return dict(_round)


def round_detail() -> list[dict]:
    """Snapshot of the current round's calls: [{"role","model","tier"}, …]."""
    return [dict(d) for d in _round_detail]


def session_counts() -> dict:
    return dict(_session)


def _ordered(counts: dict) -> list[str]:
    extra = [t for t in counts if t not in TIERS]
    return [t for t in TIERS if counts.get(t)] + [t for t in extra if counts.get(t)]


def format_duration(seconds: float) -> str:
    """Human round duration: ``8.2s`` / ``1m 04s`` / ``1h 02m``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def format_line(counts: dict, label: str = "models",
                duration: float | None = None) -> str:
    """One-line breakdown, e.g. ``models: 4 calls (local=3 paid=1) in 8.2s``.
    Empty if no calls. *duration* (seconds) is appended when given."""
    total = sum(counts.values())
    if not total:
        return ""
    parts = " ".join(f"{t}={counts[t]}" for t in _ordered(counts))
    line = f"{label}: {total} call{'s' if total != 1 else ''} ({parts})"
    if duration is not None and duration > 0:
        line += f" in {format_duration(duration)}"
    return line


def format_detail(detail: "Counter | dict") -> list[str]:
    """Table lines for a (role, model, tier) → count mapping, biggest first."""
    if not detail:
        return []
    items = sorted(detail.items(), key=lambda kv: (-kv[1], kv[0]))
    rw = max(len(k[0]) for k, _ in items)
    mw = max(len(k[1]) for k, _ in items)
    return [f"{r:<{rw}}  {m:<{mw}}  {t:<7}  ×{n}" for (r, m, t), n in items]


def run_modelcalls_command(arg: str = "") -> str:
    """Slash handler — session totals; ``detail`` adds the role × model table;
    ``reset`` clears the counters."""
    a = arg.strip().lower()
    if a == "reset":
        _session.clear()
        _session_detail.clear()
        return "model-call counters reset."
    line = format_line(session_counts(), label="model calls this session")
    if not line:
        return "model calls this session: none yet."
    if a == "detail":
        lines = format_detail(_session_detail)
        if lines:
            line += "\n" + "\n".join(lines)
    return line
