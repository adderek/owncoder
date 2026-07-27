"""`/mode` slash command — view/switch the automatic model-mode.

The model-mode controls which cost tiers (local / free / paid) the agent may
pick for AUTOMATIC selection: idle/background work (session naming, summaries,
compaction) and the spawn_agents decision-maker. An explicitly pinned
``[models] default`` is always honored regardless of mode.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent.config import make_registry, entry_tier, MODE_TIERS

if TYPE_CHECKING:
    from agent.config import Config

_ORDER = ["local-only", "lan-only", "free-cloud", "free-hybrid", "paid-cloud",
          "manual", "any"]


# Probes per role on a mode switch. A slash command must not freeze the UI for
# the length of a whole pool walk, and the first few allowed candidates are the
# user's declared preference anyway; runtime failover handles the rest.
_REPIN_PROBE_LIMIT = 3


def _repin_roles(config: "Config") -> list[str]:
    """Move pool-resolved roles onto live endpoints the new mode allows.

    Roles pinned from a pool at startup (``default``, usually ``summarizer``)
    are concrete entries in ``model_roles`` — a mode switch alone would leave
    them on a now-disallowed endpoint, and ``background``/``namer``/
    ``compaction`` inherit that pin through the fallback chain. Roles the user
    pinned by hand in TOML are indistinguishable here and get repinned too;
    ``embeddings`` is exempt (vector service, not a cost/location decision).
    Returns ["role → entry", …] for the roles actually moved.
    """
    from agent.config.loader import _apply_entry_to_llm, _probe_models
    from agent.config.registry import mode_allows

    mode = config.agent.model_mode
    moved: list[str] = []
    roles = ["default"] + [r for r in sorted(config.model_pools)
                           if r not in ("default", "embeddings")]
    for role in roles:
        if role != "default" and role not in config.model_roles:
            continue  # unresolved pool: nothing pinned to move
        cur_name = config.model_roles.get(role)
        cur = config.model_entries.get(cur_name) if cur_name else None
        if cur is not None and mode_allows(cur, mode):
            continue
        # Only entries the user declared in this role's pool are candidates:
        # the pool is the declared preference order, and it keeps this command
        # from probing every configured endpoint on a plain mode switch.
        tried = 0
        for name in config.model_pools.get(role) or []:
            entry = config.model_entries.get(name)
            if entry is None or not mode_allows(entry, mode):
                continue
            if tried >= _REPIN_PROBE_LIMIT:
                break
            tried += 1
            if _probe_models(entry.base_url, entry.api_key, timeout=2) is None:
                continue
            config.model_roles[role] = name
            if role == "default":
                _apply_entry_to_llm(config, name, entry)
            moved.append(f"{role} → {name}")
            break
    return moved


def run_mode_command(config: "Config", arg: str) -> str:
    """Show or set ``config.agent.model_mode``. Returns text for the sys log."""
    from agent.config.registry import is_lan_entry, mode_allows

    arg = (arg or "").strip().lower()
    if arg and arg not in MODE_TIERS:
        valid = ", ".join(_ORDER)
        return f"unknown mode {arg!r}. valid: {valid}"

    repinned: list[str] = []
    if arg and arg != config.agent.model_mode:
        config.agent.model_mode = arg
        repinned = _repin_roles(config)

    cur = config.agent.model_mode
    reg = make_registry(config)
    if cur == "lan-only":
        head = "model-mode: lan-only  (LAN endpoints only; embeddings exempt)"
    else:
        head = f"model-mode: {cur}  (tiers: {', '.join(sorted(MODE_TIERS.get(cur, set())))})"
    lines = [head]

    # Group configured entries by tier so the user sees what each mode unlocks.
    # LAN entries are cost-tier "free" but shown separately — "lan-only" picks
    # on location, so the free row alone would not explain what it admits.
    by_tier: dict[str, list[str]] = {"local": [], "lan": [], "free": [], "paid": []}
    for name, entry in config.model_entries.items():
        tier = "lan" if is_lan_entry(entry) else entry_tier(entry)
        by_tier.setdefault(tier, []).append(name)
    for tier in ("local", "lan", "free", "paid"):
        names = ", ".join(sorted(by_tier.get(tier, []))) or "—"
        sample = by_tier.get(tier) or []
        ok = bool(sample) and mode_allows(config.model_entries[sample[0]], cur)
        lines.append(f" {'*' if ok else ' '} {tier:5s}: {names}")

    allowed = reg.allowed_names()
    bg = reg.background
    bg_name = next((n for n, e in config.model_entries.items() if e is bg), "?")
    lines.append(f"background/idle → {bg_name}")
    if repinned:
        lines.append("re-pinned for this mode: " + ", ".join(repinned))
    if not allowed:
        lines.append("WARNING: no configured model matches this mode")
    if not arg:
        lines.append(f"switch: /mode {{{('|'.join(_ORDER))}}}")
    return "\n".join(lines)
