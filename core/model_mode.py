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

_ORDER = ["local-only", "free-cloud", "free-hybrid", "paid-cloud", "manual", "any"]


def run_mode_command(config: "Config", arg: str) -> str:
    """Show or set ``config.agent.model_mode``. Returns text for the sys log."""
    arg = (arg or "").strip().lower()
    if arg and arg not in MODE_TIERS:
        valid = ", ".join(_ORDER)
        return f"unknown mode {arg!r}. valid: {valid}"

    if arg:
        config.agent.model_mode = arg

    cur = config.agent.model_mode
    reg = make_registry(config)
    lines = [f"model-mode: {cur}  (tiers: {', '.join(sorted(MODE_TIERS.get(cur, set())))})"]

    # Group configured entries by tier so the user sees what each mode unlocks.
    by_tier: dict[str, list[str]] = {"local": [], "free": [], "paid": []}
    for name, entry in config.model_entries.items():
        by_tier.setdefault(entry_tier(entry), []).append(name)
    for tier in ("local", "free", "paid"):
        names = ", ".join(sorted(by_tier.get(tier, []))) or "—"
        mark = "*" if tier in MODE_TIERS.get(cur, set()) else " "
        lines.append(f" {mark} {tier:5s}: {names}")

    allowed = reg.allowed_names()
    bg = reg.background
    bg_name = next((n for n, e in config.model_entries.items() if e is bg), "?")
    lines.append(f"background/idle → {bg_name}")
    if not allowed:
        lines.append("WARNING: no configured model matches this mode")
    if not arg:
        lines.append(f"switch: /mode {{{('|'.join(_ORDER))}}}")
    return "\n".join(lines)
