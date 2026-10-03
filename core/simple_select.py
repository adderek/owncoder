"""Simple model selection: where · model · effort.

Most users want three plain choices, the way codex / Claude Code / Cursor offer
them, not a role matrix:

    where   auto | local | cloud           → agent.model_mode
    model   auto | fast | balanced | strong → auto_tier ladder effort
    effort  off | low | medium | high | xhigh → llm.think_level (thinking budget)

This module only writes the existing knobs; /mode, /effort, /think and /model
stay the precise controls underneath. Reading goes the other way: the simple
view is derived from those knobs, and a state it cannot name (a /model pin,
paid-cloud mode, …) shows as "custom" with the raw value.

`/use cloud strong xhigh`, `/use fast`, `/use auto` — tokens are disjoint, so
order does not matter and any subset may be given. Session-only, like the
knobs it sets.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

WHERE = ("auto", "local", "cloud")
MODEL = ("auto", "fast", "balanced", "strong")
EFFORT = ("off", "low", "medium", "high", "xhigh")

_WHERE_MODE = {"local": "private", "cloud": "cloud"}
_MODEL_EFFORT = {"auto": "smart", "fast": "quick", "balanced": "balanced", "strong": "deep"}
_EFFORT_THINK = {"off": "off", "low": "low", "medium": "normal", "high": "high", "xhigh": "max"}
_THINK_EFFORT = {"off": "off", "low": "low", "normal": "medium", "med": "medium",
                 "medium": "medium", "high": "high", "max": "xhigh"}
# Accepted spellings → canonical token.
_ALIASES = {
    "private": "local", "lan": "local", "offline": "local", "remote": "cloud",
    "quick": "fast", "small": "fast", "mid": "balanced", "normal-model": "balanced",
    "smart": "balanced", "deep": "strong", "big": "strong",
    "none": "off", "med": "medium", "max": "xhigh", "x-high": "xhigh",
}


def _auto_mode(config) -> str:
    """The model-mode "auto" returns to: whatever was active before the first
    simple switch (configured default or the startup profile's pick)."""
    mode = getattr(config, "_simple_auto_mode", None)
    if mode is None:
        mode = config.agent.model_mode
        config._simple_auto_mode = mode
    return mode


def state(config: "Config") -> dict:
    """Current simple view, derived from the precise knobs."""
    auto_mode = getattr(config, "_simple_auto_mode", None) or config.agent.model_mode
    mode = config.agent.model_mode
    where = {"private": "local", "cloud": "cloud"}.get(mode)
    if where is None:
        where = "auto" if mode == auto_mode else "custom"

    at = getattr(config, "auto_tier", None)
    ladder = bool(at and at.enabled and getattr(at, "ladder", False))
    pinned = bool(getattr(config, "runtime_model_pinned", False))
    if pinned:
        model = "custom"
    elif not ladder:
        # No per-turn picking: the configured default entry answers.
        model = "default" if not (at and at.enabled) else "custom"
    else:
        model = {v: k for k, v in _MODEL_EFFORT.items()}.get(getattr(at, "effort", "smart"), "custom")

    think = (config.llm.think_level or "normal").lower()
    effort = _THINK_EFFORT.get(think, "custom")
    return {"where": where, "model": model, "effort": effort,
            "raw": {"mode": mode, "auto_tier": (getattr(at, "effort", None) if ladder else
                                                ("on" if at and at.enabled else "off")),
                    "think": think, "active_model": config.llm.model,
                    "pinned": pinned},
            "choices": {"where": list(WHERE), "model": list(MODEL), "effort": list(EFFORT)}}


def label(st: dict) -> str:
    def part(k: str) -> str:
        v = st[k]
        if v == "default":
            return str(st["raw"]["active_model"])
        if v != "custom":
            return v
        raw = st["raw"]
        return {"where": raw["mode"],
                "model": ("📌" if raw["pinned"] else "") + str(raw["active_model"]),
                "effort": raw["think"]}[k]
    return " · ".join(part(k) for k in ("where", "model", "effort"))


def parse(arg: str) -> tuple[dict, list[str]]:
    """Tokens → {"where"/"model"/"effort": value}; returns (choice, unknown tokens).

    "auto" alone resets where + model; "key=value" forms disambiguate.
    """
    choice: dict = {}
    bad: list[str] = []
    for tok in (arg or "").lower().replace(",", " ").split():
        key = None
        if "=" in tok:
            key, tok = tok.split("=", 1)
            key = {"thinking": "effort", "think": "effort", "location": "where",
                   "tier": "model"}.get(key, key)
        tok = _ALIASES.get(tok, tok)
        if key in ("where", "model", "effort"):
            allowed = {"where": WHERE, "model": MODEL, "effort": EFFORT}[key]
            if tok in allowed:
                choice[key] = tok
            else:
                bad.append(f"{key}={tok}")
            continue
        if tok == "auto":
            choice.setdefault("where", "auto")
            choice.setdefault("model", "auto")
        elif tok in WHERE:
            choice["where"] = tok
        elif tok in MODEL:
            choice["model"] = tok
        elif tok in EFFORT:
            choice["effort"] = tok
        else:
            bad.append(tok)
    return choice, bad


def apply(config: "Config", choice: dict) -> list[str]:
    """Write the knobs behind *choice*; returns notes worth showing."""
    notes: list[str] = []
    _auto_mode(config)   # remember the pre-simple mode before anything moves it
    if "where" in choice:
        from agent.core.model_mode import _repin_roles
        target = _auto_mode(config) if choice["where"] == "auto" else _WHERE_MODE[choice["where"]]
        if target != config.agent.model_mode:
            config.agent.model_mode = target
            moved = _repin_roles(config)
            if moved:
                notes.append("re-pinned: " + ", ".join(moved))
    if "model" in choice:
        from agent.core.model_tier import set_effort
        notes.extend(set_effort(config, _MODEL_EFFORT[choice["model"]]))
    if "effort" in choice:
        config.llm.think_level = _EFFORT_THINK[choice["effort"]]
    return notes


def ladder_preview(config: "Config") -> str:
    """Which entry the current model choice lands on right now (no probing)."""
    try:
        from agent.core.model_tier import _ladder_pick, build_ladder
        ladder = build_ladder(config, check_available=False)
        if not ladder:
            return "no model allowed here — check /mode"
        eff = getattr(config.auto_tier, "effort", "smart")
        if eff == "smart":
            return f"per turn: {ladder[0][0]} … {ladder[-1][0]}"
        lvl = {"quick": "quick", "deep": "deep"}.get(eff, "normal")
        return f"→ {_ladder_pick(ladder, lvl)}"
    except Exception:
        return ""


def run_use_command(config: "Config", arg: str = "") -> str:
    """/use [where] [model] [effort] — show or set the simple selection."""
    arg = (arg or "").strip()
    notes: list[str] = []
    if arg:
        choice, bad = parse(arg)
        if bad:
            return (f"unknown: {' '.join(bad)}\n"
                    f"where: {'|'.join(WHERE)}  model: {'|'.join(MODEL)}  "
                    f"effort: {'|'.join(EFFORT)}")
        notes = apply(config, choice)
    st = state(config)
    lines = [f"using: {label(st)}"]
    preview = ladder_preview(config) if st["model"] not in ("custom", "default") else ""
    if preview:
        lines.append(f"  model: {preview}")
    lines.extend("  " + n for n in notes)
    if not arg:
        lines.append(f"  /use [{'|'.join(WHERE)}] [{'|'.join(MODEL)}] [{'|'.join(EFFORT)}]")
        lines.append("  precise: /mode (where) · /effort (model ladder) · /think (effort) · "
                     "/model (pin an entry)")
    return "\n".join(lines)
