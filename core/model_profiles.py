"""Model setups: hand-made role pins + ladder order + effort, as one snapshot.

A setup is what the user arranged by hand this session. It travels two ways:

* on the session (``Session.model_setup``) — pins live only in the process, so
  a restart forgets them; on resume the UI offers to put them back;
* as named user profiles in ``~/.config/agent/model_profiles.json`` —
  machine-written, kept apart from hand-edited config like model_state.json.

Entry names are not identity: the same name may point at another model on
another host (``agent.<host>.yaml``). Every referenced entry is therefore
recorded with a ``model@base_url`` fingerprint; on apply a renamed entry with
the same fingerprint is remapped, and an entry that is gone or now points
elsewhere is reported and skipped rather than half-applied in silence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

EFFORTS = ("quick", "smart", "balanced", "deep")


def _fingerprint(entry) -> str:
    return f"{getattr(entry, 'model', '') or ''}@{getattr(entry, 'base_url', '') or ''}"


def capture(config: "Config") -> dict:
    """Snapshot what the user set by hand. Empty dict when nothing was."""
    entries = config.model_entries or {}
    pins = {r: config.model_roles[r]
            for r in sorted(getattr(config, "session_role_pins", None) or ())
            if config.model_roles.get(r) in entries}
    at = config.auto_tier
    ladder = [n for n in (getattr(at, "ladder_order", None) or []) if n in entries]
    # A hand-ordered ladder only means something with ladder mode on, and
    # apply() turns it on — so it carries the effort it will run at.
    effort = at.effort if ((at.enabled and at.ladder) or ladder) else ""
    if not (pins or ladder or effort):
        return {}
    used = set(pins.values()) | set(ladder)
    return {"pins": pins, "ladder": ladder, "effort": effort,
            "entries": {n: _fingerprint(entries[n]) for n in sorted(used)}}


def same(a: dict, b: dict) -> bool:
    """Equal as far as behaviour goes (fingerprints are bookkeeping)."""
    keys = ("pins", "ladder", "effort")
    return all(((a or {}).get(k) or None) == ((b or {}).get(k) or None) for k in keys)


def resolve(config: "Config", setup: dict) -> tuple[dict, list[str]]:
    """Map *setup* onto the live config. Returns (resolved setup, problems).

    Resolved names are live entry names; anything unmappable is dropped from
    the result and listed in problems. No problems = fully compatible.
    """
    from agent.config.loader import env_locked_roles
    entries = config.model_entries or {}
    by_fp: dict[str, str] = {}
    for n, e in sorted(entries.items()):
        by_fp.setdefault(_fingerprint(e), n)
    fps = setup.get("entries") or {}
    problems: list[str] = []
    mapped: dict[str, str | None] = {}

    def _map(name: str) -> str | None:
        if name in mapped:
            return mapped[name]
        want = fps.get(name)
        out: str | None = None
        if name in entries and (not want or _fingerprint(entries[name]) == want):
            out = name
        elif want and want in by_fp:
            out = by_fp[want]
            problems.append(f"'{name}' → '{out}' (renamed, same model)")
        elif name in entries:
            problems.append(f"'{name}' now points at {_fingerprint(entries[name])}, "
                            f"was {want}")
        else:
            problems.append(f"'{name}' is not configured here")
        mapped[name] = out
        return out

    locked = env_locked_roles()
    pins: dict[str, str] = {}
    for role, name in (setup.get("pins") or {}).items():
        if role in locked:
            problems.append(f"role '{role}' is forced by the environment")
            continue
        got = _map(name)
        if got:
            pins[role] = got
    ladder = [g for g in (_map(n) for n in setup.get("ladder") or []) if g]
    effort = setup.get("effort") or ""
    if effort and effort not in EFFORTS:
        problems.append(f"unknown effort '{effort}'")
        effort = ""
    # A rename onto the same model is a remap, not an incompatibility.
    hard = [p for p in problems if "(renamed, same model)" not in p]
    return {"pins": pins, "ladder": ladder, "effort": effort}, hard


def apply(agent, setup: dict) -> list[str]:
    """Make the live model setup equal *setup*. Returns one line per change.

    Pins not in the setup are released, so applying a profile replaces the
    arrangement instead of layering on whatever was pinned before.
    """
    from agent.ui.slash import _apply_model
    from agent.core.model_tier import set_effort
    cfg = agent.config
    want, problems = resolve(cfg, setup)
    lines = [f"skipped: {p}" for p in problems]
    at = cfg.auto_tier
    at.ladder_order = list(want["ladder"])
    if want["effort"]:
        # Before the pins: set_effort releases the default pin by design.
        lines += set_effort(cfg, want["effort"])
    elif want["ladder"] and not (at.enabled and at.ladder):
        at.enabled = at.ladder = True
        lines.append("auto-tier ladder enabled (runtime only).")
    for role in sorted(set(getattr(cfg, "session_role_pins", None) or ()) - set(want["pins"])):
        ok, msg = _apply_model(agent, f"{role}=auto")
        if ok:
            lines.append(f"released {role}")
    for role, name in want["pins"].items():
        ok, msg = _apply_model(agent, f"{role}={name}")
        lines.append(f"pinned {role} → {name}" if ok else f"failed {role}: {msg}")
    if want["ladder"]:
        lines.append("ladder: " + " → ".join(want["ladder"]))
    return lines


# ── named profiles ───────────────────────────────────────────────────────────


def profiles_path() -> Path:
    return Path.home() / ".config" / "agent" / "model_profiles.json"


def load_profiles() -> dict[str, dict]:
    p = profiles_path()
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return {}
    except Exception:
        logger.warning("model profiles: unreadable %s", p, exc_info=True)
        return {}
    prof = data.get("profiles") if isinstance(data, dict) else None
    return {str(k): v for k, v in (prof or {}).items() if isinstance(v, dict)}


def _write(profiles: dict[str, dict]) -> None:
    p = profiles_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"profiles": profiles}, indent=2, sort_keys=True))
    os.replace(tmp, p)


def save_profile(config: "Config", name: str, setup: dict) -> None:
    """Store *setup* under *name*, fingerprinting its entries against *config*."""
    entries = config.model_entries or {}
    pins = {str(r): str(n) for r, n in (setup.get("pins") or {}).items() if n}
    ladder = [str(n) for n in setup.get("ladder") or []]
    used = set(pins.values()) | set(ladder)
    clean = {"pins": pins, "ladder": ladder, "effort": str(setup.get("effort") or ""),
             "entries": {n: _fingerprint(entries[n]) for n in sorted(used) if n in entries}}
    profiles = load_profiles()
    profiles[name] = clean
    _write(profiles)


def delete_profile(name: str) -> bool:
    profiles = load_profiles()
    if profiles.pop(name, None) is None:
        return False
    _write(profiles)
    return True
