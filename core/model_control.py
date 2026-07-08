"""Session-scoped model enable/disable + failure-cooldown status.

A disabled entry is skipped by the tier ladder, tier resolution, and local
failover for the rest of the session (runtime only — nothing is persisted).
Failure cooldowns live in model_probe (mark_rate_limited / is_rate_limited):
any hard endpoint failure puts the (base_url, model) pair on a timed cooldown,
after which the normal /models availability probe must succeed before the
entry is considered live again — that probe IS the retry-to-revive step.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def disabled_set(config) -> set:
    """The session's set of disabled entry names (created lazily)."""
    s = getattr(config, "runtime_disabled_models", None)
    if s is None:
        s = set()
        try:
            config.runtime_disabled_models = s
        except Exception:
            pass
    return s


def is_disabled(config, entry_name: str) -> bool:
    return entry_name in disabled_set(config)


def set_model_enabled(config, entry_name: str, enabled: bool) -> tuple[bool, str]:
    """Enable/disable *entry_name* for this session. Returns (ok, message)."""
    entries = getattr(config, "model_entries", None) or {}
    if entry_name not in entries:
        return False, f"unknown model entry '{entry_name}' (see /models)"
    s = disabled_set(config)
    if enabled:
        if entry_name not in s:
            return True, f"'{entry_name}' already enabled"
        s.discard(entry_name)
        return True, f"'{entry_name}' enabled"
    active = (getattr(config, "model_roles", None) or {}).get("default", "")
    s.add(entry_name)
    note = " (note: it is the active default — switch with /model)" if entry_name == active else ""
    return True, f"'{entry_name}' disabled for this session{note}"


def entry_status(config, entry_name: str, entry) -> str:
    """One-word status for display: 'off' (disabled), 'cool' (failure cooldown),
    '' (normal)."""
    if is_disabled(config, entry_name):
        return "off"
    try:
        from agent.config.model_probe import is_rate_limited
        if is_rate_limited(getattr(entry, "base_url", "") or "", getattr(entry, "model", "") or ""):
            return "cool"
    except Exception:
        pass
    return ""
