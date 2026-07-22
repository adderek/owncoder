"""Endpoint routing helpers: remote→local failover and per-turn privacy routing.

Both serve the only two reasons a local model exists in a remote-default setup:
availability (keep working when the remote endpoint is down) and privacy (keep
secrets off the wire). Every route these helpers take is *toward* local, never
away — so they can never weaken privacy or push work onto a remote the operator
did not already configure as the active endpoint.

Mutation shape mirrors ``model_tier.escalate_mid_turn``: point ``config.llm`` at
a new entry and return a fresh ``AsyncOpenAI`` client for the caller (run_turn)
to reassign. ``agent._client`` is intentionally left alone.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


def is_remote_endpoint(config: "Config") -> bool:
    """True if the active LLM endpoint is not local (loopback / no base_url)."""
    from agent.security.airgap import is_local_url
    try:
        return not is_local_url(config.llm.base_url)
    except Exception:
        return bool(config.llm.base_url)


def resolve_local_entry(config: "Config", preferred: str = "") -> Optional[str]:
    """Name of a self-hosted model entry to degrade to, or None if none is live.

    Prefers a loopback ("local") endpoint, then a LAN ("remote", private-IP)
    endpoint — both are the operator's own hardware, so degrading to either
    never sends data to a third-party cloud. When the turn is pinned local-only
    (``config.runtime_local_only`` — private session mode) LAN entries are
    excluded so a private turn cannot leave the loopback interface.

    Availability is probed: a configured-but-down box (the exact case that made
    the crash surface) is skipped instead of being switched to blindly.
    """
    from agent.core.model_control import is_disabled
    # loader.entry_tier classifies by *location* (local=loopback / remote=LAN /
    # cloud), which is what a degrade needs — not the cost tier that
    # agent.config.entry_tier returns (a LAN box is "free" there, not "remote").
    from agent.config.loader import entry_tier
    from agent.config.model_probe import entry_available
    entries = config.model_entries or {}
    local_only = bool(getattr(config, "runtime_local_only", False))
    allowed_tiers = ("local",) if local_only else ("local", "remote")

    def _usable(name: str, e) -> bool:
        return (entry_tier(e) in allowed_tiers
                and not is_disabled(config, name)
                and entry_available(e))

    if preferred and preferred in entries and _usable(preferred, entries[preferred]):
        return preferred
    # Two passes so a live loopback endpoint always wins over a live LAN one.
    for want in allowed_tiers:
        for name, e in entries.items():
            if entry_tier(e) == want and not is_disabled(config, name) and entry_available(e):
                return name
    return None


def switch_to_entry(config: "Config", entry_name: str):
    """Point ``config.llm`` at *entry_name*; return a fresh client, or None if the
    entry is unknown or already active."""
    e = (config.model_entries or {}).get(entry_name)
    if e is None:
        return None
    target_model = e.model or config.llm.model
    if config.llm.base_url == e.base_url and config.llm.model == target_model:
        return None  # already active
    config.llm.base_url = e.base_url
    config.llm.api_key = e.api_key
    if e.model:
        config.llm.model = e.model
    config.llm.ctx_window = e.ctx_window
    config.llm.max_output_tokens = e.max_output_tokens
    config.llm.temperature = e.temperature
    from agent.core.llm_client import make_llm_client
    return make_llm_client(config, base_url=e.base_url, api_key=e.api_key)


# ── Remote → local failover ───────────────────────────────────────────────────

def failover_to_local(config: "Config"):
    """Degrade a dead remote endpoint to a local model for the rest of the turn.

    Returns a fresh local client, or None if failover is disabled, the endpoint
    is already local, or no local entry is configured.
    """
    cfg = getattr(config, "failover", None)
    if cfg is None or not cfg.enabled:
        return None
    if not is_remote_endpoint(config):
        return None
    name = resolve_local_entry(config, getattr(cfg, "local_entry", ""))
    if not name:
        logger.warning("failover: no local model entry configured — cannot degrade offline")
        return None
    client = switch_to_entry(config, name)
    if client is not None:
        logger.warning("failover: remote unreachable — degraded to local entry '%s'", name)
    return client


def failover_to_alternative(config: "Config"):
    """Rescue a turn whose *local* endpoint is failing — e.g. a router that
    accepts the request but 500s because the preset's weights are missing.

    Switches to another live local-tier entry (failover.local_entry first).
    Stays within this module's invariant — never routes toward remote — so a
    turn that privacy routing pinned to local can never leak through failover.

    Returns a fresh client, or None if failover is disabled or no other live
    local entry exists. Entries on failure cooldown (mark_rate_limited) are
    skipped, so the entry that just failed is never picked again this window.
    """
    cfg = getattr(config, "failover", None)
    if cfg is None or not cfg.enabled:
        return None
    from agent.config.loader import entry_tier  # location tier (local/remote/cloud)
    from agent.config.model_probe import entry_available
    from agent.core.model_control import is_disabled
    entries = config.model_entries or {}
    local_only = bool(getattr(config, "runtime_local_only", False))
    allowed_tiers = ("local",) if local_only else ("local", "remote")
    preferred = getattr(cfg, "local_entry", "") or ""
    ordered = [preferred] if preferred in entries else []
    ordered += [n for n in entries if n not in ordered]
    active = (config.llm.base_url, config.llm.model)
    for name in ordered:
        e = entries[name]
        if entry_tier(e) not in allowed_tiers or is_disabled(config, name):
            continue
        if (e.base_url, e.model or config.llm.model) == active:
            continue
        if not entry_available(e):
            continue
        client = switch_to_entry(config, name)
        if client is not None:
            logger.warning("failover: local endpoint failing — switched to entry '%s'", name)
            return client
    return None


def failover_to_peer(config: "Config"):
    """Rescue a failing *cloud* endpoint by switching to another live cloud
    entry allowed under the current model-mode — e.g. free provider A hits its
    daily 429 cap, so the turn continues on free provider B instead of
    immediately degrading to local.

    Preserves this module's privacy invariant: only runs when the active
    endpoint is already remote (the payload was already leaving the machine)
    and never when the turn is pinned local (``runtime_local_only``), so it can
    never route a local-pinned turn toward a cloud.

    Candidate order: same cost tier as the active entry first, then the other
    mode-allowed cloud tiers. Skips disabled entries and (base_url, model)
    pairs on rate-limit/failure cooldown (mark_rate_limited), so the entry
    that just failed is never re-picked within its cooldown window.

    Returns a fresh client, or None if failover is disabled or no live peer
    exists (callers then degrade to local as before).
    """
    cfg = getattr(config, "failover", None)
    if cfg is None or not cfg.enabled:
        return None
    if not is_remote_endpoint(config):
        return None
    if bool(getattr(config, "runtime_local_only", False)):
        return None
    from agent.config.registry import MODE_TIERS, entry_tier
    from agent.config.loader import entry_tier as location_tier
    from agent.config.model_probe import entry_available
    from agent.core.model_control import is_disabled
    entries = config.model_entries or {}
    mode = getattr(getattr(config, "agent", None), "model_mode", "") or "any"
    allowed = MODE_TIERS.get(mode, MODE_TIERS["any"]) - {"local"}
    if not allowed:
        return None
    active = (config.llm.base_url, config.llm.model)
    active_entry = next(
        (e for e in entries.values()
         if (e.base_url, e.model or config.llm.model) == active),
        None,
    )
    # A LAN ("remote" location) endpoint is the operator's own hardware; its
    # payload never reached a third-party cloud, so peer failover must not
    # push it to one. Only genuine cloud endpoints may fail over cloud→cloud.
    if active_entry is None or location_tier(active_entry) != "cloud":
        return None
    active_tier = entry_tier(active_entry)

    def _candidates():
        # Same tier as the failing entry first — a free-tier turn stays free.
        if active_tier in allowed:
            yield from ((n, e) for n, e in entries.items()
                        if entry_tier(e) == active_tier)
        yield from ((n, e) for n, e in entries.items()
                    if entry_tier(e) in allowed and entry_tier(e) != active_tier)

    for name, e in _candidates():
        if is_disabled(config, name):
            continue
        if (e.base_url, e.model or config.llm.model) == active:
            continue
        if not entry_available(e):
            continue
        client = switch_to_entry(config, name)
        if client is not None:
            logger.warning("failover: cloud endpoint failing — switched to peer entry '%s' (%s)",
                           name, entry_tier(e))
            return client
    return None


# ── Per-turn privacy routing ──────────────────────────────────────────────────

def _has_secret_text(text: str, config: "Config") -> bool:
    from agent.security.redaction import redact
    return bool(text) and redact(text, config) != text


def payload_has_secret(config: "Config", api_messages: list[dict]) -> bool:
    """True if any message content carries a secret/credential shape."""
    for m in api_messages:
        c = m.get("content")
        if isinstance(c, str):
            if _has_secret_text(c, config):
                return True
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and _has_secret_text(part.get("text", ""), config):
                    return True
    return False


def redact_api_messages(config: "Config", api_messages: list[dict]) -> tuple[list[dict], int]:
    """Return a redacted copy of *api_messages* (originals untouched) + the count
    of messages that had something masked."""
    from agent.security.redaction import redact
    out: list[dict] = []
    n = 0
    for m in api_messages:
        c = m.get("content")
        if isinstance(c, str):
            r = redact(c, config)
            if r != c:
                n += 1
                m = {**m, "content": r}
        elif isinstance(c, list):
            changed = False
            new_parts = []
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    r = redact(part["text"], config)
                    if r != part["text"]:
                        changed = True
                        part = {**part, "text": r}
                new_parts.append(part)
            if changed:
                n += 1
                m = {**m, "content": new_parts}
        out.append(m)
    return out, n


def route_privacy(config: "Config", api_messages: list[dict]) -> dict:
    """Decide how an outbound payload should be handled given the privacy policy.

    Returns a dict with ``action`` one of:
      "send"   — payload safe / policy off; ``messages`` carries the (possibly
                 unchanged) payload to send.
      "redact" — ``messages`` is the masked wire copy; ``n`` masked messages.
      "switch" — route to local instead; ``entry`` is the local entry name.
      "block"  — refuse; ``reason`` explains why.
    """
    cfg = getattr(config, "privacy", None)
    if cfg is None or not cfg.enabled:
        return {"action": "send", "messages": api_messages}
    # Local endpoint: nothing leaves the machine, so policy does not apply.
    if not is_remote_endpoint(config):
        return {"action": "send", "messages": api_messages}
    if not payload_has_secret(config, api_messages):
        return {"action": "send", "messages": api_messages}

    strat = (getattr(cfg, "strategy", "redact") or "redact").lower()
    if strat == "force-local":
        name = resolve_local_entry(config, getattr(cfg, "local_entry", ""))
        if name:
            return {"action": "switch", "entry": name}
        return {"action": "block",
                "reason": "privacy: secret in payload but no local model configured to route to"}
    if strat == "block":
        return {"action": "block",
                "reason": "privacy: refused — outbound payload to a remote endpoint contains a secret/credential"}
    # default: redact then send remote
    masked, n = redact_api_messages(config, api_messages)
    return {"action": "redact", "messages": masked, "n": n}
