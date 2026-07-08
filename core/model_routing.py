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
    """Name of a local-tier model entry to route to, or None if none configured."""
    from agent.core.model_control import is_disabled
    entries = config.model_entries or {}
    if preferred and preferred in entries and not is_disabled(config, preferred):
        return preferred
    from agent.config import entry_tier
    for name, e in entries.items():
        if entry_tier(e) == "local" and not is_disabled(config, name):
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
    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url=e.base_url, api_key=e.api_key)


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
