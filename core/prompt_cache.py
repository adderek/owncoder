"""Prompt-cache support: breakpoints, cached-token telemetry, prefix stability.

Three separate concerns, all about not re-paying for the same prompt prefix on
every step of a turn:

1. **Breakpoints.** Some endpoints (Anthropic-compatible ones) only cache what
   the request explicitly marks with ``cache_control``. Others (OpenAI,
   DeepSeek, most vLLM/llama.cpp builds) cache prefixes automatically and reject
   or ignore unknown fields. So markers are opt-in per model entry — sending
   them blindly to an endpoint that validates its schema breaks the request.

2. **Telemetry.** Cache hits are invisible today: usage accounting only tracks
   ``prompt_tokens``. Providers report the cached portion under different names;
   this normalises them so the saving is measurable rather than assumed.

3. **Prefix stability.** A cache only helps if the front of the request is
   byte-identical between calls. That is a property nothing enforces — any
   future change that puts volatile text (a timestamp, a token count, a
   freshly-retrieved note) near the front silently destroys it. The prefix
   signature turns that regression into a log line instead of a bigger bill.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# Cached-prompt-token field names, by provider dialect. All mean "of the
# prompt_tokens you were billed for, this many came from cache".
_CACHED_FIELDS = ("cached_tokens", "cache_read_input_tokens")

_EPHEMERAL = {"type": "ephemeral"}

# Signature of the last request's stable prefix, per (base_url, model).
_prefix_sigs: dict[str, str] = {}


def _active_entry(config: "Config"):
    """The [models.<name>] entry currently bridged onto config.llm, if any.

    Matched on base_url + model rather than a stored name: mid-turn routing
    (auto-tier escalation, privacy force-local, failover) rewrites config.llm
    directly, so any cached name can be stale by the time a request is built.
    """
    entries = getattr(config, "model_entries", None) or {}
    base_url = getattr(config.llm, "base_url", "")
    model = getattr(config.llm, "model", "")
    for entry in entries.values():
        if getattr(entry, "base_url", "") == base_url and getattr(entry, "model", "") == model:
            return entry
    return None


def _style(config: "Config") -> str:
    """Breakpoint dialect for the active model entry: "off" or "anthropic"."""
    entry = _active_entry(config)
    value = getattr(entry, "cache_breakpoints", None) if entry is not None else None
    if not value:
        value = getattr(config.llm, "cache_breakpoints", "off")
    value = str(value or "off").strip().lower()
    return value if value in ("off", "anthropic") else "off"


def _mark(message: dict) -> dict:
    """Copy of *message* carrying an ephemeral cache breakpoint.

    Anthropic's dialect only accepts cache_control on a structured content
    block, so a plain string is promoted to a one-element text block. Content
    that is already structured gets the marker on its last block.
    """
    out = dict(message)
    content = out.get("content")
    if isinstance(content, str):
        out["content"] = [{"type": "text", "text": content, "cache_control": _EPHEMERAL}]
        return out
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        if isinstance(blocks[-1], dict):
            blocks[-1]["cache_control"] = _EPHEMERAL
            out["content"] = blocks
            return out
    # Nothing markable (tool_calls-only assistant message, empty content):
    # leave it alone rather than inventing a block the endpoint may reject.
    return out


def apply_breakpoints(api_messages: list[dict], config: "Config") -> list[dict]:
    """Mark cache breakpoints on the request, if the active entry wants them.

    Two markers, which is what the prefix actually looks like in this agent: the
    end of the system preamble (stable for a whole session) and the end of the
    conversation so far (stable for the rest of the turn, since each step only
    appends). Endpoints that cache automatically get the list back untouched.
    """
    if _style(config) != "anthropic" or not api_messages:
        return api_messages

    last_system = -1
    for i, m in enumerate(api_messages):
        if m.get("role") == "system":
            last_system = i
        else:
            break
    last = len(api_messages) - 1

    out = list(api_messages)
    for idx in {i for i in (last_system, last) if i >= 0}:
        out[idx] = _mark(out[idx])
    return out


def extract_cached_tokens(usage: object) -> int:
    """Cached prompt tokens reported by the endpoint, across provider dialects.

    Returns 0 when the endpoint says nothing — which is not the same as "no
    cache hit", so callers should treat 0 as "unknown or none" rather than
    proof of a miss.
    """
    if usage is None:
        return 0
    for field in _CACHED_FIELDS:
        value = getattr(usage, field, None)
        if isinstance(usage, dict):
            value = usage.get(field, value)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details", details)
    if details is not None:
        for field in _CACHED_FIELDS:
            value = getattr(details, field, None)
            if isinstance(details, dict):
                value = details.get(field, value)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return 0


def prefix_signature(api_messages: list[dict], depth: int = 1) -> str:
    """Hash of the first *depth* messages — the part a cache must see unchanged."""
    head = api_messages[:depth]
    try:
        blob = json.dumps(head, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = repr(head)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def check_prefix_stable(api_messages: list[dict], config: "Config") -> bool:
    """Warn when the cacheable prefix changed since the last request.

    Returns True when the prefix is unchanged (or this is the first request for
    the endpoint). A False here means every subsequent request re-pays for the
    whole prefix, so it is worth a log line even though nothing is broken.
    """
    if not api_messages:
        return True
    key = f"{getattr(config.llm, 'base_url', '')}||{getattr(config.llm, 'model', '')}"
    sig = prefix_signature(api_messages)
    previous = _prefix_sigs.get(key)
    _prefix_sigs[key] = sig
    if previous is None or previous == sig:
        return True
    logger.info(
        "prompt cache: system prefix changed (%s -> %s) — this request re-pays "
        "for the prefix", previous, sig,
    )
    return False


def cached_prefix_intact(api_messages: list[dict], config: "Config") -> bool:
    """Read-only counterpart to check_prefix_stable: is a live cache being kept?

    Unlike check_prefix_stable this records nothing, so it can be asked outside
    the request path. False when no prefix has been seen for this endpoint yet —
    a cache that does not exist cannot be lost.
    """
    if not api_messages:
        return False
    key = f"{getattr(config.llm, 'base_url', '')}||{getattr(config.llm, 'model', '')}"
    previous = _prefix_sigs.get(key)
    return previous is not None and previous == prefix_signature(api_messages)


#: How far over the compaction budget a deferral may run before compacting
#: anyway. The budget is itself well below the context ceiling, so a modest
#: overshoot is affordable; this is what stops a permanently-warm cache from
#: deferring compaction forever.
_DEFER_MAX_OVERSHOOT = 1.15

#: Fraction of the output reserve a deferral may borrow. Overshooting the
#: budget necessarily eats into the room set aside for the model's reply — the
#: budget is the window minus that reserve — so the relative overshoot above is
#: not a sufficient guard on its own: with a small max_output_tokens, 1.15x of
#: the budget lands past the end of the window. Keeping half the reserve
#: intact bounds it in terms of the thing actually at risk.
_DEFER_RESERVE_BORROW = 0.5


def defer_for_cache(config: "Config", messages: list[dict],
                    token_est: int, budget: int) -> tuple[bool, str]:
    """Should a due compaction wait for the prompt cache to expire first?

    On providers that bill cached input tokens at a fraction of the normal rate
    (DeepSeek, Anthropic, OpenAI), compaction is expensive twice over: it costs
    an LLM call, and it rewrites the message prefix, so the next request re-pays
    full price for the whole thing. If the cache is warm and still intact, the
    same compaction done after it expires costs nothing extra.

    Returns (defer, reason). Deferral requires all of:
      - llm.defer_compaction_for_cache is on (off by default: it trades context
        headroom for money, and only paid providers benefit),
      - cache tracking is on and the endpoint's cache is warm,
      - the prefix that cache holds is still intact,
      - the overshoot is small enough to be safe.
    """
    if not getattr(config.llm, "defer_compaction_for_cache", False):
        return False, ""
    ttl = int(getattr(config.llm, "cache_ttl", 0) or 0)
    if ttl <= 0:
        return False, ""
    # Overflow safety first: a deferral must never be the reason a turn blows
    # the context window, whatever the cache is worth.
    if budget <= 0 or token_est > budget * _DEFER_MAX_OVERSHOOT:
        return False, "overshoot too large"
    from agent.core.context_budget import effective_ctx_window, _PROMPT_OVERHEAD
    ctx = effective_ctx_window(config)
    max_out = int(getattr(config.llm, "max_output_tokens", 0) or 0)
    reserve = min(max_out, ctx // 2)
    if token_est > ctx - int(reserve * _DEFER_RESERVE_BORROW) - _PROMPT_OVERHEAD:
        return False, "would eat the output reserve"
    from agent.core.cache_tracker import check_cache
    warm, remaining, _msg = check_cache(
        getattr(config.llm, "base_url", ""), getattr(config.llm, "model", ""), ttl)
    if not warm:
        return False, "cache cold"
    # Normalisation is the expensive check, so it runs last and only once every
    # cheaper reason to say no has been ruled out. The recorded signature comes
    # from the normalised request, so the comparison has to use the same shape.
    from agent.core.turn_setup import normalize_api_messages
    if not cached_prefix_intact(normalize_api_messages(messages), config):
        return False, "prefix already changed"
    return True, f"cache warm {remaining}s"


def reset() -> None:
    """Forget recorded prefixes (tests, session switch)."""
    _prefix_sigs.clear()


def prepare(api_messages: list[dict], config: "Config") -> list[dict]:
    """Everything to do to a request just before it is sent."""
    try:
        check_prefix_stable(api_messages, config)
        return apply_breakpoints(api_messages, config)
    except Exception:
        # A caching optimisation must never be the reason a turn fails.
        logger.exception("prompt cache: prepare failed, sending request unchanged")
        return api_messages
