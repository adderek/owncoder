"""Rate-limit-aware failover for one-shot background/subagent LLM calls.

``core/turn.py`` already does this for the main turn: detect a 429 fast (no
backoff for daily-quota exhaustion — burst limits get a short wait, daily caps
go straight to failover), put the (base_url, model) pair on cooldown, and
switch to another live entry. The many single-shot callers outside the main
turn (security triage/verify/evolve, summarizers, commit, …) historically
built a bare ``AsyncOpenAI()`` client and made one call with no retry and no
failover at all — a 429 or a dead endpoint just failed that subsystem outright
("(triage call failed: ...)"), even when a working alternative was configured.

``call_role_with_failover`` gives those call sites the same fast-detect/switch
behavior in one function: try the role's preferred entry, and on a rate-limit
or connection/server error, cooldown-mark it and move to the next mode-allowed
candidate — no blocking wait, since a background task should degrade fast
rather than sit in a backoff loop like a foreground turn can afford to.

Recovery probing (proactively re-testing a cooled-down endpoint instead of
just waiting out its cooldown) is not implemented here — cooldown expiry plus
the existing ``entry_available`` /models probe is the current retry-to-revive
path.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.config.models import ModelEntry

logger = logging.getLogger(__name__)


def role_candidates(config: "Config", role: str, max_candidates: int = 4,
                     local_only: bool = False) -> "list[tuple[str, ModelEntry]]":
    """Ordered [(name, entry), …] to try for *role*: the role's resolved entry
    first, then other enabled, mode-allowed, non-embedding entries — endpoints
    already on a rate-limit/failure cooldown sink to the back rather than
    being dropped outright (all-cooled-down is still better than zero tries).

    ``local_only`` restricts candidates to loopback/LAN endpoints — pass it
    when the caller is air-gap-aware, so failover can never hand an air-gapped
    caller a cloud entry the primary pick happened not to be.
    """
    from agent.config import make_registry
    from agent.config.registry import MODE_TIERS, entry_tier
    from agent.core.model_control import is_disabled
    from agent.config.model_probe import is_rate_limited, maybe_schedule_recovery_probe

    entries = getattr(config, "model_entries", None) or {}
    reg = make_registry(config)
    rev = {id(e): n for n, e in entries.items()}
    mode = getattr(getattr(config, "agent", None), "model_mode", "") or "any"
    allowed = MODE_TIERS.get(mode, MODE_TIERS["any"])

    def _is_local(e) -> bool:
        if not local_only:
            return True
        from agent.security.airgap import is_local_url
        return is_local_url(e.base_url)

    def _usable(name: str, e) -> bool:
        return (not is_disabled(config, name)
                and not getattr(e, "dimensions", 0) and _is_local(e))

    ordered: list[tuple[str, "ModelEntry"]] = []
    seen: set = set()
    try:
        primary = reg.role(role)
    except Exception:
        primary = None
    if primary is not None:
        # Entries outside config.model_entries (bare/minimal configs, e.g. in
        # tests) have no reverse-lookup name — fall back to the role name so
        # the primary pick is still usable rather than silently dropped.
        name = rev.get(id(primary), "") or role
        if _usable(name, primary):
            ordered.append((name, primary))
            seen.add((primary.base_url, primary.model))
    for name, e in entries.items():
        if not _usable(name, e):
            continue
        key = (e.base_url, e.model)
        if key in seen or entry_tier(e) not in allowed:
            continue
        seen.add(key)
        ordered.append((name, e))
    def _cooled_down(pair) -> bool:
        limited = is_rate_limited(pair[1].base_url, pair[1].model or "")
        if limited:
            # W5: opportunistically re-probe in the background so a recovered
            # endpoint doesn't sit unused for the rest of its cooldown window.
            maybe_schedule_recovery_probe(config, pair[1])
        return limited

    ordered.sort(key=_cooled_down)
    return ordered[:max_candidates]


def _build_kwargs(messages, max_tokens, temperature, extra_body, stream) -> dict:
    kwargs: dict = {"messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    if stream:
        kwargs["stream"] = True
    return kwargs


async def _walk_candidates(config, role, kwargs, metrics_role, max_candidates, local_only):
    """Shared candidate walk: yields (client, name, entry) that answered, or
    raises the last error once every candidate is exhausted. On any per-
    candidate failure the client for that attempt is closed and the entry is
    cooldown-marked before moving on; on success the client is handed to the
    caller, who owns closing it (needed alive for streaming callers).
    """
    from openai import (
        RateLimitError, APIConnectionError, APITimeoutError,
        InternalServerError, APIError,
    )
    from agent.core.llm_client import make_llm_client
    from agent.config.model_probe import (
        mark_rate_limited, retry_after_seconds, is_daily_quota_429,
    )

    candidates = role_candidates(config, role, max_candidates, local_only=local_only)
    if not candidates:
        raise RuntimeError(f"no usable model entry for role {role!r}")

    last_exc: Exception | None = None
    for name, entry in candidates:
        client = None
        try:
            client = make_llm_client(config, base_url=entry.base_url, api_key=entry.api_key)
            if metrics_role:
                try:
                    from agent.metrics import model_calls
                    model_calls.record_entry(entry, role=metrics_role)
                except Exception:
                    pass
            resp = await client.chat.completions.create(model=entry.model, **kwargs)
            return resp, name, entry, client
        except RateLimitError as e:
            retry_after = retry_after_seconds(e)
            daily = is_daily_quota_429(e, retry_after)
            cooldown = max(retry_after, 21600.0) if daily else 300.0
            try:
                mark_rate_limited(entry.base_url, entry.model, cooldown_s=cooldown)
            except Exception:
                logger.debug("llm_retry: mark_rate_limited failed", exc_info=True)
            logger.warning("llm_retry: %s rate limited on '%s' (%s) — trying next candidate",
                            role, name, "daily quota" if daily else "burst")
            last_exc = e
        except (APIConnectionError, APITimeoutError, InternalServerError, APIError) as e:
            try:
                mark_rate_limited(entry.base_url, entry.model)
            except Exception:
                logger.debug("llm_retry: mark_rate_limited failed", exc_info=True)
            logger.warning("llm_retry: %s failed on '%s' (%s: %s) — trying next candidate",
                            role, name, type(e).__name__, e)
            last_exc = e
        except Exception as e:  # noqa: BLE001 - client construction or any other failure
            logger.warning("llm_retry: %s failed on '%s' (%s: %s) — trying next candidate",
                            role, name, type(e).__name__, e)
            last_exc = e
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass
    raise last_exc


async def call_role_with_failover(
    config: "Config",
    role: str,
    *,
    messages: list,
    max_tokens: int | None = None,
    temperature: float | None = None,
    extra_body: dict | None = None,
    metrics_role: str = "",
    max_candidates: int = 4,
    local_only: bool = False,
):
    """One-shot ``chat.completions.create`` for *role*, trying candidate
    entries in order until one succeeds. Returns ``(response, entry_name,
    entry)``. Raises the last error once every candidate is exhausted.

    Each attempt uses ``core.llm_client.make_llm_client`` (fast connect/read
    timeout, SDK retries off) rather than a bare ``AsyncOpenAI()`` — the SDK's
    own default retries would otherwise silently re-hit a rejecting endpoint
    for its full timeout window before this even sees the 429.
    """
    kwargs = _build_kwargs(messages, max_tokens, temperature, extra_body, stream=False)
    resp, name, entry, client = await _walk_candidates(
        config, role, kwargs, metrics_role, max_candidates, local_only)
    try:
        return resp, name, entry
    finally:
        try:
            await client.close()
        except Exception:
            pass


async def open_stream_with_failover(
    config: "Config",
    role: str,
    *,
    messages: list,
    max_tokens: int | None = None,
    temperature: float | None = None,
    extra_body: dict | None = None,
    metrics_role: str = "",
    max_candidates: int = 4,
    local_only: bool = False,
):
    """Streaming counterpart of ``call_role_with_failover``.

    A 429/connection error raised at ``create()`` establishment time (before
    any chunk is read) fails over to the next candidate exactly like the
    one-shot path. Once a stream is handed back, its own connection stays
    open — the caller is responsible for consuming it and MUST close the
    returned client when done (mid-stream drops are not this helper's
    concern, same as the establishment-only guarantee documented for the
    module).

    Returns ``(stream, entry_name, entry, client)``.
    """
    kwargs = _build_kwargs(messages, max_tokens, temperature, extra_body, stream=True)
    return await _walk_candidates(
        config, role, kwargs, metrics_role, max_candidates, local_only)
