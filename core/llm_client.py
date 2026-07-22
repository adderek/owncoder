"""Single factory for AsyncOpenAI clients bound to LLM endpoints.

Every client that serves a turn MUST come from here. A client built with bare
``AsyncOpenAI(base_url=..., api_key=...)`` inherits the SDK defaults — 600s
request timeout and 2 silent retries — so a wedged backend (accepts the
connection, answers /health, never serves inference) hangs a turn for 30
minutes before the failover logic in run_turn ever sees an error. The
mid-turn switch paths (failover, auto-tier) used to build such bare clients,
which is exactly how a stuck session looked "working" in the UI forever.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config


def build_client_timeout(config: "Config"):
    """httpx.Timeout for the LLM client, or None to use the SDK default.

    A short connect timeout fails fast on a dead endpoint; the long read timeout
    is the hard ceiling for a single request (the per-chunk stall watchdog in
    streaming catches mid-stream stalls well before this fires)."""
    secs = int(getattr(config.llm, "request_timeout", 0) or 0)
    if secs <= 0:
        return None
    try:
        import httpx
    except Exception:
        return None
    return httpx.Timeout(connect=10.0, read=float(secs), write=30.0, pool=10.0)


def make_llm_client(config: "Config", base_url: str = "", api_key: str = ""):
    """AsyncOpenAI client with the request-timeout ceiling and SDK retries off.

    max_retries=0 — run_turn owns retry/failover; SDK retries would silently
    re-hit a wedged endpoint for another full timeout window each time.
    Defaults to the active ``config.llm`` endpoint when url/key not given.
    """
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=base_url or config.llm.base_url,
        api_key=api_key or config.llm.api_key,
        timeout=build_client_timeout(config),
        max_retries=0,
    )
