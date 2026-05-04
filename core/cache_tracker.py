"""Track prompt cache state per model endpoint.

Cache is per (base_url, model) — cloud provider prompt cache usually
lasts 5 min.  Warn when next request would miss cache.
"""
import time

_state: dict[str, float] = {}

def _key(base_url: str, model: str) -> str:
    return f"{base_url}||{model}"

def check_cache(base_url: str, model: str, cache_ttl: int) -> tuple[bool, int, str]:
    """Check cache state before LLM call.

    Returns (is_warm, seconds_remaining, message).
    cache_ttl=0 means tracking disabled.
    """
    k = _key(base_url, model)
    last = _state.get(k)

    if cache_ttl <= 0:
        return (False, 0, "")

    if last is None:
        return (False, 0, "cache: cold (no prior request this session)")

    elapsed = time.time() - last
    remaining = max(0, cache_ttl - int(elapsed))

    if remaining > 0:
        return (True, remaining, f"cache: warm ({remaining}s left)")
    else:
        expired = int(elapsed) - cache_ttl
        if expired < 60:
            return (False, 0, f"cache: expired {expired}s ago — next req uncached")
        return (False, 0, f"cache: expired {expired // 60}m ago — next req uncached")

def mark_request(base_url: str, model: str) -> None:
    """Record LLM request timestamp for cache tracking."""
    k = _key(base_url, model)
    _state[k] = time.time()

def clear_cache() -> None:
    """Reset all cache state (for testing)."""
    _state.clear()
