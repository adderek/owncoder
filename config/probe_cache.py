"""One answer per endpoint per startup, fetched in parallel.

Startup used to ask the same servers the same question three times. The profile
check probes every distinct endpoint (in parallel, ~3s), then
``check_reachability`` walks the default pool probing *per entry* rather than
per URL — eight entries on one dead LAN box meant eight three-second waits for
a fact already established — and then ``enrich_model_entries`` probes them all
again. On a machine with the LAN box switched off that was 49 of the 53 seconds
before the first prompt.

Every probe funnels through ``config.loader._probe_models``, so caching there
is enough: whoever asks first pays, everyone else reads the answer. Results are
keyed by URL alone — the pool's sixteen entries share five of them.

The cache is short-lived on purpose. A server that comes up (or goes down)
while the agent is running has to be noticed, and the answer is only worth
keeping for as long as the thing it describes is unlikely to have changed:
_HIT_TTL for a server that answered, _MISS_TTL for one that did not, since a
box being started is the change people wait for. Anything acting on a user's
explicit "try again" passes force=True.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

#: An endpoint that answered is unlikely to vanish mid-startup.
_HIT_TTL = 60.0
#: One that did not may be a server the user is starting right now.
_MISS_TTL = 20.0

_lock = threading.Lock()
#: url → (expires_at, result)
_entries: dict[str, tuple[float, dict | None]] = {}


def _key(url: str) -> str:
    """One key per endpoint. Callers differ on the trailing slash — the profile
    check strips it, the pool walk passes base_url as configured — and two
    spellings of one host would each pay their own timeout."""
    return (url or "").rstrip("/")


def _fresh(url: str) -> tuple[bool, dict | None]:
    with _lock:
        hit = _entries.get(_key(url))
    if hit is None or hit[0] < time.monotonic():
        return False, None
    return True, hit[1]


def _store(url: str, result: dict | None) -> None:
    ttl = _HIT_TTL if result is not None else _MISS_TTL
    with _lock:
        _entries[_key(url)] = (time.monotonic() + ttl, result)


def get(url: str, api_key: str, timeout: int,
        fetch: Callable[[str, str, int], dict | None],
        force: bool = False) -> dict | None:
    """Probe *url*, or return the answer someone already got.

    *fetch* does the real request; passing it in keeps this module free of the
    probing itself, so a caller that monkeypatches the prober still gets its
    own function called on a miss.
    """
    if not force:
        hit, result = _fresh(url)
        if hit:
            return result
    result = fetch(url, api_key, timeout)
    _store(url, result)
    return result


def prefetch(pairs: list[tuple[str, str]], timeout: int,
             fetch: Callable[[str, str, int], dict | None],
             max_workers: int = 8) -> None:
    """Probe every distinct URL in *pairs* at once, filling the cache.

    Called before the startup prompts: a dead LAN box costs one timeout for the
    whole run instead of one per pool entry that mentions it, and it is spent
    while someone is reading the profile report rather than after they answer.
    """
    urls: dict[str, str] = {}
    for url, api_key in pairs:
        u = _key(url)
        if u and u not in urls:
            urls[u] = api_key or ""
    todo = [(u, k) for u, k in urls.items() if not _fresh(u)[0]]
    if not todo:
        return
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max_workers, len(todo))) as ex:
        futs = {ex.submit(get, u, k, timeout, fetch): u for u, k in todo}
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception:
                logger.debug("probe prefetch failed for %s", futs[fut], exc_info=True)


def invalidate(url: str | None = None) -> None:
    """Forget one endpoint's answer, or all of them."""
    with _lock:
        if url is None:
            _entries.clear()
        else:
            _entries.pop(_key(url), None)


def stats() -> dict:
    """What the cache is holding — for diagnostics, not for decisions."""
    now = time.monotonic()
    with _lock:
        items = list(_entries.items())
    return {"entries": len(items),
            "reachable": sum(1 for _u, (exp, r) in items if r is not None and exp >= now),
            "unreachable": sum(1 for _u, (exp, r) in items if r is None and exp >= now)}
