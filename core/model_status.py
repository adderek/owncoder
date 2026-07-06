"""Lightweight model-request status tracker.

Tracks how many concurrent calls are active per role (main, summarizer, embed, …).
Thread-safe via a lock; safe to call from async contexts.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager, contextmanager

_lock = threading.Lock()
_counts: dict[str, int] = {}
# Concurrent calls in flight per endpoint label (e.g. "local", "groq", "cerebras").
# Lets the UI show how many local vs cloud requests overlap at one instant —
# the role counters (_counts) collapse that detail since one role can hit
# different endpoints over a session.
_endpoints: dict[str, int] = {}
# Concurrent calls in flight per model name (e.g. "qwen2.5-coder", "llama-3.3-70b").
# Distinct from endpoints/roles: lets the UI show exactly which models — and how
# many of each — are live at one instant, the detail the user wants during
# subagent / parallel web fan-out where several models run at once.
_models: dict[str, int] = {}
_listeners: list = []

# role → bool availability snapshot (configured model live on its endpoint).
# Populated by best-effort probes; a missing role means "unknown / not probed".
_availability: dict[str, bool] = {}

# GPU concurrency semaphore — limits parallel calls to GPU models.
# Initialized by init_gpu_semaphore() during agent startup.
_gpu_sem: asyncio.Semaphore | None = None
# Cross-process counting semaphore (flock-based) — bounds GPU calls across all
# agent processes sharing one endpoint. None = in-process limit only.
_gpu_xsem = None

# Per-worker detail records for parallel agent fan-out.
_worker_seq: int = 0
_workers: dict[int, dict] = {}  # worker_id → {model, task, started_at, finished_at, status, error}


def get_states() -> dict[str, str]:
    """Return role → 'idle' | 'running' snapshot."""
    with _lock:
        return {role: ("running" if n > 0 else "idle") for role, n in _counts.items()}


def get_counts() -> dict[str, int]:
    """Return role → active request count snapshot."""
    with _lock:
        return dict(_counts)


def get_endpoint_counts() -> dict[str, int]:
    """Return endpoint-label → active request count snapshot (only non-zero)."""
    with _lock:
        return {k: v for k, v in _endpoints.items() if v > 0}


def get_model_counts() -> dict[str, int]:
    """Return model-name → active request count snapshot (only non-zero)."""
    with _lock:
        return {k: v for k, v in _models.items() if v > 0}


def provider_label(base_url: str | None) -> str:
    """Short endpoint label for concurrency display.

    "local" for loopback / non-network schemes, "remote" for private LAN
    addresses; otherwise the distinctive part of the hostname
    (api.groq.com → "groq", llama.cerebras.ai → "cerebras").
    """
    try:
        from agent.security.airgap import is_local_url
        if is_local_url(base_url):
            return "local"
    except Exception:
        if not base_url:
            return "local"
    from urllib.parse import urlparse
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return "cloud"
    if not host:
        return "cloud"
    import ipaddress
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback:
            return "local"
        return "remote" if ip.is_private else "cloud"
    except ValueError:
        pass
    parts = [p for p in host.split(".") if p not in ("api", "www", "openai", "v1")]
    # Drop the TLD; keep the most specific remaining label.
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else "cloud"


def set_availability(role_map: dict[str, bool]) -> None:
    """Merge a role → available snapshot from a probe. Unlisted roles untouched."""
    with _lock:
        _availability.update(role_map)


def get_availability() -> dict[str, bool]:
    """Return role → bool availability snapshot (empty = not yet probed)."""
    with _lock:
        return dict(_availability)


def _inc(role: str, endpoint: str | None = None, model: str | None = None) -> None:
    with _lock:
        _counts[role] = _counts.get(role, 0) + 1
        if endpoint:
            _endpoints[endpoint] = _endpoints.get(endpoint, 0) + 1
        if model:
            _models[model] = _models.get(model, 0) + 1
    _notify(role)


def _dec(role: str, endpoint: str | None = None, model: str | None = None) -> None:
    with _lock:
        _counts[role] = max(0, _counts.get(role, 0) - 1)
        if endpoint:
            _endpoints[endpoint] = max(0, _endpoints.get(endpoint, 0) - 1)
        if model:
            _models[model] = max(0, _models.get(model, 0) - 1)
    _notify(role)


def _notify(role: str) -> None:
    for cb in list(_listeners):
        try:
            cb(role)
        except Exception:
            pass


def add_listener(cb) -> None:
    _listeners.append(cb)


def remove_listener(cb) -> None:
    try:
        _listeners.remove(cb)
    except ValueError:
        pass


@asynccontextmanager
async def track_async(role: str, endpoint: str | None = None, model: str | None = None):
    _inc(role, endpoint, model)
    try:
        yield
    finally:
        _dec(role, endpoint, model)


@contextmanager
def track_sync(role: str, endpoint: str | None = None, model: str | None = None):
    _inc(role, endpoint, model)
    try:
        yield
    finally:
        _dec(role, endpoint, model)


# ── GPU concurrency semaphore ────────────────────────────────────────────────

def init_gpu_semaphore(slots: int, lock_dir=None) -> None:
    """Initialise (or re-init) the GPU request semaphore with *slots* permits.

    If *lock_dir* is given, also create a cross-process flock semaphore there so
    the cap holds across multiple agent processes sharing one GPU endpoint.
    """
    global _gpu_sem, _gpu_xsem
    _gpu_sem = asyncio.Semaphore(slots)
    if lock_dir is not None:
        try:
            from agent.core.gpu_lock import CrossProcessSemaphore
            _gpu_xsem = CrossProcessSemaphore(slots, lock_dir)
        except Exception as exc:  # never let lock setup break startup
            logging.getLogger(__name__).warning(
                "gpu cross-process lock disabled (%s): %s", lock_dir, exc)
            _gpu_xsem = None
    else:
        _gpu_xsem = None


def get_gpu_semaphore() -> asyncio.Semaphore | None:
    """Return the GPU semaphore, or None if not initialised (no GPU pool)."""
    return _gpu_sem


@asynccontextmanager
async def gpu_slot():
    """Acquire a GPU request slot, blocking until one is free. No-op if no
    GPU semaphore is configured (pure CPU setup). When a cross-process lock is
    configured, the in-process permit is taken first (fast, fair) and then the
    global file-lock slot."""
    sem = _gpu_sem
    if sem is None:
        yield
        return
    async with sem:
        if _gpu_xsem is None:
            yield
        else:
            async with _gpu_xsem.acquire():
                yield


# ── Parallel worker registry ──────────────────────────────────────────────────

def register_worker(model: str, task: str, max_task_chars: int = 80) -> int:
    """Register a new parallel worker. Returns its worker_id."""
    global _worker_seq
    preview = task[:max_task_chars] + ("…" if len(task) > max_task_chars else "")
    with _lock:
        _worker_seq += 1
        wid = _worker_seq
        _workers[wid] = {
            "model": model,
            "task": preview,
            "started_at": time.monotonic(),
            "finished_at": None,
            "status": "running",
            "error": None,
        }
    return wid


def finish_worker(worker_id: int, *, error: str | None = None) -> None:
    """Mark worker as done (success or error)."""
    with _lock:
        w = _workers.get(worker_id)
        if w is not None:
            w["finished_at"] = time.monotonic()
            w["status"] = "error" if error else "done"
            w["error"] = error


def get_workers() -> list[dict]:
    """Return snapshot of all worker records, most-recent first."""
    with _lock:
        now = time.monotonic()
        out = []
        for wid, w in sorted(_workers.items(), reverse=True):
            end = w["finished_at"] or now
            out.append({
                "id": wid,
                "model": w["model"],
                "task": w["task"],
                "elapsed": round(end - w["started_at"], 1),
                "status": w["status"],
                "error": w["error"],
            })
        return out


def clear_finished_workers() -> None:
    """Remove completed/errored workers from the registry."""
    with _lock:
        done = [wid for wid, w in _workers.items() if w["status"] != "running"]
        for wid in done:
            del _workers[wid]
