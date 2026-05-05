"""Lightweight model-request status tracker.

Tracks how many concurrent calls are active per role (main, summarizer, embed, …).
Thread-safe via a lock; safe to call from async contexts.
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager

_lock = threading.Lock()
_counts: dict[str, int] = {}
_listeners: list = []

# GPU concurrency semaphore — limits parallel calls to GPU models.
# Initialized by init_gpu_semaphore() during agent startup.
_gpu_sem: asyncio.Semaphore | None = None

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


def _inc(role: str) -> None:
    with _lock:
        _counts[role] = _counts.get(role, 0) + 1
    _notify(role)


def _dec(role: str) -> None:
    with _lock:
        _counts[role] = max(0, _counts.get(role, 0) - 1)
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
async def track_async(role: str):
    _inc(role)
    try:
        yield
    finally:
        _dec(role)


@contextmanager
def track_sync(role: str):
    _inc(role)
    try:
        yield
    finally:
        _dec(role)


# ── GPU concurrency semaphore ────────────────────────────────────────────────

def init_gpu_semaphore(slots: int) -> None:
    """Initialise (or re-init) the GPU request semaphore with *slots* permits."""
    global _gpu_sem
    _gpu_sem = asyncio.Semaphore(slots)


def get_gpu_semaphore() -> asyncio.Semaphore | None:
    """Return the GPU semaphore, or None if not initialised (no GPU pool)."""
    return _gpu_sem


@asynccontextmanager
async def gpu_slot():
    """Acquire a GPU request slot, blocking until one is free. No-op if no
    GPU semaphore is configured (pure CPU setup)."""
    sem = _gpu_sem
    if sem is None:
        yield
        return
    async with sem:
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
