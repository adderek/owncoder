"""Lightweight model-request status tracker.

Tracks how many concurrent calls are active per role (main, summarizer, embed, …).
Thread-safe via a lock; safe to call from async contexts.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager, contextmanager

_lock = threading.Lock()
_counts: dict[str, int] = {}
_listeners: list = []


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
