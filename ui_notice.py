"""Operator notices — warnings that must reach whoever is driving the agent.

Deep code (sandbox setup, context loading) has no idea which UI is running. It
used to print to stderr, which is right for the terminal UIs and wrong for HTTP
mode: the browser user never sees the terminal, so a "running without a sandbox"
warning lands where nobody is looking, and the terminal gets it twice (once from
the print, once from the logging stderr handler).

A UI registers a sink; anything without a sink falls back to stderr so CLI and
one-shot runs keep their current behaviour. Every notice is logged either way.
"""
from __future__ import annotations

import collections
import logging
import sys
import threading
from typing import Callable, Deque

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sinks: list[Callable[[str, bool], None]] = []
# Notices raised before a UI exists (startup: crash warnings, dirty files,
# index failures). Replayed to the first sink that attaches, bounded so a long
# headless run cannot grow it without limit.
_pending: Deque[tuple[str, bool]] = collections.deque(maxlen=50)


def register(cb: Callable[[str, bool], None]) -> None:
    """Register callable(text, is_error) — called from arbitrary threads."""
    with _lock:
        if cb not in _sinks:
            _sinks.append(cb)
        backlog = list(_pending)
        _pending.clear()
    for text, is_error in backlog:
        try:
            cb(text, is_error)
        except Exception:
            logger.debug("ui notice replay failed", exc_info=True)


def unregister(cb: Callable[[str, bool], None]) -> None:
    with _lock:
        if cb in _sinks:
            _sinks.remove(cb)


def emit(text: str, *, error: bool = False) -> None:
    """Show `text` to the operator, wherever they are, and log it."""
    logger.warning("%s", text)
    if _deliver(text, error):
        return
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


def record(text: str, *, error: bool = False) -> None:
    """Queue `text` for a UI that has not started yet.

    For notices the caller already showed on the terminal (startup banners in
    cli/chat.py print before any UI exists). A browser attaching later still
    gets to see why the last session ended badly; a terminal UI would only
    duplicate what is already on screen, so nothing is replayed there.
    """
    if _deliver(text, error):
        return
    with _lock:
        _pending.append((text, error))


def _deliver(text: str, error: bool) -> bool:
    with _lock:
        sinks = list(_sinks)
    delivered = False
    for cb in sinks:
        try:
            cb(text, error)
            delivered = True
        except Exception:
            logger.debug("ui notice sink failed", exc_info=True)
    return delivered
