"""Thread-safe in-memory store for truncated tool outputs.

Stores full results keyed by call_id so agents can retrieve specific
ranges without re-executing the tool. FIFO eviction when max_bytes exceeded.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import OutputStoreConfig

_HEAD_CHARS = 2000
_TAIL_CHARS = 1000
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


class OutputStore:
    """Thread-safe store of full tool-call results for later retrieval."""

    def __init__(self, config: OutputStoreConfig | None = None, *,
                 max_bytes: int | None = None,
                 head_chars: int | None = None,
                 tail_chars: int | None = None) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, _Entry] = OrderedDict()
        self._total_bytes = 0
        if config:
            self._max_bytes = config.max_bytes
            self.head_chars = config.head_chars
            self.tail_chars = config.tail_chars
        else:
            self._max_bytes = max_bytes if max_bytes is not None else _MAX_BYTES
            self.head_chars = head_chars if head_chars is not None else _HEAD_CHARS
            self.tail_chars = tail_chars if tail_chars is not None else _TAIL_CHARS

    # -- public api -------------------------------------------------------

    def store(self, call_id: str, result: str) -> None:
        """Store full result under *call_id*. Evict oldest entries if over budget."""
        size = len(result.encode("utf-8"))
        with self._lock:
            self._store[call_id] = _Entry(
                result=result,
                bytes=size,
                lines=result.count("\n") + 1,
                timestamp=time.time(),
            )
            self._total_bytes += size
            self._evict()

    def get(self, call_id: str) -> str | None:
        """Return full stored result or None."""
        with self._lock:
            e = self._store.get(call_id)
            return e.result if e else None

    def get_range(self, call_id: str, start: int = 0, end: int | None = None) -> str | None:
        """Return character range from stored result."""
        with self._lock:
            e = self._store.get(call_id)
            if e is None:
                return None
            return e.result[start:end]

    def get_lines(self, call_id: str, start: int = 0, end: int | None = None) -> str | None:
        """Return line range (0-indexed, end exclusive) from stored result."""
        with self._lock:
            e = self._store.get(call_id)
            if e is None:
                return None
            lines = e.result.splitlines(keepends=True)
            return "".join(lines[start:end])

    def info(self, call_id: str) -> dict | None:
        """Return metadata for a stored entry or None."""
        with self._lock:
            e = self._store.get(call_id)
            if e is None:
                return None
            return {"bytes": e.bytes, "lines": e.lines, "chars": len(e.result)}

    # -- truncation helpers -----------------------------------------------

    def truncate(self, result: str) -> tuple[str, bool]:
        """Return (truncated_result, was_truncated).

        When *result* exceeds (head_chars + tail_chars) produce a compact
        head + tail envelope.  Otherwise return result unchanged.
        """
        total = len(result)
        head_n = self.head_chars
        tail_n = self.tail_chars
        if total <= head_n + tail_n:
            return result, False
        head = result[:head_n]
        tail = result[-tail_n:] if tail_n else ""
        truncated = (
            head
            + _TRUNC_GAP
            + tail
        )
        return truncated, True

    # -- internal ---------------------------------------------------------

    def _evict(self) -> None:
        """FIFO eviction until under *max_bytes*."""
        while self._total_bytes > self._max_bytes and self._store:
            _cid, e = self._store.popitem(last=False)  # oldest first
            self._total_bytes -= e.bytes


class _Entry:
    __slots__ = ("result", "bytes", "lines", "timestamp")

    def __init__(self, result: str, bytes: int, lines: int, timestamp: float) -> None:
        self.result = result
        self.bytes = bytes
        self.lines = lines
        self.timestamp = timestamp


# ── Module-level global (initialized from agent startup) ──────────────────
_instance: OutputStore | None = None


def init_store(config: OutputStoreConfig | None = None) -> OutputStore:
    """Initialize the global output store. Called once from agent startup."""
    global _instance
    _instance = OutputStore(config)
    return _instance


def get_store() -> OutputStore:
    """Return the global output store. Raises RuntimeError if not initialized."""
    if _instance is None:
        raise RuntimeError("OutputStore not initialized — call init_store() first")
    return _instance


# ── Separator inserted between head and tail ──────────────────────────────
_TRUNC_GAP = "\n\n...\n\n"
