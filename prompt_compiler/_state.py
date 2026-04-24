"""Shared mutable state for the prompt-compiler module."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Entry:
    """One row in the compiled-prompts index, keyed by cache_key."""
    name: str
    model: str
    api_base: str
    original_sha: str
    status: str = "pending"          # pending|compiled|suspect|disabled
    disabled_reason: str = ""        # "compile_failed" | "no_savings" | "high_error_rate"
    attempts: int = 0
    calls: int = 0
    errors: int = 0
    original_chars: int = 0
    compiled_chars: int = 0
    original_tokens: int = 0
    compiled_tokens: int = 0
    tokens_saved_total: int = 0      # cumulative across every served compiled load
    created_at: str = ""
    last_call: str = ""
    last_error_at: str = ""

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    @property
    def savings_ratio(self) -> float:
        if not self.original_tokens:
            return 0.0
        return 1.0 - (self.compiled_tokens / self.original_tokens)


# In-memory state — guarded by _lock for thread-safety.
_lock = threading.Lock()
_index: dict[str, _Entry] | None = None      # cache_key -> _Entry
_index_path: Path | None = None
_in_flight: set[str] = set()                  # cache_keys currently compiling
_active: dict[str, str] = {}                  # name -> cache_key in use this session


def reset_state_for_tests() -> None:
    """Drop in-memory caches; tests use this to switch agent_dir between cases."""
    global _index, _index_path, _in_flight, _active
    with _lock:
        _index = None
        _index_path = None
        _in_flight = set()
        _active = {}
