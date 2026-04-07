"""Shared token counting utilities."""
from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def _get_encoder():
    """Load tiktoken encoder once; fall back to char-based estimate on import error."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens_approx(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base). Falls back to len/4 if unavailable."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return len(text) // 4
