"""Shared token estimation utilities."""
from __future__ import annotations


def count_tokens_approx(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4
