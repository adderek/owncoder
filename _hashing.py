"""Shared SHA-256 helpers.

Several modules independently defined the same one-line digests. Centralized
here so the encoding policy (utf-8, replace errors) lives in one place.
"""
from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
