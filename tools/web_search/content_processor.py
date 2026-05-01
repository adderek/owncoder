"""Layer 3: Convert raw HTTP response to clean plain text.

Binary detection → charset handling → HTML→text extraction →
text normalization → size cap → integrity hash.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_binary(data: bytes) -> bool:
    """Reject if >5% null bytes or >30% non-printable bytes in first 512 bytes."""
    sample = data[:512]
    if len(sample) == 0:
        return False
    nulls = sample.count(0)
    non_printable = sum(1 for b in sample if b < 0x20 and b not in (0x09, 0x0A, 0x0D))
    return (nulls / len(sample) > 0.05) or (non_printable / len(sample) > 0.30)


def _detect_charset(data: bytes, content_type: str | None = None) -> str:
    """Detect charset from Content-Type → <meta charset> → chardet → UTF-8."""
    # Try Content-Type header
    if content_type:
        m = re.search(r"charset=([^\s;]+)", content_type, re.I)
        if m:
            return m.group(1).strip("\"'")

    # Try <meta charset> in first 4KB
    head = data[:4096].decode("ascii", errors="ignore")
    m = re.search(r'<meta[^>]+charset=["\']?([^"\';\s>]+)', head, re.I)
    if m:
        return m.group(1)

    # Try chardet
    try:
        import chardet
        detected = chardet.detect(data)
        if detected and detected.get("encoding") and detected.get("confidence", 0) > 0.5:
            return detected["encoding"]
    except ImportError:
        pass

    return "utf-8"


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|iframe|object|embed|svg|math)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&#?\w+;")
_WHITESPACE_RE = re.compile(r"\s{3,}")


def _strip_html(raw: str) -> str:
    """Strip <script>, <style>, all tags, decode entities, normalize whitespace."""
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub("\n\n", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    """Strip null bytes, replace invalid UTF-8, NFC normalize, strip C0 controls."""
    text = text.replace("\x00", "")
    text = unicodedata.normalize("NFC", text)
    # Keep \n \r \t; strip other C0 controls
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return text


_MAX_CHARS = 32_768


def process(
    raw_body: bytes,
    *,
    content_type: str | None = None,
    max_chars: int = _MAX_CHARS,
) -> dict:
    """Process raw HTTP response body into clean text.

    Returns dict with:
      - text: processed plain text
      - hash: SHA-256 of raw body
      - truncated: bool
      - content_type_original: str | None
      - charset_used: str
      - binary_rejected: bool
      - error: str | None
    """
    result: dict = {
        "hash": _sha256(raw_body),
        "content_type_original": content_type,
        "charset_used": "utf-8",
        "binary_rejected": False,
        "truncated": False,
        "error": None,
        "text": "",
    }

    if not raw_body:
        return result

    # Binary detection
    if _is_binary(raw_body):
        result["binary_rejected"] = True
        result["error"] = "Binary content rejected"
        return result

    # Charset detection
    charset = _detect_charset(raw_body, content_type)
    result["charset_used"] = charset

    # Decode
    try:
        text = raw_body.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw_body.decode("utf-8", errors="replace")
        result["charset_used"] = "utf-8"

    # HTML → text
    is_html = content_type and ("html" in content_type.lower() or "xml" in content_type.lower())
    if is_html or bool(_TAG_RE.search(text[:4096])):
        text = _strip_html(text)

    # Normalize
    text = _normalize_text(text)

    # Size cap
    if len(text) > max_chars:
        text = text[:max_chars]
        result["truncated"] = True

    result["text"] = text
    return result
