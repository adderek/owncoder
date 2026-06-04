"""Layer 1: Query gate — sanitize before network egress.

Secret detection, URL/IP validation (DNS rebind check), rate limiting,
Unicode normalization, audit logging. Deny-closed: any failure rejects.
"""
from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import os
import re
import socket
import time
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

_config = None

# ── Secret detection patterns ────────────────────────────────────────────
_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{32,}"),                    # OpenAI
    re.compile(r"sk-ant-[a-zA-Z0-9_-]{32,}"),               # Anthropic
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),                     # GitHub classic
    re.compile(r"github_pat_[a-zA-Z0-9_]{36,}"),            # GitHub fine-grained
    re.compile(r"Bearer\s+[a-zA-Z0-9._\-+=]{20,}"),         # Bearer tokens
    re.compile(r"-----BEGIN (?:RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----"),
    re.compile(r'(?:api[_-]?key|apikey|api_secret|secret_key)[\"\s:=]+[a-zA-Z0-9._\-]{16,}', re.I),
    re.compile(r"(?:mongodb|postgres|mysql|redis)://[^/\s]+:[^@\s]+@"),  # DB connection strings
    re.compile(r"eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}"),  # JWT
]

# ── Blocked IP ranges ───────────────────────────────────────────────────
_BLOCKED_NETWORKS = [
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("fc00::/7"),
    ipaddress.IPv6Network("fe80::/10"),
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1)
]

_BLOCKED_SCHEMES = {"file", "ftp", "gopher", "dict", "data", "javascript", "vbscript"}

# ── Rate limiting ───────────────────────────────────────────────────────
_search_count = 0
_fetch_count = 0
_last_call_ts = 0.0
_COOLDOWN_S = 1.0

# ── Unicode ──────────────────────────────────────────────────────────────
_ZERO_WIDTH = re.compile("[​-‏‪-‮⁠-⁤﻿]")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def setup(config) -> None:
    global _config
    _config = config


def _audit_log_path(session_id: str = "") -> Path:
    agent_dir = Path(_config.tools.agent_dir if _config else ".agent")
    audit_dir = agent_dir / "audit" / "search"
    audit_dir.mkdir(parents=True, exist_ok=True)
    return audit_dir / f"{session_id or 'default'}.jsonl"


def _audit(entry: dict, session_id: str = "") -> None:
    try:
        path = _audit_log_path(session_id)
        entry["ts"] = time.time()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


def _check_secrets(query: str) -> str | None:
    """Return the matched secret pattern description, or None if clean."""
    for pat in _SECRET_PATTERNS:
        m = pat.search(query)
        if m:
            return f"Secret pattern detected: {m.group()[:30]}..."
    return None


def _is_ip_safe(ip_str: str) -> bool:
    """Check raw IP against blocklist."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    for net in _BLOCKED_NETWORKS:
        if addr in net:
            return False
    return True


def _validate_url(url: str) -> str | None:
    """Validate URL safety. Returns error string or None if safe."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        return f"Invalid URL: {url[:100]}"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return f"Blocked URL scheme: {scheme}"

    hostname = parsed.hostname
    if not hostname:
        return f"URL has no hostname: {url[:100]}"

    # Check for raw IP in hostname
    try:
        ip = ipaddress.ip_address(hostname)
        if not _is_ip_safe(str(ip)):
            return f"Blocked IP address: {hostname}"
    except ValueError:
        pass  # Not an IP, proceed to DNS check

    # DNS rebind check: resolve hostname, verify resolved IPs
    _DNS_TIMEOUT_S = 5.0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(
                socket.getaddrinfo, hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
            )
            resolved = _fut.result(timeout=_DNS_TIMEOUT_S)
        for family, _, _, _, sockaddr in resolved:
            ip = sockaddr[0]
            if not _is_ip_safe(ip):
                return f"DNS resolved to blocked IP: {hostname} → {ip}"
    except concurrent.futures.TimeoutError:
        return f"DNS resolution timed out for: {hostname}"
    except socket.gaierror:
        return f"DNS resolution failed for: {hostname}"
    except Exception as e:
        return f"DNS check error for {hostname}: {e}"

    return None


def _sanitize_unicode(text: str) -> str:
    """NFC normalize, strip zero-width chars and control chars (keep \n \r \t)."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    return text


def _check_rate_limit(is_fetch: bool = False) -> str | None:
    """Return error if rate limit exceeded, None if allowed."""
    global _search_count, _fetch_count, _last_call_ts
    max_search = _config.web_search.max_search_calls_per_turn if _config else 3
    max_fetch = _config.web_search.max_fetch_calls_per_turn if _config else 5

    now = time.monotonic()
    elapsed = now - _last_call_ts
    if elapsed < _COOLDOWN_S:
        return f"Rate limit: minimum {_COOLDOWN_S}s between calls ({elapsed:.1f}s elapsed)"

    if is_fetch:
        _fetch_count += 1
        if _fetch_count > max_fetch:
            return f"Rate limit: max {max_fetch} fetch calls per turn"
    else:
        _search_count += 1
        if _search_count > max_search:
            return f"Rate limit: max {max_search} search calls per turn"

    _last_call_ts = now
    return None


def reset_rate_limits() -> None:
    """Reset per-turn counters (call at start of each agent turn)."""
    global _search_count, _fetch_count, _last_call_ts
    _search_count = 0
    _fetch_count = 0
    _last_call_ts = 0.0


def gate_query(query: str, session_id: str = "") -> str | dict:
    """Sanitize a search query. Returns cleaned query or error dict."""
    if _config and not _config.web_search.enabled:
        return {"error": "Web search is disabled (config.web_search.enabled = false)"}

    # Rate limit
    rl_err = _check_rate_limit(is_fetch=False)
    if rl_err:
        _audit({"event": "search.rate_limited", "query": query}, session_id)
        return {"error": rl_err}

    # Sanitize unicode first
    query = _sanitize_unicode(query)

    # Secret check
    secret = _check_secrets(query)
    if secret:
        _audit({"event": "search.secret_blocked", "query_hash": _query_hash(query)}, session_id)
        return {"error": f"Query rejected: {secret}"}

    _audit({"event": "search.query", "query_hash": _query_hash(query)}, session_id)
    return query


def gate_fetch(url: str, session_id: str = "") -> str | dict:
    """Validate and sanitize a fetch URL. Returns cleaned URL or error dict."""
    if _config and not _config.web_search.enabled:
        return {"error": "Web search is disabled (config.web_search.enabled = false)"}

    # Rate limit
    rl_err = _check_rate_limit(is_fetch=True)
    if rl_err:
        _audit({"event": "fetch.rate_limited", "url": url}, session_id)
        return {"error": rl_err}

    # URL validation
    url_err = _validate_url(url)
    if url_err:
        _audit({"event": "fetch.url_blocked", "url": url, "reason": url_err}, session_id)
        return {"error": url_err}

    # Sanitize unicode in URL
    url = _sanitize_unicode(url)

    _audit({"event": "fetch.query", "url": url}, session_id)
    return url


def _query_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
