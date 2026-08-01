"""HTTP auth helpers for the multi-project router and project processes (s5).

Origin/Host validation blocks DNS-rebinding (foreign domain → 127.0.0.1).
Session tokens protect against unauthorised cross-project access.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from http.server import BaseHTTPRequestHandler


def validate_origin_host(handler: BaseHTTPRequestHandler) -> bool:
    """Reject if Origin or Host header points to a non-loopback / foreign host.

    Called on EVERY request, even single-project on loopback.
    Blocks DNS-rebinding: a foreign domain resolving to 127.0.0.1 is rejected
    because the browser's Origin header reflects the foreign domain.
    """
    host = handler.headers.get("Host", "")
    origin = handler.headers.get("Origin", "")

    # Loopback or localhost are always allowed.
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}

    if host:
        host_clean = host.rsplit(":", 1)[0]  # strip port
        if host_clean not in allowed_hosts and not host_clean.startswith("127."):
            return False

    if origin:
        # Origin is a full URL: https://example.com:8080
        # Extract the hostname portion.
        try:
            from urllib.parse import urlparse
            parsed = urlparse(origin)
            origin_host = parsed.hostname or ""
        except Exception:
            origin_host = ""
        if origin_host and origin_host not in allowed_hosts and not origin_host.startswith("127."):
            return False

    return True


# ── Session token ────────────────────────────────────────────────────────────

# Cookie name for the double-submit token.
_TOKEN_COOKIE = "owncoder_auth"
# Header name for the double-submit token (CSRF protection).
_TOKEN_HEADER = "X-Owncoder-Auth"
# Token TTL (seconds) — regenerated on login.
_TOKEN_TTL = 86400 * 7  # 7 days


def generate_auth_token(secret: bytes | None = None) -> str:
    """Generate a random session token (URL-safe, 43 chars)."""
    return secrets.token_urlsafe(32)


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


class AuthState:
    """Per-process auth state: token + secret for router↔project trust."""

    def __init__(self, *, token: str = "", project_secret: str = ""):
        self.token = token or generate_auth_token()
        self.project_secret = project_secret or secrets.token_urlsafe(32)
        self._hash = hashlib.sha256(self.token.encode()).hexdigest()

    def validate_token(self, candidate: str) -> bool:
        return constant_time_compare(
            hashlib.sha256(candidate.encode()).hexdigest(), self._hash)

    def validate_request(self, handler: BaseHTTPRequestHandler) -> bool:
        """Full validation for a project process: Origin/Host + token or secret.

        Returns True if the request is authorised.
        """
        if not validate_origin_host(handler):
            return False

        # Project secret (router↔project, §7.1) — set via X-Project-Secret header.
        if self.project_secret:
            given = handler.headers.get("X-Project-Secret", "")
            if constant_time_compare(given, self.project_secret):
                return True

        # Session token — cookie double-submit.
        cookie_token = _extract_cookie(handler, _TOKEN_COOKIE)
        header_token = handler.headers.get(_TOKEN_HEADER, "")

        if cookie_token and header_token and cookie_token == header_token:
            return self.validate_token(cookie_token)

        # Query-string token — SSE only (GET /api/events), with TTL check.
        if handler.command == "GET" and handler.path.startswith("/api/events"):
            qs_token = _extract_query_param(handler.path, "token")
            if qs_token and self.validate_token(qs_token):
                return True

        return False

    def auth_required(self, project_count: int, bind_host: str) -> bool:
        """Whether auth is required given the current state."""
        if project_count > 1:
            return True
        if bind_host not in ("127.0.0.1", "localhost", "::1"):
            return True
        return False

    def set_auth_cookie(self, handler: BaseHTTPRequestHandler) -> None:
        """Set the double-submit cookie and return the header token for the client."""
        handler.send_header(
            "Set-Cookie",
            f"{_TOKEN_COOKIE}={self.token}; "
            "Path=/; HttpOnly; SameSite=Strict; Max-Age=604800",
        )
        handler.send_header(_TOKEN_HEADER, self.token)


def _extract_cookie(handler: BaseHTTPRequestHandler, name: str) -> str:
    cookie_header = handler.headers.get("Cookie", "")
    if not cookie_header:
        return ""
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith(name + "="):
            return part[len(name) + 1:]
    return ""


def _extract_query_param(path: str, name: str) -> str:
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(path).query)
    return (qs.get(name) or [""])[0]
