"""Unit tests for agent/ui_server/auth.py (MULTI_PROJECT_PLAN s5)."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from agent.ui_server.auth import (
    AuthState,
    constant_time_compare,
    generate_auth_token,
    validate_origin_host,
)


class TestValidateOriginHost:
    def _handler(self, host="127.0.0.1:8180", origin=""):
        h = MagicMock()
        h.headers = {"Host": host}
        if origin:
            h.headers["Origin"] = origin
        return h

    def test_allows_loopback(self):
        assert validate_origin_host(self._handler("127.0.0.1:8180")) is True

    def test_allows_localhost(self):
        assert validate_origin_host(self._handler("localhost:8180")) is True

    def test_allows_ipv6_loopback(self):
        assert validate_origin_host(self._handler("::1:8180")) is True

    def test_rejects_foreign_host(self):
        assert validate_origin_host(self._handler("evil.com")) is False

    def test_rejects_foreign_origin(self):
        h = self._handler("127.0.0.1:8180", "https://evil.com")
        assert validate_origin_host(h) is False

    def test_allows_localhost_origin(self):
        h = self._handler("127.0.0.1:8180", "http://localhost:8180")
        assert validate_origin_host(h) is True

    def test_no_host_passes(self):
        h = self._handler("")
        assert validate_origin_host(h) is True


class TestConstantTimeCompare:
    def test_equal(self):
        assert constant_time_compare("abc", "abc") is True

    def test_not_equal(self):
        assert constant_time_compare("abc", "abd") is False

    def test_different_lengths(self):
        assert constant_time_compare("abc", "abcdef") is False


class TestAuthToken:
    def test_generated_token_is_urlsafe(self):
        tok = generate_auth_token()
        assert len(tok) >= 32
        assert "/" not in tok
        assert "+" not in tok

    def test_tokens_are_unique(self):
        assert generate_auth_token() != generate_auth_token()


class TestAuthState:
    def test_validate_token_roundtrip(self):
        state = AuthState(token="test-token-32-bytes")
        assert state.validate_token("test-token-32-bytes") is True
        assert state.validate_token("wrong-token") is False

    def test_project_secret_present_in_state(self):
        state = AuthState(project_secret="shared-secret")
        assert state.project_secret == "shared-secret"

    def test_auth_required_single_project_loopback(self):
        state = AuthState()
        assert state.auth_required(1, "127.0.0.1") is False

    def test_auth_required_multi_project(self):
        state = AuthState()
        assert state.auth_required(2, "127.0.0.1") is True

    def test_auth_required_non_loopback(self):
        state = AuthState()
        assert state.auth_required(1, "0.0.0.0") is True

    def test_validate_request_with_secret(self):
        state = AuthState(project_secret="s3cret")
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180", "X-Project-Secret": "s3cret"}
        h.command = "GET"
        h.path = "/api/chat"
        assert state.validate_request(h) is True

    def test_validate_request_without_secret_rejected(self):
        state = AuthState(project_secret="s3cret")
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180"}
        h.command = "GET"
        h.path = "/api/chat"
        assert state.validate_request(h) is False

    def test_validate_request_cookie_double_submit(self):
        state = AuthState(token="test-token")
        h = MagicMock()
        h.headers = {
            "Host": "127.0.0.1:8180",
            "Cookie": "owncoder_auth=test-token",
            "X-Owncoder-Auth": "test-token",
        }
        h.command = "POST"
        h.path = "/api/chat"
        assert state.validate_request(h) is True

    def test_validate_request_bad_origin_rejected(self):
        state = AuthState()
        h = MagicMock()
        h.headers = {"Host": "evil.com:80"}
        h.command = "GET"
        h.path = "/api/chat"
        assert state.validate_request(h) is False
