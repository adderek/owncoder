"""Integration tests for multi-project isolation (MULTI_PROJECT_PLAN s8).

Verifies the BLOCKER from §3: project processes must not leak state between
each other (working dir, sessions, path grants, security root).

Tests use temporary directories as project workdirs and a real ProjectRegistry.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from agent.ui_server.registry import ProjectRegistry, project_id, load_host_id
from agent.ui_server.auth import validate_origin_host, AuthState


class TestProjectIsolation:
    """Verify two projects do not interfere with each other."""

    def test_distinct_project_ids(self, tmp_path):
        """Different workdirs produce different project_ids."""
        a = tmp_path / "project-a"
        b = tmp_path / "project-b"
        a.mkdir()
        b.mkdir()

        host_id = load_host_id()
        id_a = project_id(str(a), host_id)
        id_b = project_id(str(b), host_id)

        assert id_a != id_b, f"project ids must differ: {id_a} == {id_b}"
        assert len(id_a) == 16
        assert len(id_b) == 16

    def test_same_workdir_same_id(self, tmp_path):
        """Same canonical workdir yields same project_id (stable)."""
        d = tmp_path / "project"
        d.mkdir()
        host_id = load_host_id()
        id1 = project_id(str(d), host_id)
        id2 = project_id(str(d), host_id)
        assert id1 == id2

    def test_registry_local_isolation(self, tmp_path):
        """Registering project B does not leak into project A's record."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir(); b.mkdir()

        reg = ProjectRegistry(whitelist=[str(tmp_path)])
        rec_a = reg.register_local(str(a), pid=100, port=8180)
        rec_b = reg.register_local(str(b), pid=200, port=8181)

        assert rec_a is not None
        assert rec_b is not None
        assert rec_a.project_id != rec_b.project_id
        assert len(reg.local_projects()) == 2

        # Each project's record is independent
        assert reg.get(rec_a.project_id).port == 8180
        assert reg.get(rec_b.project_id).port == 8181


class TestAuthGuards:
    """Auth checks mandated by §7."""

    def test_origin_validation_blocks_dns_rebinding(self):
        """Foreign Origin header is rejected."""
        from unittest.mock import MagicMock
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180", "Origin": "https://evil.com"}
        assert validate_origin_host(h) is False

    def test_origin_validation_allows_localhost(self):
        from unittest.mock import MagicMock
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180", "Origin": "http://localhost:8180"}
        assert validate_origin_host(h) is True

    def test_project_secret_blocks_unproxied(self):
        """Without X-Project-Secret, requests are rejected when secret is set."""
        state = AuthState(project_secret="s3cret")
        from unittest.mock import MagicMock
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180"}
        h.command = "POST"
        h.path = "/api/chat"
        assert state.validate_request(h) is False

    def test_project_secret_allows_proxied(self):
        """With X-Project-Secret, requests pass."""
        state = AuthState(project_secret="s3cret")
        from unittest.mock import MagicMock
        h = MagicMock()
        h.headers = {"Host": "127.0.0.1:8180", "X-Project-Secret": "s3cret"}
        h.command = "POST"
        h.path = "/api/chat"
        assert state.validate_request(h) is True


class TestBackwardCompat:
    """No regression for single-project setups."""

    def test_registry_empty_is_harmless(self):
        """An empty registry has zero projects and does not crash."""
        reg = ProjectRegistry()
        assert len(reg.projects()) == 0
        assert reg.get("nonexistent") is None

    def test_non_whitelisted_dir_rejected(self):
        """A workdir outside the whitelist is refused."""
        reg = ProjectRegistry(whitelist=["/allowed"])
        with tempfile.TemporaryDirectory() as td:
            rec = reg.register_local(td)
            assert rec is None  # not in whitelist

    def test_whitelisted_dir_accepted(self, tmp_path):
        """A workdir inside the whitelist is registered."""
        reg = ProjectRegistry(whitelist=[str(tmp_path)])
        rec = reg.register_local(str(tmp_path / "sub"), pid=1, port=8180)
        assert rec is not None
        assert len(reg.local_projects()) == 1

    def test_auth_not_required_single_loopback(self):
        """Default: single project, loopback — no auth required."""
        state = AuthState()
        assert state.auth_required(1, "127.0.0.1") is False

    def test_auth_required_multi_project(self):
        """Multi-project always requires auth."""
        state = AuthState()
        assert state.auth_required(2, "127.0.0.1") is True
