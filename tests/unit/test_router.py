"""Unit tests for agent/ui_server/router.py (MULTI_PROJECT_PLAN s4)."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from agent.ui_server.registry import ProjectRegistry, project_id
from agent.ui_server.router import (
    _RouterHandler,
    _write_project_pidfile,
)


class TestProjectIdFromPath:
    def test_extracts_project_id(self):
        assert _RouterHandler._project_id_from_path(
            "/api/chat?project=abc123&other=1") == "abc123"

    def test_no_project_returns_empty(self):
        assert _RouterHandler._project_id_from_path("/api/chat") == ""

    def test_ignores_fragment(self):
        assert _RouterHandler._project_id_from_path(
            "/api/sessions?q=test&project=deadbeef") == "deadbeef"


class TestWriteProjectPidfile:
    def test_creates_file_with_port_and_workdir(self, tmp_path):
        workdir = str(tmp_path / "my-project")
        os.makedirs(workdir, exist_ok=True)
        pidfile_dir = str(tmp_path / "run")

        path = _write_project_pidfile(workdir, port=8199, pidfile_dir=pidfile_dir)
        assert os.path.exists(path)

        data = json.loads(Path(path).read_text())
        assert data["port"] == 8199
        assert data["pid"] == os.getpid()
        assert os.path.realpath(data["workdir"]) == os.path.realpath(workdir)

    def test_creates_secret_when_provided(self, tmp_path):
        workdir = str(tmp_path / "my-project")
        os.makedirs(workdir, exist_ok=True)

        path = _write_project_pidfile(workdir, port=8190, secret="s3cret",
                                      pidfile_dir=str(tmp_path / "run"))
        data = json.loads(Path(path).read_text())
        assert data["secret"] == "s3cret"

    def test_0600_permissions(self, tmp_path):
        workdir = str(tmp_path / "my-project")
        os.makedirs(workdir, exist_ok=True)

        path = _write_project_pidfile(workdir, port=8181,
                                      pidfile_dir=str(tmp_path / "run"))
        st = os.stat(path)
        assert st.st_mode & 0o777 == 0o600


class TestRegistryIntegration:
    def test_router_resolves_by_project_id(self):
        registry = ProjectRegistry()
        rec = registry.register_local("/tmp/test-proj", pid=12345, port=8180)
        assert rec is not None

        # _project_id_from_path is a staticmethod — callable without an instance.
        pid = _RouterHandler._project_id_from_path("/api/chat?project=" + rec.project_id)
        assert pid == rec.project_id


class TestSubmitReview:
    def test_remote_project_refused_no_http(self, monkeypatch):
        """Remote/unreachable projects must be refused without any HTTP call."""
        registry = ProjectRegistry()
        registry.apply_presence({"peers": {
            "peer-1": {"project_id": "rp1", "label": "remote-proj", "host": "otherhost"}
        }})
        rec = registry.get("rp1")
        assert rec is not None and rec.host != "local"

        handler = _RouterHandler.__new__(_RouterHandler)
        handler.registry = registry
        handler.project_secret = ""
        out = {}
        handler._json = lambda obj, code=200: out.setdefault("json", obj) or obj

        called = {"n": 0}

        def _no_call(*a, **k):
            called["n"] += 1
            raise AssertionError("must not call urlopen for remote project")

        import agent.ui_server.router as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", _no_call)

        handler._submit_review(rec.project_id)
        assert "error" in out["json"]
        assert called["n"] == 0

    def test_local_project_submits_clean_chat_url(self, monkeypatch):
        """Local project: prompt must POST to /api/chat WITHOUT ?project= query."""
        registry = ProjectRegistry()
        rec = registry.register_local("/tmp/test-proj", pid=12345, port=8180)
        handler = _RouterHandler.__new__(_RouterHandler)
        handler.registry = registry
        handler.project_secret = "s3cret"

        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            # urllib normalizes header names via str.capitalize() on add_header.
            seen["secret"] = next(
                (v for k, v in req.headers.items()
                 if k.lower() == "x-project-secret"), None)
            seen["body"] = req.data

            class _Resp:
                status = 200

                def read(self):
                    return b'{"ok": true, "injected": false}'

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

            return _Resp()

        import agent.ui_server.router as mod
        monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

        out = {}
        handler._json = lambda obj, code=200: out.setdefault("json", obj) or obj
        handler._submit_review(rec.project_id)

        assert "?project=" not in seen["url"]
        assert seen["url"].startswith(f"http://127.0.0.1:{rec.port}/api/chat")
        assert seen["secret"] == "s3cret"
        body = json.loads(seen["body"])
        assert "MULTI_PROJECT_PLAN" in body["text"]
        assert out["json"]["ok"] is True


# Minimal import to make pytest happy about the Path reference above.
from pathlib import Path
