"""SSRF redirect and DNS-rebind protection tests for the HTTP fetcher script.

Runs _http_fetcher.py as a plain subprocess (bypassing sandbox runner) against
a local mock HTTP server so tests are fast and deterministic.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


# ── Mock server ─────────────────────────────────────────────────────────────

class _RedirectHandler(BaseHTTPRequestHandler):
    """Configurable redirect/response server for tests."""

    # Map path → (status, headers, body)
    routes: dict[str, tuple[int, dict, bytes]] = {}

    def log_message(self, *args):
        pass  # suppress output

    def do_GET(self):
        route = self.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        status, headers, body = route
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    do_HEAD = do_GET


def _start_server(routes: dict) -> tuple[HTTPServer, int]:
    _RedirectHandler.routes = routes
    server = HTTPServer(("127.0.0.1", 0), _RedirectHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


# ── Fetcher helper ───────────────────────────────────────────────────────────

def _write_fetcher_script() -> Path:
    """Write (or reuse) the fetcher script from http_executor."""
    from agent.tools.web_search import http_executor
    tmp = Path(tempfile.mkdtemp()) / "_http_fetcher.py"
    tmp.write_text(http_executor._FETCHER_SCRIPT)
    return tmp


@pytest.fixture(scope="module")
def fetcher_script():
    return _write_fetcher_script()


def _run_fetcher(fetcher_script: Path, request: dict) -> dict:
    result = subprocess.run(
        [sys.executable, str(fetcher_script)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not result.stdout.strip():
        return {"error": f"No output. stderr: {result.stderr[:200]}"}
    return json.loads(result.stdout)


# ── Redirect SSRF tests ──────────────────────────────────────────────────────

class TestRedirectSSRF:
    def test_redirect_to_loopback_blocked(self, fetcher_script):
        """302 → http://127.0.0.1/ must be rejected."""
        server, port = _start_server({
            "/redirect": (302, {"Location": "http://127.0.0.1/"}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/redirect",
                "pinned_ip": "127.0.0.1",  # initial hop allowed (local test server)
                "max_redirects": 3,
            })
            # The redirect target is 127.0.0.1 (default port 80), which is blocked
            assert resp.get("error"), f"Expected error, got: {resp}"
            assert "Blocked" in resp["error"] or "blocked" in resp["error"]
        finally:
            server.shutdown()

    def test_redirect_to_link_local_blocked(self, fetcher_script):
        """302 → http://169.254.169.254/ (cloud metadata) must be rejected."""
        server, port = _start_server({
            "/redirect-meta": (302, {"Location": "http://169.254.169.254/latest/meta-data/"}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/redirect-meta",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 3,
            })
            assert resp.get("error"), f"Expected error, got: {resp}"
        finally:
            server.shutdown()

    def test_redirect_to_file_scheme_blocked(self, fetcher_script):
        """302 → file:///etc/passwd must be rejected (scheme check)."""
        server, port = _start_server({
            "/redirect-file": (302, {"Location": "file:///etc/passwd"}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/redirect-file",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 3,
            })
            assert resp.get("error"), f"Expected error, got: {resp}"
            assert "scheme" in resp["error"].lower()
        finally:
            server.shutdown()

    def test_max_redirects_enforced(self, fetcher_script):
        """Redirect loop must stop after max_redirects hops."""
        # /loop → /loop → /loop ... forever
        server, port = _start_server({
            "/loop": (302, {"Location": "/loop"}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/loop",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 2,
            })
            assert resp.get("error"), f"Expected error, got: {resp}"
            assert "redirect" in resp["error"].lower()
        finally:
            server.shutdown()

    def test_direct_fetch_no_redirect(self, fetcher_script):
        """Direct 200 response (no redirect) works with pinned IP."""
        server, port = _start_server({
            "/target": (200, {"Content-Type": "text/plain"}, b"hello direct"),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/target",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 3,
            })
            assert not resp.get("error"), f"Unexpected error: {resp.get('error')}"
            assert resp["status_code"] == 200
            body = base64.b64decode(resp["body_base64"])
            assert body == b"hello direct"
        finally:
            server.shutdown()

    def test_redirect_chain_blocked_at_second_hop(self, fetcher_script):
        """Multi-hop: first redirect is to same server (same loopback IP, different port)
        — verifies the per-hop re-validation fires on the second hop too."""
        server2, port2 = _start_server({
            "/final": (200, {}, b"should not reach"),
        })
        server1, port1 = _start_server({
            "/hop1": (302, {"Location": f"http://10.0.0.1:{port2}/final"}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port1}/hop1",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 3,
            })
            assert resp.get("error"), f"Expected block on private-IP redirect, got: {resp}"
            assert "Blocked" in resp["error"] or "blocked" in resp["error"]
        finally:
            server1.shutdown()
            server2.shutdown()

    def test_no_missing_location_header(self, fetcher_script):
        """302 with no Location header returns a clear error."""
        server, port = _start_server({
            "/bad-redirect": (302, {}, b""),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/bad-redirect",
                "pinned_ip": "127.0.0.1",
                "max_redirects": 3,
            })
            assert resp.get("error"), f"Expected error, got: {resp}"
        finally:
            server.shutdown()


# ── Pinned IP tests ──────────────────────────────────────────────────────────

class TestPinnedIP:
    def test_pinned_ip_used_for_initial_request(self, fetcher_script):
        """Fetcher connects to pinned_ip, not re-resolved host."""
        server, port = _start_server({
            "/": (200, {"Content-Type": "text/plain"}, b"pinned"),
        })
        try:
            # Use 127.0.0.1 pinned_ip — server listens on 127.0.0.1
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/",
                "pinned_ip": "127.0.0.1",
            })
            assert not resp.get("error"), resp.get("error")
            assert resp["status_code"] == 200
        finally:
            server.shutdown()

    def test_none_pinned_ip_resolves_locally(self, fetcher_script):
        """With pinned_ip=None, fetcher resolves hostname itself."""
        server, port = _start_server({
            "/": (200, {"Content-Type": "text/plain"}, b"resolved"),
        })
        try:
            resp = _run_fetcher(fetcher_script, {
                "url": f"http://127.0.0.1:{port}/",
                "pinned_ip": None,
            })
            # 127.0.0.1 is in blocklist, so self-resolve should block it
            assert resp.get("error"), f"Expected block on loopback, got: {resp}"
        finally:
            server.shutdown()
