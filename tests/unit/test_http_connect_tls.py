"""Operator-approved browser access and self-signed TLS for the HTTP UI."""
from __future__ import annotations

import json
import os
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler

import pytest

from agent.ui_server import client_auth
from agent.ui_server.client_auth import ClientRegistry


@pytest.fixture
def reg(tmp_path):
    return ClientRegistry(store=tmp_path / "clients.json", remember_days=30)


class TestApprovalFlow:
    def test_approve_once_hands_token_to_claim_holder_only(self, reg):
        p, claim, err = reg.request("192.168.31.42", "Firefox")
        assert not err and len(p.code) == 3
        assert reg.status(p.id, "wrong-claim")[0] == "unknown"
        assert reg.status(p.id, claim) == ("pending", "", False)
        assert reg.decide(p.id, True)
        state, token, remember = reg.status(p.id, claim)
        assert state == "approved" and token and not remember
        assert reg.status(p.id, claim)[1] == ""          # handed out once
        assert reg.validate(token) is not None
        assert not (reg.store).exists()                  # session-only: not persisted

    def test_host_recorded_and_sanitised(self, reg):
        p, _, _ = reg.request("10.0.0.7", "x", "evil.example:8180\r\n<b>")
        assert p.host == "evil.example:8180b" and p.public()["host"] == p.host
        assert reg.request("10.0.0.8", "x")[0].host == ""

    def test_deny(self, reg):
        p, claim, _ = reg.request("10.0.0.5", "x")
        reg.decide(p.id, False)
        assert reg.status(p.id, claim)[0] == "denied"
        assert not reg.decide(p.id, True)                # decided once

    def test_remember_persists_hash_only_0600(self, reg, tmp_path):
        p, claim, _ = reg.request("10.0.0.5", "x")
        reg.decide(p.id, True, remember=True)
        _, token, remember = reg.status(p.id, claim)
        assert remember
        raw = reg.store.read_text()
        assert token not in raw
        assert oct(os.stat(reg.store).st_mode & 0o777) == "0o600"
        again = ClientRegistry(store=reg.store)
        assert again.validate(token) is not None

    def test_external_revoke_applies_to_running_registry(self, reg):
        p, claim, _ = reg.request("10.0.0.5", "x")
        reg.decide(p.id, True, remember=True)
        token = reg.status(p.id, claim)[1]
        cli = ClientRegistry(store=reg.store)
        time.sleep(0.01)
        assert cli.revoke(cli.clients()[0].id)
        os.utime(reg.store, (time.time() + 5, time.time() + 5))
        assert reg.validate(token) is None

    def test_limits(self, reg):
        assert reg.request("10.0.0.1", "a")[2] == ""
        assert "already waiting" in reg.request("10.0.0.1", "a")[2]
        reg.request("10.0.0.2", "b")
        reg.request("10.0.0.3", "c")
        assert "too many pending" in reg.request("10.0.0.4", "d")[2]
        for _ in range(client_auth.RATE_MAX):
            reg.request("10.9.9.9", "spam")
        assert "too many connection requests" in reg.request("10.9.9.9", "spam")[2]

    def test_expiry(self, reg, monkeypatch):
        p, claim, _ = reg.request("10.0.0.1", "a")
        p.created -= client_auth.PENDING_TTL_S + 1
        assert reg.status(p.id, claim)[0] == "expired"
        assert not reg.decide(p.id, True)

    def test_listener_called(self, reg):
        seen = []
        reg.on_request(seen.append)
        p, _, _ = reg.request("10.0.0.1", "a")
        assert seen == [p]

    def test_cookie_flags(self, reg):
        h = reg.cookie_header("t", remember=True, secure=True)
        assert "HttpOnly" in h and "SameSite=Strict" in h and "Secure" in h and "Max-Age=" in h
        assert "Max-Age" not in reg.cookie_header("t", remember=False, secure=False)


crypto = pytest.importorskip("cryptography")


class TestSelfSignedTls:
    def test_cert_covers_names_and_is_reused(self, tmp_path):
        from agent.ui_server.tls import ensure_cert
        c1, k1, fp1 = ensure_cert(["localhost", "127.0.0.1", "192.168.2.64"], tmp_path)
        assert oct(os.stat(k1).st_mode & 0o777) == "0o600"
        _, _, fp2 = ensure_cert(["192.168.2.64"], tmp_path)
        assert fp2 == fp1                                  # subset → same cert
        _, _, fp3 = ensure_cert(["sharkoon.local"], tmp_path)
        assert fp3 != fp1                                  # new name → reissued
        names = json.loads((tmp_path / "http-ui.json").read_text())["names"]
        assert {"192.168.2.64", "sharkoon.local", "localhost"} <= set(names)

    def test_dual_protocol_server(self, tmp_path, monkeypatch):
        from agent.ui_server.tls import DualProtocolServer, ensure_cert, server_context
        monkeypatch.delenv("AGENT_ALLOWED_HOSTS", raising=False)
        cert, key, _ = ensure_cert(["127.0.0.1", "localhost"], tmp_path)

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(body)

        srv = DualProtocolServer(("127.0.0.1", 0), H)
        srv.ssl_context = server_context(str(cert), str(key))
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            # plain HTTP → redirect to https on the same port
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                s.sendall(f"GET /x?token=a HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
                resp = s.recv(4096).decode()
            assert resp.startswith("HTTP/1.1 308") and f"Location: https://127.0.0.1:{port}/x?token=a" in resp
            # foreign Host → no redirect
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                s.sendall(b"GET / HTTP/1.1\r\nHost: evil.example\r\n\r\n")
                assert s.recv(4096).startswith(b"HTTP/1.1 400")
            # --allow-any-host: well-formed foreign Host redirects, malformed still refused
            srv.any_host = True
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                s.sendall(b"GET / HTTP/1.1\r\nHost: box.lan:8180\r\n\r\n")
                assert f"Location: https://box.lan:{port}/".encode() in s.recv(4096)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
                s.sendall(b"GET / HTTP/1.1\r\nHost: a/b@c\r\n\r\n")
                assert s.recv(4096).startswith(b"HTTP/1.1 400")
            srv.any_host = False
            # HTTPS works
            ctx = ssl.create_default_context(cafile=str(cert))
            with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                with ctx.wrap_socket(raw, server_hostname="127.0.0.1") as s:
                    s.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                    assert s.recv(4096).startswith(b"HTTP/1.0 200")
        finally:
            srv.shutdown()
