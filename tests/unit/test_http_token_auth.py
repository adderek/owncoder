"""The HTTP UI requires the per-process token on every request.

Loopback is not a credential — any local process can reach 127.0.0.1:port — so
a request must also carry the token printed at startup, either as `?token=…`
(the page load, which is handed the cookie) or as the cookie afterwards.
Router-proxied requests keep using the project secret.
"""
from __future__ import annotations

import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from agent.ui import http_loop


class _FakeUI:
    def __init__(self) -> None:
        from agent.ui_server.auth import AuthState
        self.auth = AuthState()

    def state(self) -> dict:
        return {"ok": True}


@pytest.fixture()
def ui_server():
    ui = _FakeUI()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), http_loop._make_handler(ui))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1], ui
    finally:
        srv.shutdown()
        srv.server_close()


def _get(port: int, path: str, *, cookie: str = "", headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    head = {"Host": f"127.0.0.1:{port}"}
    if cookie:
        head["Cookie"] = cookie
    head.update(headers or {})
    conn.request("GET", path, headers=head)
    res = conn.getresponse()
    body = res.read()
    set_cookie = res.getheader("Set-Cookie", "")
    conn.close()
    return res.status, body, set_cookie


def test_page_without_a_token_is_refused(ui_server):
    port, _ = ui_server
    status, body, _ = _get(port, "/")
    assert status == 403
    assert b"token" in body
    assert b"<html" not in body.lower() or b"forbidden" in body


def test_api_without_a_token_is_refused(ui_server):
    port, _ = ui_server
    status, _, _ = _get(port, "/api/state")
    assert status == 403


def test_wrong_token_is_refused(ui_server):
    port, _ = ui_server
    status, _, _ = _get(port, "/?token=not-the-token")
    assert status == 403


def test_bootstrap_token_sets_the_cookie_that_then_authorises(ui_server):
    port, ui = ui_server
    status, _, set_cookie = _get(port, f"/?token={ui.auth.token}")
    assert status == 200
    assert set_cookie.startswith("owncoder_auth=")
    assert "HttpOnly" in set_cookie
    cookie = set_cookie.split(";", 1)[0]

    status, body, _ = _get(port, "/api/state", cookie=cookie)
    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_cookie_from_a_previous_process_is_refused(ui_server):
    port, _ = ui_server
    status, _, _ = _get(port, "/api/state", cookie="owncoder_auth=some-old-token")
    assert status == 403


def test_project_secret_still_authorises_router_proxied_requests(ui_server, monkeypatch):
    port, _ = ui_server
    monkeypatch.setenv("AGENT_PROJECT_SECRET", "s3cret")
    status, body, _ = _get(port, "/api/state", headers={"X-Project-Secret": "s3cret"})
    assert status == 200
    assert json.loads(body) == {"ok": True}

    status, _, _ = _get(port, "/api/state")  # unproxied → blocked
    assert status == 403
    status, _, _ = _get(port, "/api/state", headers={"X-Project-Secret": "wrong"})
    assert status == 403
