"""Tests for security research-mode harvester (isolated fetch -> quarantine)."""
from __future__ import annotations

import types
from pathlib import Path

from agent.security import harvest, _harvester


def _cfg(tmp_path):
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=types.SimpleNamespace(airgap=False),
    )


def test_build_targets_query_and_urls():
    ts = harvest._build_targets("libyaml", ["https://example.com/a"])
    names = [t["name"] for t in ts]
    assert any("nvd" in n for n in names)
    assert any("osv" in n for n in names)
    assert any("ghsa" in n for n in names)
    assert any(t["url"] == "https://example.com/a" for t in ts)
    # OSV is a POST with a package body.
    osv = next(t for t in ts if "osv" in t["name"])
    assert osv["method"] == "POST" and osv["body"]["package"]["name"] == "libyaml"


def test_build_targets_empty():
    assert harvest._build_targets("", []) == []


def test_harvester_fetches_to_quarantine_via_http(tmp_path, monkeypatch):
    # Hermetic: serve the intel over loopback HTTP (the harvester only accepts
    # http(s) schemes — file:// is refused, see test below). Loopback needs the
    # explicit operator opt-in, which is what this test is standing in for.
    monkeypatch.setenv(_harvester._ALLOW_PRIVATE_ENV, "1")
    import http.server
    import socketserver
    import threading

    payload = b"CVE-2014-9130 libyaml overflow details"

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # silence test output
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as srv:
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            out = tmp_path / "q"
            out.mkdir()
            ok, path = _harvester.fetch_one(
                {"name": "intel", "url": f"http://127.0.0.1:{port}/intel.txt"}, str(out)
            )
        finally:
            srv.shutdown()
    assert ok
    content = Path(path).read_text()
    assert "# SOURCE:" in content
    assert "untrusted external content" in content
    assert "CVE-2014-9130" in content


def test_harvester_refuses_file_url(tmp_path):
    # file:// must be rejected so a crafted spec can't read the local filesystem.
    src = tmp_path / "secret.txt"
    src.write_text("local secret")
    out = tmp_path / "q"
    out.mkdir()
    ok, info = _harvester.fetch_one({"name": "evil", "url": src.as_uri()}, str(out))
    assert ok is False
    assert "non-http(s)" in info


def test_harvester_failure_is_nonfatal(tmp_path, monkeypatch):
    monkeypatch.setenv(_harvester._ALLOW_PRIVATE_ENV, "1")
    out = tmp_path / "q"; out.mkdir()
    ok, info = _harvester.fetch_one({"name": "bad", "url": "http://127.0.0.1:1/nope"}, str(out))
    assert ok is False
    assert "bad:" in info


def test_harvester_refuses_loopback_without_the_opt_in(tmp_path):
    # Default policy: no intranet, no loopback — a crafted spec cannot probe
    # the local machine just by naming it.
    allowed, why = _harvester._url_policy("http://127.0.0.1:8080/admin")
    assert allowed is False and "refused loopback" in why
    assert _harvester._ALLOW_PRIVATE_ENV in why
    out = tmp_path / "q"; out.mkdir()
    ok, info = _harvester.fetch_one(
        {"name": "evil", "url": "http://127.0.0.1:1/nope"}, str(out))
    assert ok is False and "refused loopback" in info


def test_harvester_refuses_cloud_metadata_even_with_opt_in(monkeypatch):
    monkeypatch.setenv(_harvester._ALLOW_PRIVATE_ENV, "1")
    allowed, why = _harvester._url_policy("https://169.254.169.254/latest/meta-data/")
    assert allowed is False and "not a public host" in why


def test_harvester_requires_https_off_loopback(monkeypatch):
    monkeypatch.setenv(_harvester._ALLOW_PRIVATE_ENV, "1")
    allowed, why = _harvester._url_policy("http://93.184.216.34/notes.txt")
    assert allowed is False and "use https://" in why
    allowed, why = _harvester._url_policy("https://93.184.216.34/notes.txt")
    assert allowed is True, why


def test_harvester_refuses_unresolvable_host():
    allowed, why = _harvester._url_policy("https://nonexistent.invalid/a")
    assert allowed is False and "not a public host" in why


def test_harvester_refuses_credentials_in_url():
    allowed, why = _harvester._url_policy("https://user:pw@93.184.216.34/a")
    assert allowed is False and "credentials" in why


def test_harvester_refuses_redirect_to_a_private_host(tmp_path, monkeypatch):
    # A reputable public host that bounces to the metadata service must not be
    # followed: the hop is re-vetted, not just the URL from the spec.
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

        def log_message(self, *a):
            pass

    monkeypatch.setenv(_harvester._ALLOW_PRIVATE_ENV, "1")
    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            out = tmp_path / "q"; out.mkdir()
            ok, info = _harvester.fetch_one(
                {"name": "bounce", "url": f"http://127.0.0.1:{port}/go"}, str(out))
        finally:
            srv.shutdown()
    assert ok is False
    assert "redirect" in info and "not a public host" in info


def test_research_refused_under_airgap(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.security.airgap = True
    out = harvest.run_research_command(cfg, "libyaml")
    assert "Air-gap is ON" in out


def test_research_usage_when_empty(tmp_path):
    out = harvest.run_research_command(_cfg(tmp_path), "")
    assert "Usage:" in out
