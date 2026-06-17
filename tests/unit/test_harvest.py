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


def test_harvester_fetches_to_quarantine_via_http(tmp_path):
    # Hermetic: serve the intel over loopback HTTP (the harvester only accepts
    # http(s) schemes — file:// is refused, see test below).
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


def test_harvester_failure_is_nonfatal(tmp_path):
    out = tmp_path / "q"; out.mkdir()
    ok, info = _harvester.fetch_one({"name": "bad", "url": "http://127.0.0.1:1/nope"}, str(out))
    assert ok is False
    assert "bad:" in info


def test_research_refused_under_airgap(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.security.airgap = True
    out = harvest.run_research_command(cfg, "libyaml")
    assert "Air-gap is ON" in out


def test_research_usage_when_empty(tmp_path):
    out = harvest.run_research_command(_cfg(tmp_path), "")
    assert "Usage:" in out
