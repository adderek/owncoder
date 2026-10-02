"""The 403 for a failed Origin/Host check names the --allow-host value that fixes it."""
from types import SimpleNamespace as NS

from agent.ui_server.auth import origin_host_error, validate_origin_host


def _h(host="", origin=""):
    hdr = {}
    if host:
        hdr["Host"] = host
    if origin:
        hdr["Origin"] = origin
    return NS(headers=hdr)


def test_hint_names_the_server_address_the_browser_used(monkeypatch):
    monkeypatch.delenv("AGENT_ALLOWED_HOSTS", raising=False)
    h = _h("192.168.2.64:8180")
    assert not validate_origin_host(h)
    assert "--allow-host 192.168.2.64" in origin_host_error(h)["error"]


def test_hint_uses_origin_when_host_is_allowed(monkeypatch):
    monkeypatch.setenv("AGENT_ALLOWED_HOSTS", "192.168.2.64")
    h = _h("192.168.2.64:8180", "http://evil.example:80")
    assert "--allow-host evil.example" in origin_host_error(h)["error"]


def test_hint_sanitises_echoed_name(monkeypatch):
    monkeypatch.delenv("AGENT_ALLOWED_HOSTS", raising=False)
    msg = origin_host_error(_h("<script>x</script>:1"))["error"]
    assert "<" not in msg and ">" not in msg
