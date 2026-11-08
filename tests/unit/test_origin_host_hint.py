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


def test_own_ips_are_non_loopback_ip_literals(monkeypatch):
    """Bound to 0.0.0.0 the UI accepts this machine's own IPs as Host, so the
    LAN link it prints works without --allow-host."""
    from agent.ui import http_loop
    monkeypatch.setattr("agent.ui_server.tls.local_names",
                        lambda extra=None: ["127.0.0.1", "::1", "localhost", "fractal",
                                            "192.168.31.42", "172.17.0.1"])
    assert http_loop._own_ips() == ["192.168.31.42", "172.17.0.1"]


def test_lan_ip_allow_list_is_wired_for_wildcard_bind():
    import inspect
    from agent.ui import http_loop
    src = inspect.getsource(http_loop)
    assert '_lan_ips = _own_ips() if host in ("0.0.0.0", "") else []' in src
    assert "_extra_hosts += [ip for ip in _lan_ips" in src
