"""Regressions from the 2026-11-11 security review (network-facing surface)."""
from __future__ import annotations

import asyncio
import json

import pytest

from agent.notify.relay_server import PRESENCE_MAX_FIELD_LEN, RelayHub
from agent.security.query_gate import _is_ip_safe
from agent.ui_server.client_auth import ClientRegistry


@pytest.mark.parametrize("ip", [
    "::",                 # unspecified: the kernel connects it to this host
    "100.100.1.1",        # CGNAT / Tailscale peer
    "64:ff9b::7f00:1",    # NAT64 wrapping 127.0.0.1
    "2002:7f00:1::",      # 6to4 wrapping 127.0.0.1
    "240.0.0.1", "255.255.255.255", "198.18.0.1",
])
def test_fetch_gate_blocks_internal_ranges(ip):
    assert _is_ip_safe(ip) is False


@pytest.mark.parametrize("ip", ["1.1.1.1", "8.8.8.8", "2606:4700::1111"])
def test_fetch_gate_still_allows_public(ip):
    assert _is_ip_safe(ip) is True


def test_fetcher_script_blocklist_matches_gate():
    """The sandboxed fetcher re-validates every redirect hop with its own copy."""
    from agent.security.query_gate import _BLOCKED_NETWORKS
    from agent.tools.web_search.http_executor import _FETCHER_SCRIPT
    import ipaddress
    import re
    in_script = {ipaddress.ip_network(n)
                 for n in re.findall(r'IPv[46]Network\("([^"]+)"\)', _FETCHER_SCRIPT)}
    assert in_script == set(_BLOCKED_NETWORKS)


def test_connect_request_user_agent_loses_control_characters(tmp_path):
    reg = ClientRegistry(store=tmp_path / "clients.json")
    p, _claim, err = reg.request("10.0.0.9", "Mozilla\x1b[2J\x1b]0;pwn\x07 ok\r\n")
    assert not err
    assert p.agent == "Mozilla[2J]0;pwn ok"


class _WS:
    remote_address = ("10.0.0.9", 1)

    def __init__(self, hello):
        self._hello = hello
        self.closed = None

    async def recv(self):
        return json.dumps(self._hello)

    async def close(self, code=1000, reason=""):
        self.closed = code


def test_relay_non_ascii_token_is_refused_not_raised():
    hub = RelayHub(token="secret")
    ws = _WS({"type": "hello", "token": "zażółć", "role": "client"})
    assert asyncio.run(hub._auth(ws)) is None
    assert ws.closed == 4401


def test_relay_hello_name_is_clamped():
    hub = RelayHub(token="secret")
    ws = _WS({"type": "hello", "token": "secret", "role": "client", "name": "n" * 5000})
    _role, name, _v = asyncio.run(hub._auth(ws))
    assert len(name) == PRESENCE_MAX_FIELD_LEN
