"""Unit tests for the MCP stdio client + tool-registration manager.

Uses a tiny in-repo fake MCP server (JSON-RPC over stdio) so no network or
external server is needed.
"""
from __future__ import annotations

import sys
import textwrap
from types import SimpleNamespace

import pytest

from agent.config.models import MCPConfig, MCPServerConfig
from agent.mcp.client import MCPClient, MCPError
from agent.mcp import manager


# A minimal MCP server: initialize, tools/list (one "echo" tool), tools/call.
_FAKE_SERVER = textwrap.dedent('''
    import sys, json
    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        mid = msg.get("id")
        method = msg.get("method")
        if method == "initialize":
            send({"jsonrpc":"2.0","id":mid,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"fake"}}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send({"jsonrpc":"2.0","id":mid,"result":{"tools":[
                {"name":"echo","description":"Echo back text","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
                {"name":"boom","description":"Always errors","inputSchema":{"type":"object","properties":{}}}
            ]}})
        elif method == "tools/call":
            p = msg.get("params", {})
            name = p.get("name"); args = p.get("arguments", {})
            if name == "echo":
                send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"echo: "+str(args.get("text",""))}]}})
            elif name == "boom":
                send({"jsonrpc":"2.0","id":mid,"result":{"content":[{"type":"text","text":"kaboom"}],"isError":True}})
            else:
                send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown tool"}})
        else:
            send({"jsonrpc":"2.0","id":mid,"error":{"code":-32601,"message":"unknown method"}})
''')


@pytest.fixture()
def fake_server(tmp_path):
    script = tmp_path / "fake_mcp.py"
    script.write_text(_FAKE_SERVER)
    return MCPServerConfig(
        name="fake",
        command=sys.executable,
        args=[str(script)],
        init_timeout_s=10,
        call_timeout_s=10,
    )


@pytest.fixture()
def clean_registry():
    """Snapshot + restore the global tool registry so MCP tools don't leak."""
    import agent.tools as t
    reg = dict(t._registry)
    sch = list(t._schemas)
    yield
    t._registry.clear(); t._registry.update(reg)
    t._schemas[:] = sch
    manager._clients.clear()


class TestClient:
    def test_list_and_call(self, fake_server):
        client = MCPClient(fake_server)
        client.start()
        try:
            tools = client.list_tools()
            assert {t["name"] for t in tools} == {"echo", "boom"}
            assert client.call_tool("echo", {"text": "hi"}) == "echo: hi"
            err = client.call_tool("boom", {})
            assert err["isError"] is True
            assert "kaboom" in err["error"]
        finally:
            client.close()

    def test_unknown_tool_raises(self, fake_server):
        client = MCPClient(fake_server)
        client.start()
        try:
            with pytest.raises(MCPError):
                client.call_tool("nope", {})
        finally:
            client.close()

    def test_bad_command_fails_to_start(self):
        client = MCPClient(MCPServerConfig(name="x", command="this-cmd-does-not-exist-zzz"))
        with pytest.raises(Exception):
            client.start()


class TestManager:
    def test_disabled_returns_zero(self, clean_registry):
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=False, servers=[]))
        assert manager.load_mcp_tools(cfg) == 0

    def test_registers_tools(self, fake_server, clean_registry):
        import agent.tools as t
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=True, servers=[fake_server]))
        n = manager.load_mcp_tools(cfg)
        assert n == 2
        fn = t.get_tool("mcp__fake__echo")
        assert fn is not None
        assert fn(text="yo") == "echo: yo"
        names = {s["function"]["name"] for s in t.get_schemas()}
        assert "mcp__fake__echo" in names and "mcp__fake__boom" in names
        manager.shutdown_mcp()

    def test_bad_server_isolated(self, clean_registry):
        bad = MCPServerConfig(name="bad", command="this-cmd-does-not-exist-zzz")
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=True, servers=[bad]))
        # Should not raise; just registers nothing.
        assert manager.load_mcp_tools(cfg) == 0

    def test_wrapper_reports_error_on_dead_client(self, fake_server, clean_registry):
        import agent.tools as t
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=True, servers=[fake_server]))
        manager.load_mcp_tools(cfg)
        manager.shutdown_mcp()  # kill the server
        fn = t.get_tool("mcp__fake__echo")
        out = fn(text="x")
        assert isinstance(out, dict) and "error" in out
