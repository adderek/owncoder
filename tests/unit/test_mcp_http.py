"""Unit tests for the MCP Streamable HTTP transport (stdlib server, no network)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from agent.config.models import MCPConfig, MCPServerConfig
from agent.mcp.http_client import MCPHttpClient
from agent.mcp.client import MCPError
from agent.mcp import manager


def _handle(method, mid, params, sse: bool):
    """Return (status, headers, body_bytes) for a JSON-RPC request."""
    def rpc(result=None, error=None):
        obj = {"jsonrpc": "2.0", "id": mid}
        if error is not None:
            obj["error"] = error
        else:
            obj["result"] = result
        if sse:
            return ("text/event-stream", f"event: message\ndata: {json.dumps(obj)}\n\n".encode())
        return ("application/json", json.dumps(obj).encode())

    if method == "initialize":
        return rpc({"protocolVersion": "2024-11-05", "capabilities": {}})
    if method == "tools/list":
        return rpc({"tools": [
            {"name": "echo", "description": "echo", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
        ]})
    if method == "tools/call":
        if params.get("name") == "echo":
            return rpc({"content": [{"type": "text", "text": "echo: " + str(params.get("arguments", {}).get("text", ""))}]})
        return rpc(error={"code": -32601, "message": "unknown tool"})
    return rpc(error={"code": -32601, "message": "unknown method"})


def _make_handler(sse: bool):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            msg = json.loads(self.rfile.read(length) or b"{}")
            mid = msg.get("id")
            if mid is None:  # notification
                self.send_response(202)
                self.end_headers()
                return
            ctype, body = _handle(msg.get("method"), mid, msg.get("params", {}), sse)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            if msg.get("method") == "initialize":
                self.send_header("Mcp-Session-Id", "sess-123")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_DELETE(self):
            self.send_response(200)
            self.end_headers()

    return H


@pytest.fixture(params=[False, True], ids=["json", "sse"])
def http_server(request):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(request.param))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    yield MCPServerConfig(
        name="remote",
        transport="http",
        url=f"http://127.0.0.1:{port}/mcp",
        init_timeout_s=5,
        call_timeout_s=5,
    )
    srv.shutdown()


class TestHttpClient:
    def test_init_list_call(self, http_server):
        c = MCPHttpClient(http_server)
        c.start()
        assert c._session_id == "sess-123"
        tools = c.list_tools()
        assert [t["name"] for t in tools] == ["echo"]
        assert c.call_tool("echo", {"text": "hi"}) == "echo: hi"
        c.close()

    def test_tool_error_raises(self, http_server):
        c = MCPHttpClient(http_server)
        c.start()
        with pytest.raises(MCPError):
            c.call_tool("missing", {})
        c.close()

    def test_no_url_raises(self):
        with pytest.raises(MCPError):
            MCPHttpClient(MCPServerConfig(name="x", transport="http")).start()


class TestManagerHttp:
    @pytest.fixture(autouse=True)
    def _clean(self):
        import agent.tools as t
        reg, sch = dict(t._registry), list(t._schemas)
        yield
        t._registry.clear(); t._registry.update(reg)
        t._schemas[:] = sch
        manager._clients.clear(); manager._status.clear()

    def test_registers_http_tools(self, http_server):
        import agent.tools as t
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=True, servers=[http_server]))
        n = manager.load_mcp_tools(cfg)
        assert n == 1
        fn = t.get_tool("mcp__remote__echo")
        assert fn(text="yo") == "echo: yo"
        assert "remote" in manager.run_mcp_command(cfg, "")
        manager.shutdown_mcp()

    def test_unsupported_transport_skipped(self):
        bad = MCPServerConfig(name="ws", transport="websocket", url="ws://x")
        cfg = SimpleNamespace(mcp=MCPConfig(enabled=True, servers=[bad]))
        assert manager.load_mcp_tools(cfg) == 0
        assert "unsupported" in manager.run_mcp_command(cfg, "").lower() or "FAILED" in manager.run_mcp_command(cfg, "")
