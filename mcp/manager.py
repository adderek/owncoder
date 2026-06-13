"""Spawn configured MCP servers and register their tools in the agent registry.

Each discovered tool is exposed as ``mcp__<server>__<tool>`` so it can't collide
with built-in tools and is visibly external. Failures are isolated per server:
a server that won't start or won't list tools is logged and skipped, never
crashing agent startup.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# Live clients, kept alive for the session; shut down in teardown.
_clients: list = []
# Per-server load outcome for /mcp visibility: name → {ok, tools, error}.
_status: dict = {}
_NAME_RE = re.compile(r"[^a-z0-9_]+")


def _safe(part: str) -> str:
    return _NAME_RE.sub("_", part.strip().lower()).strip("_")


def _make_tool_name(server: str, tool: str) -> str:
    return f"mcp__{_safe(server)}__{_safe(tool)}"


def _wrap(client, server_name: str, tool_name: str):
    """Build a registry callable that proxies to client.call_tool."""

    def _call(**kwargs):
        try:
            return client.call_tool(tool_name, kwargs)
        except Exception as e:  # MCPError or transport failure
            logger.debug("mcp[%s]: call %s failed: %s", server_name, tool_name, e)
            return {"error": f"MCP tool '{tool_name}' failed: {e}"}

    return _call


def load_mcp_tools(config: "Config | None") -> int:
    """Start enabled MCP servers and register their tools. Returns tool count."""
    if config is None or not getattr(getattr(config, "mcp", None), "enabled", False):
        return 0

    from agent.mcp.client import MCPClient

    _status.clear()
    registered = 0
    for server in config.mcp.servers:
        if not getattr(server, "enabled", True):
            continue
        if getattr(server, "transport", "stdio") != "stdio":
            logger.warning("mcp[%s]: unsupported transport %r, skipping", server.name, server.transport)
            continue
        name = server.name or server.command
        client = MCPClient(server)
        try:
            client.start()
            tools = client.list_tools()
        except Exception as e:
            logger.warning("mcp[%s]: startup failed: %s", name, e)
            _status[name] = {"ok": False, "tools": [], "error": str(e)}
            try:
                client.close()
            except Exception:
                pass
            continue

        _clients.append(client)
        tool_names: list[str] = []
        for tool in tools:
            tname = tool.get("name")
            if not tname:
                continue
            schema = {
                "description": (tool.get("description") or f"MCP tool {tname} from server {name}.")[:1024],
                "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
            }
            full = _make_tool_name(name, tname)
            register(full, schema)(_wrap(client, name, tname))
            tool_names.append(full)
            registered += 1
        _status[name] = {"ok": True, "tools": tool_names, "error": None}
        logger.info("mcp[%s]: registered %d tool(s)", name, len(tools))

    return registered


def shutdown_mcp() -> None:
    """Terminate all live MCP server subprocesses."""
    while _clients:
        client = _clients.pop()
        try:
            client.close()
        except Exception:
            pass


def run_mcp_command(config, arg: str) -> str:
    """Text handler for the /mcp slash command: status of configured servers."""
    mcp_cfg = getattr(config, "mcp", None)
    if mcp_cfg is None or not getattr(mcp_cfg, "enabled", False):
        return "MCP disabled. Enable with [mcp] enabled = true and configure [[mcp.servers]]."
    servers = list(getattr(mcp_cfg, "servers", []) or [])
    if not servers:
        return "MCP enabled but no servers configured."
    lines = [f"MCP servers ({len(servers)}):"]
    for s in servers:
        name = s.name or s.command
        if not getattr(s, "enabled", True):
            lines.append(f"  {name}: disabled (config)")
            continue
        st = _status.get(name)
        if st is None:
            lines.append(f"  {name}: not loaded")
        elif st["ok"]:
            tools = ", ".join(t.split("__", 2)[-1] for t in st["tools"]) or "(none)"
            lines.append(f"  {name}: ok — {len(st['tools'])} tools: {tools}")
        else:
            lines.append(f"  {name}: FAILED — {st['error']}")
    return "\n".join(lines)
