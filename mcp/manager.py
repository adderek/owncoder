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
            try:
                client.close()
            except Exception:
                pass
            continue

        _clients.append(client)
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
            registered += 1
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
