"""Model Context Protocol client — register external MCP-server tools.

Off by default. See agent/mcp/manager.py for the integration entry points
(`load_mcp_tools`, `shutdown_mcp`) and agent/mcp/client.py for the stdio
JSON-RPC transport.
"""
from .manager import load_mcp_tools, shutdown_mcp  # noqa: F401
