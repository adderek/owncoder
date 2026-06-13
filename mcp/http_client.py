"""MCP Streamable HTTP transport — stdlib urllib, no external deps.

Implements the request/response subset of the MCP Streamable HTTP transport:
each JSON-RPC message is POSTed to a single endpoint; the server replies either
with ``application/json`` (one response) or a ``text/event-stream`` SSE body
carrying the response. A session id returned on ``initialize`` is echoed on
every later request. The optional standalone GET stream (unsolicited
server→client messages) is not used — initialize, tools/list, and tools/call
are all simple request/response.

Mirrors MCPClient's public API (start/list_tools/call_tool/close) so the
manager can treat transports interchangeably.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from agent.mcp.client import MCPError, flatten_tool_result

if TYPE_CHECKING:
    from agent.config.models import MCPServerConfig

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"


class MCPHttpClient:
    def __init__(self, server: "MCPServerConfig") -> None:
        self._server = server
        self._id = 0
        self._session_id: str | None = None
        self._closed = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self._server.url:
            raise MCPError(f"mcp server {self._server.name!r}: no url configured")
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "owncoder", "version": "1"},
            },
            timeout=self._server.init_timeout_s,
        )
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._session_id:
            return
        # Best-effort session teardown (servers may ignore / 405).
        try:
            req = urllib.request.Request(self._server.url, method="DELETE", headers=self._headers())
            urllib.request.urlopen(req, timeout=5).read()
        except Exception:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        result = self._request("tools/list", {}, timeout=self._server.init_timeout_s)
        return list(result.get("tools") or [])

    def call_tool(self, name: str, arguments: dict) -> Any:
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=self._server.call_timeout_s,
        )
        return flatten_tool_result(result)

    # ── transport internals ──────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        h.update({k: str(v) for k, v in (self._server.headers or {}).items()})
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _post(self, payload: dict, timeout: int):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self._server.url, data=body, method="POST", headers=self._headers())
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise MCPError(f"mcp http {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise MCPError(f"mcp http connection failed: {e.reason}") from e

    def _notify(self, method: str, params: dict) -> None:
        resp = self._post({"jsonrpc": "2.0", "method": method, "params": params}, timeout=self._server.init_timeout_s)
        try:
            resp.read()
        except Exception:
            pass

    def _request(self, method: str, params: dict, *, timeout: int) -> dict:
        rid = self._next_id()
        resp = self._post({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}, timeout=timeout)

        # Capture the session id handed out on initialize.
        sid = resp.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid

        ctype = (resp.headers.get_content_type() if hasattr(resp.headers, "get_content_type")
                 else resp.headers.get("Content-Type", "")).lower()
        raw = resp.read().decode("utf-8", "replace")

        if "text/event-stream" in ctype:
            msg = self._match_sse(raw, rid)
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise MCPError(f"mcp http: bad JSON response: {e}")
            msg = self._match_json(data, rid)

        if msg is None:
            raise MCPError(f"mcp http: no response for {method!r}")
        if "error" in msg:
            err = msg["error"]
            raise MCPError(f"mcp {method} error: {err.get('message', err)}")
        return msg.get("result") or {}

    @staticmethod
    def _match_json(data, rid: int) -> dict | None:
        if isinstance(data, dict):
            return data if data.get("id") == rid else None
        if isinstance(data, list):  # JSON-RPC batch
            for m in data:
                if isinstance(m, dict) and m.get("id") == rid:
                    return m
        return None

    @classmethod
    def _match_sse(cls, raw: str, rid: int) -> dict | None:
        # SSE: events separated by blank lines; payload carried in `data:` lines.
        for block in raw.replace("\r\n", "\n").split("\n\n"):
            data_lines = [ln[5:].lstrip() for ln in block.split("\n") if ln.startswith("data:")]
            if not data_lines:
                continue
            try:
                msg = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                continue
            hit = cls._match_json(msg, rid)
            if hit is not None:
                return hit
        return None
