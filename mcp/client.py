"""Minimal MCP stdio client — newline-delimited JSON-RPC 2.0, no external deps.

Speaks the subset of the Model Context Protocol the agent needs: ``initialize``
handshake, ``tools/list`` discovery, and ``tools/call`` invocation over a
subprocess's stdin/stdout. A background thread reads responses and dispatches
them to per-request queues so notifications/logs interleaved on the stream
don't desync request/response matching.

Synchronous by design: the agent executes tools in a thread-pool executor, so
blocking calls here are fine and avoid pulling in an async MCP SDK.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.config.models import MCPServerConfig

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2024-11-05"


class MCPError(RuntimeError):
    pass


def flatten_tool_result(result: dict) -> Any:
    """Normalise an MCP tools/call result into a string or structured dict.

    Shared by every transport. Text-only content collapses to a string; an
    error result becomes {"error", "isError"}; mixed content is handed back raw.
    """
    content = result.get("content") or []
    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
    joined = "\n".join(p for p in text_parts if p)
    if result.get("isError"):
        return {"error": joined or "MCP tool reported an error", "isError": True}
    if not content:
        return {"ok": True}
    if joined and len(text_parts) == len(content):
        return joined
    return {"content": content}


class MCPClient:
    def __init__(self, server: "MCPServerConfig") -> None:
        self._server = server
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._id_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._pending_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._closed = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        s = self._server
        if not s.command:
            raise MCPError(f"mcp server {s.name!r}: no command configured")
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (s.env or {}).items()})
        argv = [s.command, *[str(a) for a in (s.args or [])]]
        logger.debug("mcp[%s]: spawning %s", s.name, argv)
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=s.cwd or None,
            env=env,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_reader.start()

        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "owncoder", "version": "1"},
            },
            timeout=s.init_timeout_s,
        )
        self._notify("notifications/initialized", {})

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
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
        with self._id_lock:
            self._id += 1
            return self._id

    def _send(self, payload: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise MCPError("mcp client not started")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._write_lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise MCPError(f"mcp write failed: {e}") from e

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, *, timeout: int) -> dict:
        rid = self._next_id()
        q: queue.Queue = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = q
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            try:
                msg = q.get(timeout=timeout)
            except queue.Empty:
                raise MCPError(f"mcp request {method!r} timed out after {timeout}s")
        finally:
            with self._pending_lock:
                self._pending.pop(rid, None)
        if "error" in msg:
            err = msg["error"]
            raise MCPError(f"mcp {method} error: {err.get('message', err)}")
        return msg.get("result") or {}

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("mcp[%s]: non-JSON line: %.120s", self._server.name, line)
                continue
            rid = msg.get("id")
            if rid is None:
                continue  # server notification — ignored
            with self._pending_lock:
                q = self._pending.get(rid)
            if q is not None:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass
        # stdout closed → fail any waiters so callers don't hang.
        with self._pending_lock:
            waiters = list(self._pending.values())
        for q in waiters:
            try:
                q.put_nowait({"error": {"message": "mcp server closed stdout"}})
            except queue.Full:
                pass

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            logger.debug("mcp[%s] stderr: %s", self._server.name, line.rstrip())
