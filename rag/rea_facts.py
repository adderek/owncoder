"""Ground-truth facts about a native binary, read from the REA MCP server.

The LLM pipeline (binary_pipeline.py) describes; this module only fetches what
Ghidra already knows: procedure boundaries, signatures, callees, pseudocode,
strings. Every REA answer comes either as ``{"result": ...}`` or as an evidence
envelope whose payload sits in ``raw_result``; `_payload` unwraps both.

Only read-only tools are called, plus ``annotate_native_function`` when the
caller asks for write-back. Each call goes through the server's
tools_allow/tools_deny, so a tool the operator did not admit is refused here
exactly as it would be for the chat model.
"""
from __future__ import annotations

import json
from typing import Any, Callable

READ_TOOLS = ("open_binary", "list_procedures", "procedure_info",
              "procedure_callees", "procedure_pseudo_code", "list_strings")
ANNOTATE_TOOL = "annotate_native_function"


class ReaError(RuntimeError):
    pass


def _payload(resp: Any) -> Any:
    """Unwrap an REA answer to its payload; raise ReaError on a tool error."""
    if isinstance(resp, dict) and resp.get("isError"):
        raise ReaError(str(resp.get("error"))[:500])
    if isinstance(resp, str):
        try:
            resp = json.loads(resp)
        except json.JSONDecodeError:
            return resp
    if isinstance(resp, dict):
        if "raw_result" in resp:
            resp = resp["raw_result"]
        elif "result" in resp and len(resp) == 1:
            resp = resp["result"]
    # Single-value procedure answers: {"value": ..., "limitations": [...]}
    if isinstance(resp, dict) and "value" in resp and set(resp) <= {"value", "limitations"}:
        return resp["value"]
    return resp


class ReaFacts:
    """``call(tool, args) -> raw MCP result``; ``allowed(tool) -> bool``."""

    def __init__(self, call: Callable[[str, dict], Any],
                 allowed: Callable[[str], bool] = lambda _t: True) -> None:
        self._call = call
        self._allowed = allowed

    def _get(self, tool: str, args: dict) -> Any:
        if not self._allowed(tool):
            raise ReaError(f"REA tool {tool!r} is not in the server's tools_allow")
        return _payload(self._call(tool, args))

    def open(self, path: str) -> dict:
        """Open the binary in Ghidra. Returns {sha256, format, architecture, ...}."""
        return self._get("open_binary", {"path": path, "provider_id": "ghidra"})

    def procedures(self) -> tuple[list[dict], dict[str, str]]:
        """(internal procedures [{addr, name}], import map {thunk/ext addr: 'DLL::fn'})."""
        internal: list[dict] = []
        imports: dict[str, str] = {}
        for p in self._get("list_procedures", {}) or []:
            meta = p.get("procedure") or {}
            addr, name = p.get("address"), p.get("value") or ""
            if not addr:
                continue
            if meta.get("external") or meta.get("thunk"):
                imports[addr] = name
            else:
                internal.append({"addr": addr, "name": name})
        return internal, imports

    def info(self, addr: str) -> dict:
        v = self._get("procedure_info", {"procedure": addr}) or {}
        body = v.get("body") or {}
        return {
            "name": v.get("name"),
            "signature": v.get("signature"),
            "length": v.get("length") or body.get("total_bytes") or 0,
        }

    def callees(self, addr: str) -> list[str]:
        return list(self._get("procedure_callees", {"procedure": addr}) or [])

    def pseudocode(self, addr: str) -> str:
        v = self._get("procedure_pseudo_code", {"procedure": addr})
        return v if isinstance(v, str) else json.dumps(v)

    def strings(self) -> dict[str, str]:
        return {s["address"]: s.get("value", "")
                for s in (self._get("list_strings", {}) or []) if s.get("address")}

    def annotate(self, addr: str, name: str, comment: str) -> None:
        self._get(ANNOTATE_TOOL, {"procedure": addr, "name": name, "comment": comment})
