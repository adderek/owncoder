"""analyze_binary + asm_tree: index a native binary through REA, then navigate it.

analyze_binary is the expensive offline pass (one LLM call per function and
per group); asm_tree is the cheap read path the chat uses afterwards — top
summary first, then drill into the groups that matter, then fetch exact
pseudocode from REA only for the few functions that need it.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from agent.tools import register
from agent.tools.analyze_asm import get_interrupt_flag

logger = logging.getLogger(__name__)

_config = None
_data_provider = None


def setup(config, data_provider) -> None:
    global _config, _data_provider
    _config = config
    _data_provider = data_provider


def _store():
    return _data_provider.get_asm_store() if _data_provider else None


@register(
    "analyze_binary",
    {
        "description": (
            "Index a native binary (EXE/DLL/ELF/DOS) through the REA MCP server: Ghidra "
            "facts (functions, calls, imports, pseudocode) + LLM descriptions, callees "
            "first, grouped along the call graph up to one whole-application summary. "
            "Expensive (one LLM call per function); resumable — re-run continues. "
            "Then navigate with asm_tree and search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Binary path (must be inside the REA jail)"},
                "max_functions": {"type": "integer", "description": "Describe at most N more functions this run"},
                "force": {"type": "boolean", "description": "Discard previous results and start over"},
                "annotate": {"type": "boolean", "description": "Write inferred names/comments back into Ghidra (needs annotate_native_function in tools_allow)"},
            },
            "required": ["path"],
        },
    },
)
def analyze_binary(path: str, max_functions: int | None = None, force: bool = False,
                   annotate: bool = False) -> dict:
    store = _store()
    if _config is None or store is None:
        return {"error": "analyze_binary not initialized. Run 'agent init' first."}
    if not _config.asm.enabled:
        return {"error": "Assembly analysis is disabled. Set asm_analysis.enabled = true."}
    from agent.mcp import manager
    server = manager.get_server(_config.asm.rea_server)
    if server is None:
        return {"error": f"MCP server {_config.asm.rea_server!r} is not running. "
                         "See agent/docs/rea.md (mcp.enabled, server config)."}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"error": f"File not found: {p}"}

    from openai import OpenAI
    from agent.rag.asm_describer import AsmDescriber
    from agent.rag.binary_pipeline import BinaryAnalysisPipeline
    from agent.rag.rea_facts import ReaError, ReaFacts

    client = OpenAI(base_url=_config.llm.base_url, api_key=_config.llm.api_key)
    describer = AsmDescriber(client, _config.asm, _config.llm, token_limits=_config.token_limits)
    started = time.time()

    def progress(event: str, d: dict) -> None:
        if event == "described" and (d["index"] % 25 == 0 or d["index"] == d["total"]):
            logger.info("analyze_binary %s: %d/%d functions (%.0fs)",
                        p.name, d["index"], d["total"], time.time() - started)
        elif event == "facts":
            logger.info("analyze_binary %s: %d functions, %d imports", p.name,
                        d["functions"], d["imports"])

    interrupt = get_interrupt_flag()
    interrupt.clear()
    pipeline = BinaryAnalysisPipeline(
        store, _data_provider.get_embedder(), describer, ReaFacts(*server),
        _config.asm, interrupt_flag=interrupt, progress_cb=progress)
    try:
        result = pipeline.analyze(str(p), force=force, annotate=annotate,
                                  max_functions=max_functions)
    except ReaError as e:
        return {"error": f"REA: {e}"}
    result["seconds"] = round(time.time() - started)
    if result.get("pending"):
        result["message"] = (f"{result['pending']} functions still to describe; "
                             "call analyze_binary again to continue.")
    return result


def _short(text: str | None, n: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[:n - 1] + "…"


def _line(u: dict) -> str:
    where = u.get("addr") or f"L{u['level']}"
    return (f"[{u['id']}] {u.get('inferred_name') or '?'} ({where}, {u.get('confidence') or '?'}): "
            f"{_short(u.get('description'))}")


@register(
    "asm_tree",
    {
        "description": (
            "Navigate an analyzed binary/asm hierarchy top-down. No args: list analyzed "
            "files. path: the top-level summary and its parts. unit_id: that unit and its "
            "children; for a function also signature, behaviour, side effects, callers, callees."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Analyzed file (as given to analyze_binary/analyze_asm)"},
                "unit_id": {"type": "string", "description": "Unit id from a previous asm_tree/search result"},
            },
        },
    },
)
def asm_tree(path: str | None = None, unit_id: str | None = None) -> dict:
    store = _store()
    if store is None:
        return {"error": "asm index not initialized."}
    if unit_id:
        u = store.get_unit(unit_id)
        if u is None:
            return {"error": f"No unit {unit_id!r}."}
        return {"text": _unit_view(store, u)}
    if not path:
        files = store.list_files()
        if not files:
            return {"text": "No analyzed files. Use analyze_binary or analyze_asm."}
        return {"text": "\n".join(f"{f['path']} [{f['status']}]" for f in files)}
    resolved = str(Path(path).expanduser().resolve())
    units = store.get_units_for_file(resolved) or store.get_units_for_file(path)
    if not units:
        return {"error": f"{path} not analyzed."}
    top = max(u["level"] for u in units)
    tops = [u for u in units if u["level"] == top]
    if len(tops) == 1:
        return {"text": _unit_view(store, tops[0])}
    return {"text": f"{len(tops)} top units at level {top} (hierarchy incomplete):\n"
                    + "\n".join(_line(u) for u in tops[:60])}


def _unit_view(store, u: dict) -> str:
    out = [_line(u)]
    if u.get("description") and len(u["description"]) > 160:
        out.append(f"purpose: {u['description']}")
    if u.get("parent_id"):
        out.append(f"parent: {u['parent_id']}")
    if u["level"] > 0:
        if u.get("key_patterns"):
            out.append(f"patterns: {_short(u['key_patterns'], 400)}")
        kids = store.get_children(u["id"])
        out.append(f"children ({len(kids)}):")
        out += ["  " + _line(k) for k in kids]
        return "\n".join(out)
    ev = json.loads(u["evidence"]) if u.get("evidence") else {}
    if ev.get("signature"):
        out.append(f"signature: {ev['signature']}  (ghidra: {ev.get('ghidra_name')})")
    if u.get("key_patterns"):
        out.append(f"behaviour: {u['key_patterns']}")
    if u.get("side_effects"):
        out.append(f"side effects: {u['side_effects']}")
    if u.get("calls"):
        out.append(f"calls: {', '.join(json.loads(u['calls']))}")
    callers = [s for s, d in store.get_edges(u["path"]) if d == u["id"]]
    if callers:
        names = [(store.get_unit(c) or {}).get("inferred_name") or c for c in callers]
        out.append("called by: " + ", ".join(f"{n} [{c}]" for n, c in zip(names, callers)))
    if ev.get("unverified_apis"):
        out.append(f"UNVERIFIED claims (not in call list): {', '.join(ev['unverified_apis'])}")
    if u.get("addr"):
        from agent.mcp.manager import _make_tool_name
        server = _config.asm.rea_server if _config else "rea"
        out.append(f"exact code: {_make_tool_name(server, 'procedure_pseudo_code')}"
                   f"(procedure=\"{u['addr']}\")")
    return "\n".join(out)
