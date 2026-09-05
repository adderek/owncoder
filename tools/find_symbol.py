"""find_symbol — one entry point for "tell me about this symbol".

Eleven retrieval tools (search_code, grep_code, kb_search, kb_get, kb_callers,
kb_deps, graph_context, graph_query, graph_path, recall_*) is more choice than a
small local model can route, so it falls back on the tool it always understands:
read_file. This tool removes the choice for the most common structural question.

It owns no retrieval logic. It calls the graph, the KB and grep in that order,
merges what answered, and reports which sources were used — including the ones
that were unavailable, so "no result" is never mistaken for "no such symbol".

Notably it takes a NAME. kb_callers/kb_deps need a hex node id, which the model
can only obtain via a prior kb_search; that two-call handshake is most of why
they go unused.
"""
from __future__ import annotations

import json
import logging

from agent.tools import register

logger = logging.getLogger(__name__)

_MAX_PER_SECTION = 25
# Definition-site patterns, widest-used languages first. Word-boundary anchored
# so "createHouse" does not match "createHouseGrid".
_DEF_PATTERN = (
    r"(?:^|\s)(?:async\s+)?(?:def|class|function|func|fn|interface|struct|type)\s+{name}\b"
    r"|(?:const|let|var)\s+{name}\s*="
)


def _trim(items: list) -> list:
    return items[:_MAX_PER_SECTION]


def _from_graph(name: str) -> tuple[dict, str | None]:
    """(payload, unavailable_reason)."""
    try:
        from agent.tools.graph.main import graph_context
    except Exception:
        return {}, "graph tools not loaded"
    try:
        ctx = graph_context(name)
    except Exception as exc:
        logger.debug("find_symbol: graph_context failed", exc_info=True)
        return {}, f"graph error: {exc}"
    if not isinstance(ctx, dict) or ctx.get("error"):
        return {}, str(ctx.get("error") if isinstance(ctx, dict) else "graph unavailable")
    node = ctx.get("node") or {}
    out: dict = {
        "callers": _trim(ctx.get("callers") or []),
        "callees": _trim(ctx.get("callees") or []),
        "imports": _trim(ctx.get("imports") or []),
        "inherits_from": ctx.get("inherits_from") or [],
        "inherited_by": _trim(ctx.get("inherited_by") or []),
    }
    if node:
        out["definition"] = [{
            "id": node.get("id"),
            "file": node.get("file") or node.get("path"),
            "line": node.get("line"),
            "kind": node.get("kind") or node.get("type"),
        }]
    if ctx.get("warning"):
        out["graph_warning"] = ctx["warning"]
    if ctx.get("also_matched"):
        out["also_matched"] = ctx["also_matched"]
    return out, None


def _from_kb(name: str) -> tuple[dict, str | None]:
    try:
        from agent.tools.kb import kb_search
    except Exception:
        return {}, "kb not enabled"
    try:
        payload = json.loads(kb_search(name, limit=5))
    except Exception as exc:
        logger.debug("find_symbol: kb_search failed", exc_info=True)
        return {}, f"kb error: {exc}"
    if payload.get("error"):
        return {}, str(payload["error"])
    nodes = payload.get("nodes") or []
    if not nodes:
        return {}, None
    return {"facts": nodes[:5]}, None


def _from_grep(name: str, path: str | None) -> tuple[dict, str | None]:
    try:
        from agent.tools.search.grep import grep_code
    except Exception:
        return {}, "grep unavailable"
    try:
        res = grep_code(_DEF_PATTERN.format(name=name), path=path, max_results=_MAX_PER_SECTION)
    except Exception as exc:
        logger.debug("find_symbol: grep_code failed", exc_info=True)
        return {}, f"grep error: {exc}"
    if res.get("error"):
        return {}, str(res["error"])
    hits = [
        {"file": r["path"], "line": r["line"], "text": r.get("content", "").strip()}
        for r in res.get("results", [])
    ]
    return ({"definition": hits} if hits else {}), None


@register(
    "find_symbol",
    {
        "description": (
            "Everything about one symbol in a single call: where it is defined, what "
            "calls it, what it calls, imports, inheritance, and curated facts. "
            "Takes a plain NAME — no node ids, no prior lookup. "
            "Use this instead of reading files to find a function, and instead of "
            "chaining kb_search -> kb_callers or graph_query -> graph_context. "
            "Consults the call graph, the knowledge base and grep, and says which "
            "sources answered — an empty result with sources listed means the symbol "
            "is genuinely absent, not that an index was missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Symbol name: function, class, method, or module.",
                },
                "want": {
                    "type": "string",
                    "description": (
                        "Narrow the answer: 'definition', 'callers', 'callees', "
                        "'facts', or 'all' (default)."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Restrict the grep fallback to this directory.",
                },
            },
            "required": ["name"],
        },
    },
)
def find_symbol(name: str, want: str = "all", path: str | None = None) -> dict:
    name = (name or "").strip()
    if not name:
        return {"error": "name is required"}

    merged: dict = {}
    sources: list[str] = []
    unavailable: list[str] = []

    for label, fn in (("graph", lambda: _from_graph(name)),
                      ("kb", lambda: _from_kb(name)),
                      ("grep", lambda: _from_grep(name, path))):
        # grep only has to run when the graph did not already place the symbol.
        if label == "grep" and merged.get("definition"):
            continue
        payload, reason = fn()
        if reason:
            unavailable.append(f"{label}: {reason}")
        if payload:
            sources.append(label)
            for key, value in payload.items():
                merged.setdefault(key, value)

    out: dict = {"name": name, "sources": sources}
    if unavailable:
        out["unavailable"] = unavailable

    wanted = {
        "definition": ("definition",),
        "callers": ("callers",),
        "callees": ("callees", "imports"),
        "facts": ("facts",),
    }.get(want)
    for key, value in merged.items():
        if wanted is None or key in wanted or key.endswith("_warning"):
            out[key] = value

    if not sources:
        out["hint"] = (
            f"Nothing found for {name!r}. If the graph and KB are listed as unavailable "
            f"above, build them; otherwise try grep_code with a partial name."
        )
    elif out.get("definition"):
        first = out["definition"][0]
        if first.get("file") and first.get("line"):
            out["next"] = (
                f"read_file('{first['file']}', start_line={max(1, int(first['line']) - 5)}, "
                f"end_line={int(first['line']) + 40})"
            )
    return out
