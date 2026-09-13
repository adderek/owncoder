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
        from agent.tools.graph.main import graph_context, node_location, symbol_name
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
    # graph_context matches substrings; callers of a node that merely contains
    # the name are callers of a different symbol.
    if node.get("id") != name and symbol_name(node.get("label", "")).lower() != name.lower():
        return {}, f"no graph node named {name!r}"
    out: dict = {
        "callers": _trim(ctx.get("callers") or []),
        "callees": _trim(ctx.get("callees") or []),
        "imports": _trim(ctx.get("imports") or []),
        "inherits_from": ctx.get("inherits_from") or [],
        "inherited_by": _trim(ctx.get("inherited_by") or []),
    }
    if node:
        file, line = node_location(node)
        out["definition"] = [{
            "id": node.get("id"),
            "file": file,
            "line": line,
            "kind": node.get("kind") or node.get("type"),
        }]
    if ctx.get("warning"):
        out["graph_warning"] = ctx["warning"]
    if (ctx.get("exact_matches") or 1) > 1:
        # Several symbols share the name; the graph picked one. grep lists them all.
        out["graph_ambiguous"] = ctx["exact_matches"]
    if ctx.get("also_matched"):
        out["also_matched"] = ctx["also_matched"]
    return out, None


def _from_kb(name: str) -> tuple[dict, str | None]:
    """The KB node that IS this symbol: its description and the notes attached to it.

    Exact resolution, not a search: fuzzy hits for other symbols cost context
    and answer a different question.
    """
    try:
        from agent.tools.kb import _get_corpus, resolve_refs
    except Exception:
        return {}, "kb not enabled"
    try:
        corpus = _get_corpus()
        cands = [c for c in resolve_refs(corpus, name)
                 if c["kind"] in ("function", "method", "class", "file", "symbol")]
        if not cands:
            return {}, None
        best = cands[0]
        node = corpus.get(best["id"])
        rows = corpus.conn.execute(
            "SELECT n.kind, n.body FROM notes n JOIN note_attachments a ON a.note_id = n.id "
            "WHERE a.target_kind = 'node' AND a.target_ref = ? ORDER BY n.created_at DESC LIMIT 5",
            (best["id"],)).fetchall()
    except Exception as exc:
        logger.debug("find_symbol: kb lookup failed", exc_info=True)
        return {}, f"kb error: {exc}"
    facts: dict = {}
    description = node and (node.description_override or node.description_base)
    if description:
        facts["description"] = description
    if rows:
        facts["notes"] = [{"kind": r[0], "text": _clip(r[1], 300)} for r in rows]
    if not facts:
        return {}, None
    return {"facts": facts}, None


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
        # grep only has to run when a fresh graph already placed the symbol at a
        # file and line. A stale graph's line numbers drift, so grep re-checks.
        if label == "grep":
            placed = merged.get("definition") or []
            if (placed and all(d.get("file") and d.get("line") for d in placed)
                    and not merged.get("graph_warning") and not merged.get("graph_ambiguous")):
                continue
        payload, reason = fn()
        if reason:
            unavailable.append(f"{label}: {reason}")
        if payload:
            sources.append(label)
            for key, value in payload.items():
                if label == "grep" and key == "definition":
                    merged[key] = value  # live files beat a stale or unplaced graph node
                else:
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
