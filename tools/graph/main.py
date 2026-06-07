"""Knowledge graph tools backed by graphifyy (pip install graphifyy).

graph_build  — run graphify extraction on project root, produce graphify-out/
graph_query  — search nodes by keyword across id/label/file
graph_path   — shortest path between two nodes
graph_context — callers, callees, imports for a symbol
"""
from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

_config: "Config | None" = None
_graph_cache: dict | None = None
_graph_mtime: float | None = None


def setup(config) -> None:
    global _config
    _config = config


def _graph_json_path() -> Path | None:
    if _config is None:
        return None
    root = Path(_config.root)
    return root / "graphify-out" / "graph.json"


def _load_graph() -> dict | None:
    global _graph_cache, _graph_mtime
    p = _graph_json_path()
    if p is None or not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _graph_cache is None or mtime != _graph_mtime:
        with p.open() as f:
            _graph_cache = json.load(f)
        _graph_mtime = mtime
    return _graph_cache


def _node_matches(node: dict, term: str) -> bool:
    term = term.lower()
    return (
        term in node.get("id", "").lower()
        or term in node.get("label", "").lower()
        or term in node.get("source_file", "").lower()
    )


@register(
    "graph_build",
    {
        "description": (
            "Run graphify knowledge-graph extraction on the project (or a subdirectory). "
            "Produces graphify-out/graph.json with call graph, import chains, and dependency edges. "
            "Run once before using graph_query/graph_path/graph_context. "
            "Requires `graphifyy` installed (`pip install graphifyy`). "
            "AST extraction is deterministic and makes no LLM calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to extract (default: project root)",
                },
            },
            "required": [],
        },
    },
)
def graph_build(path: str | None = None) -> dict:
    global _graph_cache, _graph_mtime
    _graph_cache = None
    _graph_mtime = None

    if not shutil.which("graphify"):
        return {
            "error": "graphify not found. Install with: pip install graphifyy",
            "hint": "Run inside agent venv: agent/.venv/bin/pip install graphifyy",
        }

    root = Path(_config.root) if _config else Path(".")
    target = Path(path).resolve() if path else root

    try:
        result = subprocess.run(
            ["graphify", str(target)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "graphify timed out after 5 minutes"}
    except Exception as e:
        return {"error": str(e)}

    out_path = root / "graphify-out" / "graph.json"
    if not out_path.exists():
        return {
            "success": False,
            "returncode": result.returncode,
            "stderr": result.stderr[-2000:] if result.stderr else "",
            "stdout": result.stdout[-1000:] if result.stdout else "",
        }

    graph = _load_graph()
    node_count = len(graph.get("nodes", [])) if graph else 0
    edge_count = len(graph.get("edges", [])) if graph else 0
    return {
        "success": True,
        "nodes": node_count,
        "edges": edge_count,
        "output_dir": str(root / "graphify-out"),
        "report": str(root / "graphify-out" / "GRAPH_REPORT.md"),
    }


@register(
    "graph_query",
    {
        "description": (
            "Search the knowledge graph for nodes (files, classes, functions) matching a keyword. "
            "Returns matching nodes with their direct edges (calls, imports, inherits). "
            "Use graph_build first if graph.json doesn't exist. "
            "Better than search_code for structural questions: 'what calls X', 'what does Y import'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name, file name, or keyword to search for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max nodes to return (default: 10)",
                },
            },
            "required": ["query"],
        },
    },
)
def graph_query(query: str, top_k: int = 10) -> dict:
    graph = _load_graph()
    if graph is None:
        return {"error": "graph.json not found. Run graph_build first."}

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    matched_nodes = [n for n in nodes if _node_matches(n, query)][:top_k]
    if not matched_nodes:
        return {"results": [], "count": 0, "query": query}

    matched_ids = {n["id"] for n in matched_nodes}
    relevant_edges = [
        e for e in edges
        if e.get("source") in matched_ids or e.get("target") in matched_ids
    ]

    return {
        "query": query,
        "results": matched_nodes,
        "edges": relevant_edges[:50],
        "count": len(matched_nodes),
    }


@register(
    "graph_path",
    {
        "description": (
            "Find the shortest structural path between two nodes in the knowledge graph "
            "(e.g., how does module A depend on module B, or how is function X connected to class Y). "
            "Use node IDs or names from graph_query results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_node": {
                    "type": "string",
                    "description": "Source node id or label (partial match OK)",
                },
                "to_node": {
                    "type": "string",
                    "description": "Target node id or label (partial match OK)",
                },
            },
            "required": ["from_node", "to_node"],
        },
    },
)
def graph_path(from_node: str, to_node: str) -> dict:
    graph = _load_graph()
    if graph is None:
        return {"error": "graph.json not found. Run graph_build first."}

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    def find_node_id(term: str) -> str | None:
        term_lower = term.lower()
        for n in nodes:
            if n.get("id", "").lower() == term_lower or n.get("label", "").lower() == term_lower:
                return n["id"]
        for n in nodes:
            if term_lower in n.get("id", "").lower() or term_lower in n.get("label", "").lower():
                return n["id"]
        return None

    src_id = find_node_id(from_node)
    tgt_id = find_node_id(to_node)

    if src_id is None:
        return {"error": f"Node not found: {from_node!r}"}
    if tgt_id is None:
        return {"error": f"Node not found: {to_node!r}"}

    # BFS
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        s, t, rel = e.get("source", ""), e.get("target", ""), e.get("relation", "")
        adjacency.setdefault(s, []).append((t, rel))

    from collections import deque
    visited = {src_id}
    queue: deque[list[tuple[str, str, str]]] = deque([[]])  # list of (from, to, relation)
    node_queue: deque[str] = deque([src_id])

    while queue:
        current_id = node_queue.popleft()
        current_path = queue.popleft()

        if current_id == tgt_id:
            return {
                "from": from_node,
                "to": to_node,
                "path": current_path,
                "hops": len(current_path),
            }

        if len(current_path) >= 8:
            continue

        for neighbor_id, rel in adjacency.get(current_id, []):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                node_queue.append(neighbor_id)
                queue.append(current_path + [(current_id, neighbor_id, rel)])

    return {
        "from": from_node,
        "to": to_node,
        "path": None,
        "error": "No path found (within 8 hops)",
    }


@register(
    "graph_context",
    {
        "description": (
            "Get structural context for a symbol: what it calls, what calls it, "
            "what it imports, what inherits from it. "
            "More precise than search_code for dependency and impact analysis."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name, class, function, or file (partial match OK)",
                },
            },
            "required": ["symbol"],
        },
    },
)
def graph_context(symbol: str) -> dict:
    graph = _load_graph()
    if graph is None:
        return {"error": "graph.json not found. Run graph_build first."}

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    matched = [n for n in nodes if _node_matches(n, symbol)]
    if not matched:
        return {"error": f"No node found matching {symbol!r}"}

    node = matched[0]
    nid = node["id"]

    callers, callees, imports, inherited_by, inherits_from, contains, contained_by = [], [], [], [], [], [], []

    for e in edges:
        s, t, rel = e.get("source", ""), e.get("target", ""), e.get("relation", "")
        if t == nid:
            if rel == "calls":
                callers.append(s)
            elif rel in ("imports", "imports_from", "uses"):
                imported_by = s
                imports.append({"imported_by": imported_by, "relation": rel})
            elif rel == "inherits":
                inherited_by.append(s)
            elif rel == "contains":
                contained_by.append(s)
        if s == nid:
            if rel == "calls":
                callees.append(t)
            elif rel in ("imports", "imports_from", "uses"):
                imports.append({"imports": t, "relation": rel})
            elif rel == "inherits":
                inherits_from.append(t)
            elif rel == "contains":
                contains.append(t)

    return {
        "node": node,
        "callers": callers,
        "callees": callees,
        "imports": imports,
        "inherits_from": inherits_from,
        "inherited_by": inherited_by,
        "contains": contains,
        "contained_by": contained_by,
    }
