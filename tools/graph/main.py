"""Knowledge graph tools backed by graphifyy (pip install graphifyy).

graph_build   — run graphify extraction, produce graphify-out/
graph_query   — BFS search via graphify query CLI
graph_path    — shortest path via graphify path CLI
graph_context — callers/callees/imports parsed from graph.json directly
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _graphify_bin() -> Path | None:
    """Find graphify in the same venv as the running interpreter."""
    candidate = Path(sys.executable).parent / "graphify"
    if candidate.exists():
        return candidate
    return None


def _root() -> Path:
    if _config is not None:
        return Path(_config.tools.working_dir).resolve()
    return Path(".").resolve()


def _graph_json_path() -> Path:
    return _root() / "graphify-out" / "graph.json"


def _load_graph() -> dict | None:
    global _graph_cache, _graph_mtime
    p = _graph_json_path()
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    if _graph_cache is None or mtime != _graph_mtime:
        with p.open() as f:
            _graph_cache = json.load(f)
        _graph_mtime = mtime
    return _graph_cache


def _run(cmd: list[str], timeout: int = 300) -> dict:
    """Run a graphify subcommand. Returns {stdout, stderr, returncode}."""
    bin_ = _graphify_bin()
    if bin_ is None:
        return {"error": "graphify not found. Install: agent/.venv/bin/pip install graphifyy"}
    try:
        r = subprocess.run(
            [str(bin_)] + cmd,
            cwd=str(_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"error": f"graphify timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


_SOURCE_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".kt", ".c", ".cpp", ".h", ".hpp", ".rb", ".cs", ".scala",
    ".sh", ".bash", ".lua", ".zig", ".swift",
}
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", "build", "dist",
    ".agent", ".venv", "venv", "graphify-out",
}


def _newest_source_mtime(root: Path) -> float:
    """Walk source files and return the newest mtime."""
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if Path(fname).suffix.lower() in _SOURCE_EXTS:
                try:
                    m = (Path(dirpath) / fname).stat().st_mtime
                    if m > newest:
                        newest = m
                except OSError:
                    pass
    return newest


def _graph_stale_warning() -> str | None:
    """Return a warning string if graph.json is older than newest source file, else None."""
    p = _graph_json_path()
    if not p.exists():
        return None  # no graph yet — handled separately
    graph_mtime = p.stat().st_mtime
    source_mtime = _newest_source_mtime(_root())
    if source_mtime > graph_mtime:
        return "graph may be stale (source files changed since last graph_build). Run graph_build(update_only=True) to refresh."
    return None


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
            "Build or refresh the knowledge graph for the project (or a subdirectory). "
            "Produces graphify-out/graph.json with call graph, import chains, and dependency edges. "
            "Run once before using graph_query/graph_path/graph_context, and again after code changes. "
            "Requires `graphifyy` installed (`agent/.venv/bin/pip install graphifyy`). "
            "No LLM or API key needed — pure AST extraction."
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
def graph_build(path: str | None = None, **_kwargs) -> dict:
    global _graph_cache, _graph_mtime
    _graph_cache = None
    _graph_mtime = None

    if _graphify_bin() is None:
        return {"error": "graphify not found. Install: agent/.venv/bin/pip install graphifyy"}

    target = str(Path(path).resolve() if path else _root())

    # Always use `update` — processes code only, no LLM/API key needed.
    # --force ensures it works on first run (no existing graph.json required).
    cmd = ["update", target, "--no-cluster", "--force"]
    result = _run(cmd, timeout=600)

    if "error" in result:
        return result

    out_path = _graph_json_path()
    if not out_path.exists():
        return {
            "success": False,
            "returncode": result.get("returncode"),
            "stderr": (result.get("stderr") or "")[-2000:],
            "stdout": (result.get("stdout") or "")[-1000:],
        }

    graph = _load_graph()
    node_count = len(graph.get("nodes", [])) if graph else 0
    edge_count = len(graph.get("links", [])) if graph else 0
    return {
        "success": True,
        "nodes": node_count,
        "edges": edge_count,
        "output_dir": str(_root() / "graphify-out"),
        "report": str(_root() / "graphify-out" / "GRAPH_REPORT.md"),
    }


@register(
    "graph_query",
    {
        "description": (
            "BFS search the knowledge graph for nodes/edges related to a question. "
            "Returns structural context: callers, callees, import chains. "
            "Better than search_code for: 'what calls X', 'what does Y import', impact analysis. "
            "Run graph_build first if graph.json does not exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or keyword about codebase structure",
                },
                "budget": {
                    "type": "integer",
                    "description": "Max output tokens (default 2000)",
                },
            },
            "required": ["query"],
        },
    },
)
def graph_query(query: str, budget: int = 2000) -> dict:
    if not _graph_json_path().exists():
        return {"error": "graph.json not found. Run graph_build first."}
    graph_arg = str(_graph_json_path())
    result = _run(["query", query, "--graph", graph_arg, "--budget", str(budget)])
    if "error" in result:
        return result
    out = {
        "query": query,
        "output": result.get("stdout", ""),
        "stderr": result.get("stderr", "")[:500] if result.get("stderr") else None,
    }
    warn = _graph_stale_warning()
    if warn:
        out["warning"] = warn
    return out


@register(
    "graph_path",
    {
        "description": (
            "Find the shortest structural path between two nodes in the knowledge graph. "
            "Answers: how does module A depend on B, or how is function X connected to class Y. "
            "Use node names or partial matches from graph_query results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_node": {
                    "type": "string",
                    "description": "Source node name or id",
                },
                "to_node": {
                    "type": "string",
                    "description": "Target node name or id",
                },
            },
            "required": ["from_node", "to_node"],
        },
    },
)
def graph_path(from_node: str, to_node: str) -> dict:
    if not _graph_json_path().exists():
        return {"error": "graph.json not found. Run graph_build first."}
    graph_arg = str(_graph_json_path())
    result = _run(["path", from_node, to_node, "--graph", graph_arg])
    if "error" in result:
        return result
    if result.get("returncode", 0) != 0:
        return {
            "from": from_node,
            "to": to_node,
            "error": result.get("stderr", "No path found"),
        }
    out = {
        "from": from_node,
        "to": to_node,
        "output": result.get("stdout", ""),
    }
    warn = _graph_stale_warning()
    if warn:
        out["warning"] = warn
    return out


@register(
    "graph_context",
    {
        "description": (
            "Get structural context for a symbol: callers, callees, imports, inheritance. "
            "Parsed directly from graph.json — fast, no subprocess. "
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
    edges = graph.get("links", [])

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
                imports.append({"imported_by": s, "relation": rel})
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

    out = {
        "node": node,
        "callers": callers,
        "callees": callees,
        "imports": imports,
        "inherits_from": inherits_from,
        "inherited_by": inherited_by,
        "contains": contains,
        "contained_by": contained_by,
    }
    warn = _graph_stale_warning()
    if warn:
        out["warning"] = warn
    return out
