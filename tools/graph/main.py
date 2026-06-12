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
_graph_key: tuple | None = None
_stale_cache: tuple[float, str | None] | None = None  # (checked_at, warning)
_STALE_TTL = 30.0


def setup(config, data_provider=None) -> None:
    global _config
    _config = config
    from agent.tools.graph import asm_export
    asm_export.setup(data_provider)


def invalidate_caches() -> None:
    global _graph_cache, _graph_key, _stale_cache
    _graph_cache = None
    _graph_key = None
    _stale_cache = None


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


def _asm_graph_path() -> Path:
    return _root() / "graphify-out" / "asm-graph.json"


def _merged_graph_path() -> Path:
    return _root() / "graphify-out" / "graph-merged.json"


def _load_graph() -> dict | None:
    """Load graph.json merged with asm-graph.json (either may be absent)."""
    global _graph_cache, _graph_key
    paths = [p for p in (_graph_json_path(), _asm_graph_path()) if p.exists()]
    if not paths:
        return None
    key = tuple((str(p), p.stat().st_mtime) for p in paths)
    if _graph_cache is None or key != _graph_key:
        nodes: list = []
        links: list = []
        for p in paths:
            try:
                with p.open() as f:
                    g = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            nodes.extend(g.get("nodes", []))
            links.extend(g.get("links", []))
        _graph_cache = {"nodes": nodes, "links": links}
        _graph_key = key
    return _graph_cache


def _query_graph_arg() -> str | None:
    """Graph file to pass to the graphify CLI; merges source+asm graphs lazily."""
    gp, ap = _graph_json_path(), _asm_graph_path()
    if not gp.exists() and not ap.exists():
        return None
    if not ap.exists():
        return str(gp)
    if not gp.exists():
        return str(ap)
    mp = _merged_graph_path()
    src_mtime = max(gp.stat().st_mtime, ap.stat().st_mtime)
    if not mp.exists() or mp.stat().st_mtime < src_mtime:
        merged = _load_graph()
        with mp.open("w") as f:
            json.dump(merged, f)
    return str(mp)


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
    """Return a warning string if graph.json is older than newest source file, else None.

    Result is cached for _STALE_TTL seconds — the source-tree mtime walk is
    O(files) and this runs on every graph_query/graph_path/graph_context call.
    """
    global _stale_cache
    import time
    now = time.time()
    if _stale_cache is not None and now - _stale_cache[0] < _STALE_TTL:
        return _stale_cache[1]
    p = _graph_json_path()
    warn = None
    if p.exists():
        graph_mtime = p.stat().st_mtime
        source_mtime = _newest_source_mtime(_root())
        if source_mtime > graph_mtime:
            warn = "graph may be stale (source files changed since last graph_build). Run graph_build to refresh."
    _stale_cache = (now, warn)
    return warn


@register(
    "graph_status",
    {
        "description": (
            "Check whether the knowledge graph exists and is up to date. "
            "Call this before using graph_query/graph_path/graph_context to avoid stale results. "
            "Returns: exists, node/edge counts, age, stale flag, and graphify binary availability."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)
def graph_status(**_kwargs) -> dict:
    import datetime
    p = _graph_json_path()
    asm_p = _asm_graph_path()
    bin_ok = _graphify_bin() is not None

    if not p.exists() and not asm_p.exists():
        return {
            "exists": False,
            "stale": None,
            "graphify_installed": bin_ok,
            "hint": "Run graph_build() to create the graph." if bin_ok else "Install graphifyy first: agent/.venv/bin/pip install graphifyy",
        }

    graph = _load_graph()
    node_count = len(graph.get("nodes", [])) if graph else 0
    edge_count = len(graph.get("links", [])) if graph else 0

    out = {
        "exists": True,
        "nodes": node_count,
        "edges": edge_count,
        "asm_graph": asm_p.exists(),
        "graphify_installed": bin_ok,
    }

    if p.exists():
        graph_mtime = p.stat().st_mtime
        source_mtime = _newest_source_mtime(_root())
        stale = source_mtime > graph_mtime
        out.update({
            "stale": stale,
            "built": datetime.datetime.fromtimestamp(graph_mtime).isoformat(timespec="seconds"),
            "age_seconds": int(__import__("time").time() - graph_mtime),
            "hint": "Run graph_build() to refresh." if stale else "Graph is current.",
        })
    else:
        out.update({
            "stale": None,
            "hint": "Only asm-graph.json present. Run graph_build() to add source code graph.",
        })
    return out


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
    invalidate_caches()

    if _graphify_bin() is None:
        return {"error": "graphify not found. Install: agent/.venv/bin/pip install graphifyy"}

    target = str(Path(path).resolve() if path else _root())

    # `update` processes code only, no LLM/API key needed.
    # --force only on first run (incremental update when graph.json exists).
    cmd = ["update", target, "--no-cluster"]
    if not _graph_json_path().exists():
        cmd.append("--force")
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
    graph_arg = _query_graph_arg()
    if graph_arg is None:
        return {"error": "No graph found. Run graph_build (and/or graph_build_asm) first."}
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
    graph_arg = _query_graph_arg()
    if graph_arg is None:
        return {"error": "No graph found. Run graph_build (and/or graph_build_asm) first."}
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
        return {"error": "No graph found. Run graph_build (and/or graph_build_asm) first."}

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
    if len(matched) > 1:
        out["also_matched"] = [n["id"] for n in matched[1:11]]
    warn = _graph_stale_warning()
    if warn:
        out["warning"] = warn
    return out
