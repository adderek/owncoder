"""Cross-file taint reachability (Tier-1/2 #24).

Single-window review sees one function at a time, so a vulnerability where untrusted
input enters in file A, is passed through helpers, and reaches a dangerous sink in file B
is invisible to it. This builds a project-wide call graph (tree-sitter functions + the
names each calls) and flags SOURCE→…→SINK call paths — especially ones crossing files.

Deterministic and heuristic: it does not prove data actually flows tainted (no real
dataflow), only that a source-classified function can reach a sink-classified one through
calls. Treat hits as "review this path", confirm with /security review or /security verify.
stdlib + the existing tree-sitter chunker; offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Functions whose body reads untrusted input.
_SOURCE_RE = re.compile(
    r"\b(scanf|sscanf|gets|fgets|fread|read|recv|recvfrom|getenv|argv|os\.Args|"
    r"sys\.argv|input\s*\(|request\.|r\.URL|r\.Form|ReadString|ReadAll|Unmarshal|"
    r"json\.loads|yaml\.load|pickle\.loads)\b")
# Functions whose body performs a dangerous operation.
_SINK_RE = re.compile(
    r"\b(system|popen|execv?[ple]*|exec\.Command|os\.system|subprocess|eval|"
    r"memcpy|memmove|strcpy|strcat|sprintf|vsprintf|alloca|"
    r"Query|Exec|Sprintf.*SELECT|innerHTML)\b")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w{2,})\s*\(")

_MAX_PATHS = 40
_MAX_DEPTH = 8


@dataclass
class Func:
    name: str
    file: str
    line: int
    calls: set = field(default_factory=set)
    is_source: bool = False
    is_sink: bool = False


def _parse_functions(config, files, base) -> dict:
    """name -> Func, via the tree-sitter chunker. Empty if unavailable."""
    funcs: dict = {}
    try:
        from agent.rag.chunker import chunk_file
        cfg = getattr(config, "rag", None)
        if cfg is None:
            return funcs
        for f in files:
            rel = str(f.relative_to(base)) if base in f.parents or base == f.parent else str(f)
            for c in chunk_file(str(f), cfg) or []:
                name = c.get("name")
                nt = c.get("node_type") or ""
                if not name or ("function" not in nt and "method" not in nt):
                    continue
                body = c.get("content") or ""
                calls = {m.group(1) for m in _CALL_RE.finditer(body)}
                fn = Func(name=name, file=rel, line=c.get("start_line", 0), calls=calls,
                          is_source=bool(_SOURCE_RE.search(body)),
                          is_sink=bool(_SINK_RE.search(body)))
                # First definition wins; keep source/sink if any def has it.
                if name in funcs:
                    funcs[name].calls |= calls
                    funcs[name].is_source |= fn.is_source
                    funcs[name].is_sink |= fn.is_sink
                else:
                    funcs[name] = fn
    except Exception:  # noqa: BLE001
        pass
    return funcs


def find_taint_paths(funcs: dict) -> list[dict]:
    """Forward-search from each source for a reachable sink. Returns path dicts."""
    names = set(funcs)
    paths: list[dict] = []
    sources = [n for n, fn in funcs.items() if fn.is_source]
    for src in sources:
        # DFS to first reachable sink (bounded depth), tracking the path.
        stack = [(src, [src])]
        visited = {src}
        while stack:
            node, path = stack.pop()
            fn = funcs.get(node)
            if fn is None:
                continue
            if fn.is_sink and node != src:
                s, k = funcs[src], fn
                paths.append({
                    "source": f"{s.file}:{s.line} {src}",
                    "sink": f"{k.file}:{k.line} {node}",
                    "path": " → ".join(path),
                    "cross_file": s.file != k.file,
                    "hops": len(path) - 1,
                })
                break
            if len(path) > _MAX_DEPTH:
                continue
            for callee in fn.calls & names:
                if callee not in visited:
                    visited.add(callee)
                    stack.append((callee, path + [callee]))
        if len(paths) >= _MAX_PATHS:
            break
    # Cross-file paths first, then by fewest hops.
    paths.sort(key=lambda p: (not p["cross_file"], p["hops"]))
    return paths


def run_taint_command(config, arg: str) -> str:
    """Text handler for `/security taint [path]`."""
    from agent.security.review import _select_files, _resolve_target

    target, err = _resolve_target(config, arg)
    if err:
        return f"Error: {err}"
    files = _select_files(str(target))
    if not files:
        return f"No source files under {target}."
    base = Path(str(target)) if Path(str(target)).is_dir() else Path(str(target)).parent
    funcs = _parse_functions(config, files, base)
    if not funcs:
        return ("No call graph (tree-sitter unavailable for these files). "
                "Taint analysis needs parseable source.")
    paths = find_taint_paths(funcs)
    n_src = sum(1 for f in funcs.values() if f.is_source)
    n_sink = sum(1 for f in funcs.values() if f.is_sink)
    lines = [
        f"# Taint reachability — {Path(str(target)).resolve()}",
        f"- functions: {len(funcs)}  sources: {n_src}  sinks: {n_sink}",
        f"- source→sink paths: {len(paths)}",
        "",
        "> Heuristic call-graph reachability, NOT proven dataflow. Each path is a "
        "'review this' lead — confirm with /security review or /security verify.",
        "",
    ]
    if not paths:
        lines.append("No source→sink call paths found.")
        return "\n".join(lines)
    lines += ["| x-file | hops | source | sink | path |", "|---|---|---|---|---|"]
    for p in paths:
        xf = "yes" if p["cross_file"] else ""
        lines.append(f"| {xf} | {p['hops']} | {p['source']} | {p['sink']} | {p['path']} |")
    return "\n".join(lines)
