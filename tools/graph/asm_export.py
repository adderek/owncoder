"""Export analyze_asm results into knowledge-graph format.

Bridges the two systems: analyze_asm stores per-routine `calls` lists and a
summarization hierarchy in AsmStore; this module materializes them as
graph.json-compatible nodes/links so graph_query/graph_path/graph_context
work on analyzed binaries.

Resolution: call targets are matched case-insensitively against level-0
inferred_names. Unresolved targets (raw addresses, external labels) become
shared `asm_ext:` nodes so they remain queryable.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.data_provider import DataProviderProtocol

logger = logging.getLogger(__name__)

_data_provider: "DataProviderProtocol | None" = None


def setup(data_provider) -> None:
    global _data_provider
    _data_provider = data_provider


def _node_id(unit: dict) -> str:
    return f"asm:{unit['id']}"


def build_asm_graph(asm_store) -> dict:
    """Build {"nodes": [...], "links": [...]} from all units in the asm store."""
    units = asm_store.get_all_units()

    name_map: dict[str, str] = {}
    for u in units:
        name = u.get("inferred_name")
        if name and u["level"] == 0:
            name_map.setdefault(name.lower(), _node_id(u))

    nodes: list[dict] = []
    links: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    ext_nodes: dict[str, str] = {}

    def add_edge(source: str, target: str, relation: str) -> None:
        key = (source, target, relation)
        if key not in seen_edges:
            seen_edges.add(key)
            links.append({"source": source, "target": target, "relation": relation})

    for u in units:
        nid = _node_id(u)
        nodes.append({
            "id": nid,
            "label": u.get("inferred_name") or f"unit_{u['id'][:8]}",
            "source_file": u["path"],
            "type": "asm_group" if u["level"] else "asm_function",
            "level": u["level"],
            "start_line": u["start_line"],
            "end_line": u["end_line"],
            "description": u.get("description"),
        })

        if u.get("parent_id"):
            add_edge(f"asm:{u['parent_id']}", nid, "contains")

        raw_calls = u.get("calls")
        if not raw_calls or u["level"] != 0:
            continue
        try:
            targets = json.loads(raw_calls)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Unparseable calls field on unit %s", u["id"])
            continue
        if not isinstance(targets, list):
            continue
        for t in targets:
            t = str(t).strip()
            if not t:
                continue
            tid = name_map.get(t.lower())
            if tid == nid:
                continue  # self-call; recursion adds noise, not structure
            if tid is None:
                tid = ext_nodes.setdefault(t.lower(), f"asm_ext:{t}")
            add_edge(nid, tid, "calls")

    for label, eid in ext_nodes.items():
        nodes.append({
            "id": eid,
            "label": label,
            "source_file": "",
            "type": "asm_external",
        })

    return {"nodes": nodes, "links": links}


@register(
    "graph_build_asm",
    {
        "description": (
            "Export analyze_asm results (routines, call lists, summary hierarchy) "
            "into the knowledge graph as graphify-out/asm-graph.json. "
            "Afterwards graph_query/graph_path/graph_context cover analyzed binaries too. "
            "Run analyze_asm on files first; re-run this after new analyses."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)
def graph_build_asm(**_kwargs) -> dict:
    from agent.tools.graph import main as graph_main

    asm_store = _data_provider.get_asm_store() if _data_provider else None
    if asm_store is None:
        return {"error": "ASM store unavailable. Run analyze_asm first (it initializes the store)."}

    graph = build_asm_graph(asm_store)
    if not graph["nodes"]:
        return {"error": "ASM store is empty. Run analyze_asm on a file first."}

    out_path = graph_main._asm_graph_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(graph, f)
    graph_main.invalidate_caches()

    functions = sum(1 for n in graph["nodes"] if n.get("type") == "asm_function")
    external = sum(1 for n in graph["nodes"] if n.get("type") == "asm_external")
    return {
        "success": True,
        "nodes": len(graph["nodes"]),
        "edges": len(graph["links"]),
        "functions": functions,
        "external_targets": external,
        "output": str(out_path),
    }
