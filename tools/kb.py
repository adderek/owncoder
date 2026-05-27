"""Knowledge-base agent tools (M5.2-M5.3).

Tools are only registered when kb.enabled = true in config.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config.models import Config

logger = logging.getLogger(__name__)

_config = None
_corpus = None


def setup(config: "Config") -> None:
    global _config, _corpus
    _config = config
    _corpus = None  # lazy-open on first call


def _get_corpus():
    global _corpus
    if _corpus is not None:
        return _corpus
    if _config is None:
        raise RuntimeError("kb tools not configured — call setup() first")
    corpus_path = getattr(_config.kb, "corpus_path", "")
    if not corpus_path:
        raise RuntimeError("kb.corpus_path not set in config")
    from kb.api import Corpus
    _corpus = Corpus.open(corpus_path)
    return _corpus


@register("kb_search", {
    "description": (
        "Search the knowledge-base corpus by full-text query. "
        "Returns matching nodes with id, name, kind, scope, and description snippet. "
        "Use to find functions, modules, or concepts by keyword."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms (space-separated; prefix with - to exclude)",
            },
            "kind": {
                "type": "string",
                "description": "Filter by node kind (e.g. 'function', 'module')",
            },
            "scope": {
                "type": "string",
                "description": "Filter by scope dimension",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    },
})
def kb_search(query: str, kind: str | None = None, scope: str | None = None, limit: int = 10) -> str:
    try:
        corpus = _get_corpus()
        nodes = corpus.search(query, kind=kind, scope=scope, limit=limit)
        results = [
            {
                "id": n.id,
                "name": n.inferred_name_base,
                "kind": n.dims.get("kind", ""),
                "scope": n.dims.get("scope", ""),
                "snippet": (n.description_base or "")[:120],
            }
            for n in nodes
        ]
        return json.dumps({"nodes": results, "count": len(results)}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_get", {
    "description": (
        "Get a single knowledge-base node by id or dimensional-link. "
        "Returns full node data including dims, description, locators. "
        "Dimensional-link syntax: kind=function:name=main"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Node id (hex) or dimensional-link (dim=val:dim=val)",
            },
        },
        "required": ["ref"],
    },
})
def kb_get(ref: str) -> str:
    try:
        corpus = _get_corpus()
        node = corpus.get(ref)
        if node is None:
            return json.dumps({"error": f"not found: {ref}"})
        return json.dumps({
            "id": node.id,
            "name": node.inferred_name_base,
            "dims": node.dims,
            "description": node.description_base,
            "completeness": node.completeness,
            "data_grade": node.data_grade,
            "priority": node.priority,
            "locators": [{"scheme": l.scheme, "value": l.value} for l in node.locators],
        }, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_deps", {
    "description": (
        "Return callees (dependencies) of a node. "
        "depth=1 returns only direct deps; depth>1 also returns transitive deps. "
        "Use to understand what a function calls."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Node id (hex)",
            },
            "kind": {
                "type": "string",
                "description": "Edge kind to follow (default: calls)",
                "default": "calls",
            },
            "depth": {
                "type": "integer",
                "description": "BFS depth (1=direct only, default 1)",
                "default": 1,
            },
        },
        "required": ["node_id"],
    },
})
def kb_deps(node_id: str, kind: str = "calls", depth: int = 1) -> str:
    try:
        corpus = _get_corpus()
        result = corpus.deps(node_id, kind=kind, depth=depth)
        return json.dumps({
            "direct": [
                {"id": n.id, "name": n.inferred_name_base, "edge_kind": e.kind}
                for e, n in result.direct
            ],
            "inherited": [
                {"id": n.id, "name": n.inferred_name_base, "edge_kind": e.kind}
                for e, n in result.inherited
            ],
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_callers", {
    "description": (
        "Return callers (inverse dependencies) of a node. "
        "depth=1 returns only direct callers; depth>1 also returns transitive callers. "
        "Use to find what calls a given function."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Node id (hex)",
            },
            "kind": {
                "type": "string",
                "description": "Edge kind to follow (default: calls)",
                "default": "calls",
            },
            "depth": {
                "type": "integer",
                "description": "BFS depth (1=direct only, default 1)",
                "default": 1,
            },
        },
        "required": ["node_id"],
    },
})
def kb_callers(node_id: str, kind: str = "calls", depth: int = 1) -> str:
    try:
        corpus = _get_corpus()
        result = corpus.callers(node_id, kind=kind, depth=depth)
        return json.dumps({
            "direct": [
                {"id": n.id, "name": n.inferred_name_base, "edge_kind": e.kind}
                for e, n in result.direct
            ],
            "inherited": [
                {"id": n.id, "name": n.inferred_name_base, "edge_kind": e.kind}
                for e, n in result.inherited
            ],
        }, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_add_note", {
    "description": (
        "Add an observation or note attached to a knowledge-base node. "
        "Returns the note id. "
        "Use to record findings, hypotheses, or context about a node."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "attach_to": {
                "type": "string",
                "description": "Node id to attach note to",
            },
            "body": {
                "type": "string",
                "description": "Note content (Markdown supported)",
            },
            "kind": {
                "type": "string",
                "description": "Note kind: 'observation', 'hypothesis', 'todo' (default: observation)",
                "default": "observation",
            },
        },
        "required": ["attach_to", "body"],
    },
})
def kb_add_note(attach_to: str, body: str, kind: str = "observation") -> str:
    try:
        corpus = _get_corpus()
        note_id = corpus.add_note(attach_to, body, kind=kind)
        return json.dumps({"note_id": note_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_propose_description", {
    "description": (
        "Propose a new description for a knowledge-base node. "
        "Stored as an override in the DB (does not modify YAML source). "
        "Use to annotate nodes with LLM-generated descriptions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Node id (hex)",
            },
            "text": {
                "type": "string",
                "description": "Proposed description text",
            },
        },
        "required": ["node_id", "text"],
    },
})
def kb_propose_description(node_id: str, text: str) -> str:
    try:
        corpus = _get_corpus()
        corpus.propose_description(node_id, text)
        return json.dumps({"ok": True, "node_id": node_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
