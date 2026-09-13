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


def kb_corpus_root(config: "Config"):
    """Corpus directory for this project, or None when the KB is off.

    A relative kb.corpus_path resolves against the project working directory, so
    one user-level setting (".agent/kb") gives every project its own corpus.
    """
    from pathlib import Path
    kb_cfg = getattr(config, "kb", None)
    corpus = getattr(kb_cfg, "corpus_path", "") or ""
    if not (getattr(kb_cfg, "enabled", False) and corpus):
        return None
    root = Path(corpus).expanduser()
    if not root.is_absolute():
        root = Path(config.tools.working_dir) / root
    return root


def kb_node_count(config: "Config") -> int | None:
    """Nodes in the configured corpus; None when not configured, not built or unreadable.

    Read-only and cheap: Corpus.open() applies the schema, which would create
    an empty corpus just by asking.
    """
    import sqlite3
    root = kb_corpus_root(config)
    if root is None:
        return None
    db = root / "index.sqlite"
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1)
        try:
            return int(conn.execute("SELECT count(*) FROM nodes").fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return None


def _may_persist() -> bool:
    """The KB is a shared corpus outside the session directory, so an
    off-the-record session must not append to it (agent/security/vault.py).
    Reads stay available — the corpus is a source, not a record."""
    from agent.security import vault
    return vault.persist_allowed()


_KIND_RANK = {"class": 0, "function": 0, "method": 0, "file": 1, "symbol": 2,
              "document": 3, "section": 3, "external": 4}


def _is_test_scope(scope: str) -> bool:
    from pathlib import Path
    parts = Path(scope).parts
    return "tests" in parts or "test" in parts or Path(scope).name.startswith("test_")


def resolve_refs(corpus, ref: str) -> list[dict]:
    """Candidate nodes for a ref, best first: [{id, name, kind, scope}].

    A ref is a node id, a dimensional link (kind=function:name=main), a file
    path (agent/core/prompts.py), Class.method, or a bare symbol name. Names are
    ambiguous by nature; ranking puts definitions before externals and source
    before tests, so the first candidate is what a human would have meant.
    """
    import re
    ref = (ref or "").strip()
    if not ref:
        return []
    conn = corpus.conn

    def describe(ids):
        out = []
        for nid in dict.fromkeys(ids):
            dims = dict(conn.execute(
                "SELECT dim_name, dim_value FROM node_dimensions WHERE node_id = ?", (nid,)).fetchall())
            if dims:
                out.append({"id": nid, "name": dims.get("name", ""), "kind": dims.get("kind", ""),
                            "scope": dims.get("scope", "")})
        return sorted(out, key=lambda n: (_KIND_RANK.get(n["kind"], 5), _is_test_scope(n["scope"]),
                                          len(n["scope"])))

    if re.fullmatch(r"[0-9a-f]{32,}", ref):
        return describe([ref])
    if "=" in ref:
        try:
            node = corpus.get(ref)
            return describe([node.id]) if node else []
        except ValueError:
            from kb.resolver import parse_link, resolve
            result = resolve(conn, parse_link(ref))
            return describe([n.id for n in getattr(result, "nodes", [])])
    if "/" in ref or re.search(r"\.[a-z]{1,5}$", ref):
        ids = [r[0] for r in conn.execute(
            "SELECT node_id FROM locators WHERE scheme = 'file' AND value = ?", (ref,))]
        files = [n for n in describe(ids) if n["kind"] in ("file", "document")]
        if files:
            return files
    owner, _, name = ref.rpartition(".")
    ids = [r[0] for r in conn.execute(
        "SELECT node_id FROM node_dimensions WHERE dim_name = 'name' AND dim_value = ?", (name,))]
    cands = describe(ids)
    if owner:
        owned = {r[0] for r in conn.execute(
            "SELECT e.dst_id FROM edges e JOIN node_dimensions d ON d.node_id = e.src_id "
            "WHERE e.kind = 'method' AND d.dim_name = 'name' AND d.dim_value = ?", (owner.split(".")[-1],))}
        cands = [c for c in cands if c["id"] in owned] or cands
    return cands


def _one(corpus, ref: str) -> tuple[str | None, dict]:
    """Best node id for a ref, plus what the caller should be told about ambiguity."""
    cands = resolve_refs(corpus, ref)
    if not cands:
        return None, {"error": f"no KB node matches {ref!r}"}
    extra = {"resolved": cands[0]}
    if len(cands) > 1:
        extra["also_matched"] = cands[1:6]
    return cands[0]["id"], extra


def _get_corpus():
    global _corpus
    if _corpus is not None:
        return _corpus
    if _config is None:
        raise RuntimeError("kb tools not configured — call setup() first")
    corpus_root = kb_corpus_root(_config)
    if corpus_root is None:
        raise RuntimeError("kb.corpus_path not set in config")
    from kb.api import Corpus
    _corpus = Corpus.open(corpus_root)
    return _corpus


@register("kb_search", {
    "description": "Full-text search over KB corpus. Returns nodes (id, name, kind, scope, description). Use to find functions, modules, concepts.",
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
    "description": "Get KB node by id or dimensional-link (e.g. kind=function:name=main). Returns dims, description, locators.",
    "parameters": {
        "type": "object",
        "properties": {
            "ref": {
                "type": "string",
                "description": "Symbol name (Class.method ok), file path, node id, or dim-link (kind=function:name=main)",
            },
        },
        "required": ["ref"],
    },
})
def kb_get(ref: str) -> str:
    try:
        corpus = _get_corpus()
        node_id, extra = _one(corpus, ref)
        node = corpus.get(node_id) if node_id else None
        if node is None:
            return json.dumps({"error": f"not found: {ref}"})
        return json.dumps({
            **({"also_matched": extra["also_matched"]} if "also_matched" in extra else {}),
            "id": node.id,
            "name": node.inferred_name_base,
            "dims": node.dims,
            "description": node.description_base,
            "completeness": node.completeness,
            "data_grade": node.data_grade,
            "priority": node.priority,
            "locators": [{"scheme": l.scheme, "value": l.value, **({"at": l.template} if l.template else {})}
                         for l in node.locators],
        }, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_deps", {
    "description": "Return callees (deps) of a node. depth=1 direct only, >1 transitive. Use to see what a function calls.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Symbol name (Class.method ok), file path or node id",
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
        resolved, extra = _one(corpus, node_id)
        if resolved is None:
            return json.dumps(extra)
        result = corpus.deps(resolved, kind=kind, depth=depth)
        return json.dumps({
            **extra,
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
    "description": "Return callers (inverse deps) of a node. depth=1 direct only, >1 transitive. Use to find what calls a function.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Symbol name (Class.method ok), file path or node id",
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
        resolved, extra = _one(corpus, node_id)
        if resolved is None:
            return json.dumps(extra)
        result = corpus.callers(resolved, kind=kind, depth=depth)
        return json.dumps({
            **extra,
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
    "description": "Attach observation/note to a KB node. Returns note id. Use to record findings, hypotheses, context.",
    "parameters": {
        "type": "object",
        "properties": {
            "attach_to": {
                "type": "string",
                "description": "Where the note belongs: symbol name(s) or file path(s), comma-separated; node ids work too",
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
    if not _may_persist():
        return json.dumps({"error": "off-the-record session: KB not written"})
    try:
        corpus = _get_corpus()
        targets, unresolved = [], []
        for ref in [r for r in (attach_to or "").split(",") if r.strip()]:
            node_id, extra = _one(corpus, ref)
            (targets.append(extra["resolved"]) if node_id else unresolved.append(ref.strip()))
        if unresolved or not targets:
            return json.dumps({"error": f"no KB node for: {', '.join(unresolved) or attach_to!r}",
                               "hint": "use a symbol name or file path that exists in the code"})
        note_id = corpus.add_note([t["id"] for t in targets], body, kind=kind)
        return json.dumps({"note_id": note_id, "attached": targets}, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@register("kb_propose_description", {
    "description": "Propose description for a KB node. Stored as DB override (no YAML change). Use to annotate nodes.",
    "parameters": {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "Symbol name (Class.method ok), file path or node id",
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
    if not _may_persist():
        return json.dumps({"error": "off-the-record session: KB not written"})
    try:
        corpus = _get_corpus()
        corpus.propose_description(node_id, text)
        return json.dumps({"ok": True, "node_id": node_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
