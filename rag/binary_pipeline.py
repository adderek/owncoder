"""Hierarchical description of a native binary, driven by REA/Ghidra facts.

Counterpart of asm_pipeline.py (text .asm, LLM-guessed boundaries, groups by
position). Here boundaries, callees and pseudocode are Ghidra's, so the LLM
only does what facts cannot: name and explain.

Phases:
  1. Facts   — procedures, imports, strings, per-function signature + callees.
  2. Describe — callees before callers (Tarjan SCC order), so each prompt
     carries the purpose of every internal function it calls and pseudocode
     with FUN_xxxx placeholders already replaced by inferred names.
  3. Group   — repeated heavy-edge coarsening of the call graph: functions
     that call each other most end up in one group, regardless of address.
     The single top unit describes what the application is for.

Units live in AsmStore like text-asm units: level 0 = function (start_line/
end_line hold the address range), higher levels = groups. Resumable: a
re-run skips described functions; the file record checksum is the binary's
sha256, and a changed binary is re-analysed from scratch.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Callable

from agent.rag.rea_facts import ReaError

if TYPE_CHECKING:
    from agent.config import AsmAnalysisConfig
    from agent.rag.asm_describer import AsmDescriber
    from agent.rag.asm_store import AsmStore
    from agent.rag.embedder import Embedder
    from agent.rag.rea_facts import ReaFacts

logger = logging.getLogger(__name__)

_FUN_RE = re.compile(r"\b(?:FUN|thunk_FUN)_[0-9a-fA-F]{6,16}\b")
_STR_LABEL_RE = re.compile(r"\bs_\w*?_([0-9a-fA-F]{8,16})\b")
_MAX_FACT_IMPORTS = 40
_MAX_LLM_FAILURES = 3
_MAX_STORED_PSEUDO = 20000


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _unit_id(path: str, addr: str) -> str:
    return _sha(f"{path}:{addr}:0")


def callee_first_order(nodes: list[str], succ: dict[str, list[str]]) -> list[str]:
    """Nodes ordered so every callee precedes its callers (cycles: any order).

    Iterative Tarjan; SCCs come out in reverse topological order, which for a
    call graph is exactly leaves first.
    """
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[str] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(succ.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        out.append(w)
                        if w == v:
                            break
    return out


def coarsen(nodes: list[str], weights: dict[tuple[str, str], int],
            max_size: int) -> list[list[str]]:
    """Partition nodes into groups of ≤ max_size, merging along heaviest edges.

    Heavy-edge matching as in multilevel graph partitioning. Nodes left alone
    are packed in input order. Always returns fewer groups than nodes when
    len(nodes) > 1, so repeated calls reach a single root.
    """
    max_size = max(2, max_size)
    pos = {n: i for i, n in enumerate(nodes)}
    parent = {n: n for n in nodes}
    size = {n: 1 for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), _w in sorted(weights.items(), key=lambda kv: (-kv[1], pos[kv[0][0]], pos[kv[0][1]])):
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_size:
            if pos[rb] < pos[ra]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    groups: dict[str, list[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    merged = [g for g in groups.values() if len(g) > 1]
    loners = [g[0] for g in groups.values() if len(g) == 1]
    for i in range(0, len(loners), max_size):
        merged.append(loners[i:i + max_size])
    if len(merged) >= len(nodes) and len(nodes) > 1:
        # No edge could merge anything and loners were packed one per group —
        # cannot happen with max_size ≥ 2, kept as a guard against loops.
        merged = [nodes[i:i + max_size] for i in range(0, len(nodes), max_size)]
    merged.sort(key=lambda g: min(pos[n] for n in g))
    return merged


def referenced_strings(pseudo: str, strings: dict[str, str]) -> list[str]:
    """String values whose Ghidra labels (s_Name_00402000) appear in pseudocode."""
    out: list[str] = []
    for m in _STR_LABEL_RE.finditer(pseudo):
        addr = "0x" + (m.group(1).lstrip("0") or "0").lower()
        val = strings.get(addr)
        if val is not None and val not in out:
            out.append(val)
    return out


class BinaryAnalysisPipeline:
    def __init__(
        self,
        asm_store: "AsmStore",
        embedder: "Embedder | None",
        describer: "AsmDescriber",
        facts: "ReaFacts",
        cfg: "AsmAnalysisConfig",
        interrupt_flag: threading.Event | None = None,
        progress_cb: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._store = asm_store
        self._embedder = embedder
        self._describer = describer
        self._facts = facts
        self._cfg = cfg
        self._interrupt = interrupt_flag or threading.Event()
        self._progress = progress_cb or (lambda _e, _d: None)

    def _embed(self, unit: dict) -> None:
        if self._embedder and unit.get("description"):
            try:
                unit["embedding"] = self._embedder.embed_one(unit["description"])
            except Exception as e:
                logger.warning("embed failed for unit %s: %s", unit["id"], e)

    def analyze(self, path: str, *, force: bool = False, annotate: bool = False,
                max_functions: int | None = None) -> dict:
        opened = self._facts.open(path)
        sha = (opened or {}).get("sha256") or ""
        rec = self._store.get_file_record(path)
        if rec and not force and rec["checksum"] == sha and rec["status"] == "complete":
            root = self._root(path)
            return {"cached": True, "functions": len(self._store.get_units_for_file(path, 0)),
                    "root": root}
        if force or (rec and rec["checksum"] != sha):
            self._store.delete_units_for_file(path)
        self._store.set_file_record(path, sha, status="describing")

        # ── Phase 1: facts ─────────────────────────────────────────────────
        procs, imports = self._facts.procedures()
        strings = self._facts.strings()
        fns: dict[str, dict] = {}
        for p in procs:
            addr = p["addr"]
            info = self._facts.info(addr)
            fns[addr] = {"addr": addr, "name": info.get("name") or p["name"],
                         "signature": info.get("signature"),
                         "length": int(info.get("length") or 1),
                         "callees": self._facts.callees(addr),
                         "id": _unit_id(path, addr)}
        succ = {a: [c for c in f["callees"] if c in fns] for a, f in fns.items()}
        edges: list[tuple[str, str]] = []
        for a, f in fns.items():
            for c in f["callees"]:
                if c in fns:
                    edges.append((f["id"], fns[c]["id"]))
                else:
                    edges.append((f["id"], "import:" + imports.get(c, c)))
        self._store.set_edges(path, edges)
        n_imports = len(set(imports.values()))
        self._progress("facts", {"functions": len(fns), "imports": n_imports})

        # ── Phase 2: describe, callees first ───────────────────────────────
        names: dict[str, str] = {}      # ghidra name → inferred name
        purposes: dict[str, tuple[str, str]] = {}   # addr → (name, purpose)
        for addr, f in fns.items():
            u = self._store.get_unit(f["id"])
            if u and u["status"] == "described":
                names[f["name"]] = u["inferred_name"]
                purposes[addr] = (u["inferred_name"], u["description"] or "")

        order = callee_first_order(sorted(fns, key=lambda a: int(a, 16)), succ)
        todo = [a for a in order if a not in purposes]
        described = unverified = failures = 0
        annotate_error = llm_error = None
        for addr in todo:
            if self._interrupt.is_set() or (max_functions is not None and described >= max_functions):
                break
            f = fns[addr]
            pseudo = self._facts.pseudocode(addr)
            pseudo = _FUN_RE.sub(lambda m: names.get(m.group(0), m.group(0)), pseudo)
            callee_ctx = []
            for c in f["callees"]:
                if c in fns:
                    callee_ctx.append(purposes.get(c) or (fns[c]["name"], "not described yet (recursion)"))
                else:
                    callee_ctx.append((imports.get(c, c), "import"))
            d = self._describer.describe_function(
                {"addr": addr, "name": f["name"], "signature": f["signature"],
                 "pseudocode": pseudo},
                callee_ctx, referenced_strings(pseudo, strings))
            if d.get("failed"):
                failures += 1
                if failures >= _MAX_LLM_FAILURES:
                    llm_error = f"{failures} LLM calls in a row failed; stopped (see agent.log)"
                    break
                continue
            failures = 0
            start = int(addr, 16)
            unit = {
                "id": f["id"], "path": path, "level": 0,
                "start_line": start, "end_line": start + f["length"] - 1,
                "checksum": _sha(pseudo + json.dumps(f["callees"])),
                "status": "described", "addr": addr,
                "description": d["description"], "inferred_name": d["inferred_name"],
                "side_effects": d["side_effects"], "key_patterns": d["key_patterns"],
                "confidence": d["confidence"],
                "calls": json.dumps([n for n, _ in callee_ctx]),
                "evidence": json.dumps({"ghidra_name": f["name"], "signature": f["signature"],
                                        "unverified_apis": d["unverified_apis"],
                                        # What the description was made from —
                                        # the reader checks it against this.
                                        "pseudocode": pseudo[:_MAX_STORED_PSEUDO]}),
            }
            self._embed(unit)
            self._store.upsert_unit(unit)
            names[f["name"]] = d["inferred_name"]
            purposes[addr] = (d["inferred_name"], d["description"])
            described += 1
            unverified += bool(d["unverified_apis"])
            self._progress("described", {"index": len(purposes), "total": len(fns),
                                         "name": d["inferred_name"], "addr": addr})
            if annotate and annotate_error is None:
                try:
                    self._facts.annotate(addr, d["inferred_name"], d["description"])
                except ReaError as e:
                    annotate_error = str(e)

        result = {"functions": len(fns), "imports": n_imports, "described": described,
                  "unverified": unverified, "levels_built": 0, "root": None,
                  "interrupted": self._interrupt.is_set()}
        if annotate_error:
            result["annotate_error"] = annotate_error
        if llm_error:
            result["llm_error"] = llm_error
        if len(purposes) < len(fns):
            result["pending"] = len(fns) - len(purposes)
            return result

        # ── Phase 3: group along the call graph ────────────────────────────
        result["levels_built"] = self._build_hierarchy(path, fns, edges)
        if not self._interrupt.is_set():
            self._store.set_file_record(path, sha, status="complete")
        result["root"] = self._root(path)
        result["interrupted"] = self._interrupt.is_set()
        return result

    def _build_hierarchy(self, path: str, fns: dict[str, dict],
                         edges: list[tuple[str, str]]) -> int:
        units = sorted(self._store.get_units_for_file(path, 0), key=lambda u: u["start_line"])
        owner = {u["id"]: u["id"] for u in units}      # level-0 id → current-level id
        imps: dict[str, set[str]] = {u["id"]: set() for u in units}
        for s, d in edges:
            if d.startswith("import:") and s in imps:
                imps[s].add(d[len("import:"):].rsplit("::", 1)[-1])
        self._link_siblings(units)
        level = 0
        while len(units) > 1 and level < self._cfg.max_levels:
            if self._interrupt.is_set():
                break
            ids = [u["id"] for u in units]
            weights: dict[tuple[str, str], int] = {}
            for s, d in edges:
                a, b = owner.get(s), owner.get(d)
                if a and b and a != b:
                    key = (a, b) if ids.index(a) < ids.index(b) else (b, a)
                    weights[key] = weights.get(key, 0) + 1
            by_id = {u["id"]: u for u in units}
            groups = coarsen(ids, weights, self._cfg.group_size)
            top = len(groups) == 1
            parents = []
            for gi, g in enumerate(groups):
                if self._interrupt.is_set():
                    break
                children = [by_id[i] for i in g]
                pid = _sha(f"{path}:{level + 1}:" + ",".join(sorted(g)))
                cs = _sha("".join(c["checksum"] for c in children))
                group_imps = set().union(*(imps.get(i, set()) for i in g))
                old = self._store.get_unit(pid)
                if old and old["status"] == "grouped" and old["checksum"] == cs:
                    parent = old
                else:
                    facts = ""
                    if group_imps:
                        facts = "Imported APIs used (exact): " + ", ".join(sorted(group_imps)[:_MAX_FACT_IMPORTS])
                    if top:
                        facts += ("\nThese are the top-level parts of one legacy program. "
                                  "In purpose, say what the application is for and who would use it.")
                    parent = {"id": pid, "path": path, "level": level + 1,
                              "start_line": min(c["start_line"] for c in children),
                              "end_line": max(c["end_line"] for c in children),
                              "checksum": cs, "status": "grouped",
                              **self._describer.summarize_group(children, facts)}
                    self._embed(parent)
                    self._store.upsert_unit(parent)
                    self._store.upsert_children(pid, g)
                    for c in children:
                        c["parent_id"] = pid
                        self._store.upsert_unit(c)
                imps[pid] = group_imps
                for leaf, cur in owner.items():
                    if cur in g:
                        owner[leaf] = pid
                parents.append(parent)
                self._progress("grouped", {"level": level + 1, "index": gi + 1, "total": len(groups)})
            self._link_siblings(parents)
            units = parents
            level += 1
        return level

    def _link_siblings(self, units: list[dict]) -> None:
        for i, u in enumerate(units):
            prev_id = units[i - 1]["id"] if i > 0 else None
            next_id = units[i + 1]["id"] if i < len(units) - 1 else None
            if u.get("prev_id") != prev_id or u.get("next_id") != next_id:
                u["prev_id"], u["next_id"] = prev_id, next_id
                self._store.upsert_unit(u)

    def _root(self, path: str) -> dict | None:
        units = self._store.get_units_for_file(path)
        if not units:
            return None
        top = max(u["level"] for u in units)
        tops = [u for u in units if u["level"] == top]
        if len(tops) != 1:
            return None
        r = tops[0]
        return {"id": r["id"], "name": r["inferred_name"], "purpose": r["description"]}
