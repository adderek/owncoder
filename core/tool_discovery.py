"""Progressive tool disclosure.

Problem: the agent registers ~70 tools. Sending every schema each turn burns a
large, fixed slice of the context window and drowns a weak local model in
choices, so it defaults to a couple of familiar tools and ignores the rest.

Solution (opt-in, `config.tool_discovery.enabled`):
- Each turn the model only receives the CORE tool schemas (the handful needed
  constantly) plus the `find_tools` meta-tool.
- A compact, grouped catalog of every other tool — name + one-line purpose,
  bucketed by category with a "use when…" hint — is injected into the system
  prompt so the model stays AWARE of what exists and when to reach for it.
- When the model needs a non-core tool it calls `find_tools("<keywords>")`;
  matches are activated (their full schemas appear on the next iteration) and
  stay active for the rest of the session.

This module is pure/stateless except for the process-lifetime ACTIVE set that
`find_tools` mutates and the turn loop reads.
"""
from __future__ import annotations

import re
from typing import Iterable

# ── Core set: always exposed with full schema when discovery is on ──────────
# The tools a coding agent needs essentially every turn. Everything else is
# discovered on demand. `find_tools` itself is always core (the entry point).
CORE_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "edit_file",
    "search_code",
    "grep_code",
    "list_files",
    "run_argv",
    "run_command",
    "save_note",
    "find_tools",
})

# ── Categories: (label, "use when" hint, prefix/name matchers) ──────────────
# Ordering is the display order in the catalog. A tool lands in the FIRST
# category it matches; anything unmatched falls into "other".
# A matcher is either a prefix string (endswith-aware) or an exact name set.
_CATEGORIES: list[tuple[str, str, tuple[str, ...], frozenset[str]]] = [
    ("read & search code", "find/inspect code before editing",
     (), frozenset({"read_file", "search_code", "grep_code", "list_files",
                    "search_archive", "retrieve_output", "project_file_stats"})),
    ("edit code", "apply changes to files",
     (), frozenset({"edit_file", "write_file", "replace_symbol", "undo_file"})),
    ("run commands", "execute shell/build/test commands",
     (), frozenset({"run_argv", "run_command"})),
    ("git", "history, blame, diffs, related files",
     ("git_",), frozenset()),
    ("call graph (structure)", "who calls / what depends on / where defined",
     ("graph_",), frozenset()),
    ("knowledge base", "curated symbol facts, callers, deps",
     ("kb_",), frozenset()),
    ("session memory & recall", "past decisions, prior intent, what was tried",
     ("recall_",), frozenset({"save_note", "rate_session"})),
    ("planning", "multi-step plans, steps, dependencies",
     ("plan_",), frozenset({"create_plan", "set_current_plan", "next_step",
                            "get_step_brief", "complete_step", "snapshot_step",
                            "revert_step", "mark_done"})),
    ("skills", "reusable procedures saved across sessions",
     ("skill",), frozenset({"search_skills", "load_skill", "save_skill",
                            "skill_history", "rollback_skill"})),
    ("checkpoints", "session-wide multi-file rollback",
     (), frozenset({"create_checkpoint", "list_checkpoints", "rollback_checkpoint"})),
    ("indexing", "build/refresh code index, graph, asm analysis",
     ("graph_build", "index_"), frozenset({"index_code", "analyze_asm"})),
    ("security audit", "scan code for vulnerabilities",
     (), frozenset({"security_audit"})),
    ("web", "search/fetch external information",
     ("web_",), frozenset({"ask_internet"})),
    ("ideas & notes", "capture out-of-scope ideas/bugs",
     (), frozenset({"submit_idea"})),
    ("agents & parallel", "delegate work to sub-agents",
     (), frozenset({"spawn_agents", "consult_crows"})),
    ("access & review", "request path grants, reviews, feedback",
     ("request_",), frozenset()),
    ("turn control", "signal completion / ask user / report blocked",
     (), frozenset({"ask_user", "mark_done", "blocked", "report_blocking_issue"})),
]


def categorize(name: str) -> tuple[str, str]:
    """Return (category_label, use_when_hint) for a tool name.

    An exact name match always wins. Otherwise the LONGEST matching prefix wins,
    so a specific prefix (e.g. ``graph_build`` → indexing) beats a broader one
    (``graph_`` → call graph) regardless of category order.
    """
    best: tuple[int, str, str] | None = None  # (prefix_len, label, hint)
    for label, hint, prefixes, names in _CATEGORIES:
        if name in names:
            return label, hint
        for p in prefixes:
            if name.startswith(p) and (best is None or len(p) > best[0]):
                best = (len(p), label, hint)
    if best is not None:
        return best[1], best[2]
    return "other", "specialized / situational"


def _first_sentence(desc: str, limit: int = 90) -> str:
    """First sentence of a tool description, trimmed for the catalog line."""
    desc = (desc or "").strip().replace("\n", " ")
    desc = re.sub(r"\s+", " ", desc)
    # Stop at the first sentence boundary.
    m = re.search(r"[.!?](\s|$)", desc)
    if m:
        desc = desc[: m.start()]
    if len(desc) > limit:
        desc = desc[: limit - 1].rstrip() + "…"
    return desc


def core_names(config) -> frozenset[str]:
    extra = getattr(getattr(config, "tool_discovery", None), "extra_core", None) or []
    return CORE_TOOLS | frozenset(extra)


def select_schemas(schemas: list[dict], active: Iterable[str], config) -> list[dict]:
    """Return the schemas to send to the model: core ∪ active.

    Unknown active names are ignored (a stale activation is harmless).
    """
    keep = core_names(config) | frozenset(active)
    return [s for s in schemas if s.get("function", {}).get("name") in keep]


def render_catalog(schemas: list[dict], config) -> str:
    """Render the compact grouped catalog block for the system prompt.

    Lists only NON-core tools (core ones already have full schemas). Each line
    is `name — first sentence`, grouped under a category + "use when" hint.
    """
    core = core_names(config)
    buckets: dict[str, list[str]] = {}
    descs: dict[str, str] = {}
    for s in schemas:
        fn = s.get("function", {})
        name = fn.get("name")
        if not name or name in core:
            continue
        label, _hint = categorize(name)
        buckets.setdefault(label, []).append(name)
        descs[name] = _first_sentence(fn.get("description", ""))

    if not buckets:
        return ""

    order = [label for label, *_ in _CATEGORIES] + ["other"]
    lines = [
        "# Tool catalog (full schemas load on demand)",
        "",
        "You see full schemas for the core tools only. To use any tool below, "
        "first call find_tools(\"<keywords>\") — its matches become callable on "
        "your next step. Pick by category:",
        "",
    ]
    for label in order:
        names = buckets.get(label)
        if not names:
            continue
        hint = next((h for lb, h, *_ in _CATEGORIES if lb == label), "")
        lines.append(f"## {label}" + (f" — {hint}" if hint else ""))
        for name in sorted(names):
            d = descs.get(name, "")
            lines.append(f"- {name}" + (f" — {d}" if d else ""))
        lines.append("")
    return "\n".join(lines).rstrip()


# ── find_tools ranking ──────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "who", "what", "which", "how", "when", "where", "why", "this", "that",
    "the", "a", "an", "to", "of", "is", "are", "in", "on", "for", "and", "or",
    "do", "does", "my", "it", "with", "by", "from", "get", "find", "use",
    "need", "i", "me", "we", "you", "function", "file", "code",
})


def find_matches(schemas: list[dict], query: str, config, max_results: int) -> list[dict]:
    """Rank tools against a free-text query. Returns [{name, description, category}].

    Scoring: name hit > category hit > description hit, summed over query tokens.
    Core tools are excluded (already available). Deterministic, no LLM.
    """
    core = core_names(config)
    tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower())
              if t and t not in _STOPWORDS]
    # If the query was ALL stopwords, fall back to the raw tokens so we still
    # return something rather than nothing.
    if not tokens:
        tokens = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if t]
    if not tokens:
        return []
    scored: list[tuple[int, str, dict]] = []
    for s in schemas:
        fn = s.get("function", {})
        name = fn.get("name")
        if not name or name in core:
            continue
        nm = name.lower()
        desc = (fn.get("description", "") or "").lower()
        label, _hint = categorize(name)
        cat = label.lower()
        score = 0
        for t in tokens:
            if t in nm:
                score += 5
            if t in cat:
                score += 3
            if t in desc:
                score += 1
        if score > 0:
            scored.append((score, name, {
                "name": name,
                "description": _first_sentence(fn.get("description", ""), limit=140),
                "category": label,
            }))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [entry for _score, _name, entry in scored[:max_results]]


# ── Session-scoped active set (mutated by find_tools, read by the turn loop) ─
_ACTIVE: set[str] = set()


def reset_active() -> None:
    _ACTIVE.clear()


def activate(names: Iterable[str]) -> None:
    _ACTIVE.update(names)


def active_names() -> frozenset[str]:
    return frozenset(_ACTIVE)
