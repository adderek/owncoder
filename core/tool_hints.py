"""Just-in-time tool-routing hints.

The system prompt states which tool answers which question, but it is read once
per session and competes with everything else in context. A hint attached to a
tool *result* arrives at the moment the model is choosing its next call, which
is when the routing decision is actually made.

Each rule fires at most once per session (per key where a key applies), carries
one short line, and never changes the result itself.
"""
from __future__ import annotations

import re

# Session state. Reset via reset_tool_hints() on session start.
_read_counts: dict[str, int] = {}
# Compaction count at the time each path was last read.
_read_compactions: dict[str, int] = {}
_fired: set[str] = set()

# read_file calls on one path before the "you are hunting, not reading" hint.
_READ_HINT_THRESHOLD = 3
# Lines above which rewriting a whole existing file is worth questioning.
_REWRITE_HINT_MIN_LINES = 200

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
# "def foo", "class Foo", "function foo(" — a structural question typed as text.
_STRUCTURAL_PATTERN_RE = re.compile(
    r"^\s*(?:def|class|function|func|fn|interface|struct)\s+\w+", re.IGNORECASE
)


def reset_tool_hints() -> None:
    """Clear per-session state (session start, tests)."""
    _read_counts.clear()
    _read_compactions.clear()
    _fired.clear()


def _compacted_since_read(path: str) -> bool:
    """True if compaction ran after this path was last read, so its content is
    no longer in context even though the model saw it earlier."""
    try:
        from agent.core import context_state
    except Exception:
        return False
    snap = context_state.current()
    if snap is None or not snap.compactions:
        return False
    return _read_compactions.get(path, -1) < snap.compactions


def _stamp_read(path: str) -> None:
    try:
        from agent.core import context_state
        snap = context_state.current()
    except Exception:
        return
    _read_compactions[path] = snap.compactions if snap else 0


def read_count(path: str) -> int:
    """How many times read_file has served *path* this session."""
    return _read_counts.get(path, 0)


def _once(key: str) -> bool:
    if key in _fired:
        return False
    _fired.add(key)
    return True


def _is_empty(result: dict) -> bool:
    if result.get("count") == 0:
        return True
    for field in ("results", "matches", "hits", "nodes"):
        if field in result:
            return not result[field]
    return False


def tool_hints(tool_name: str, args: dict, result: dict) -> list[str]:
    """Routing hints for a completed call. Advisory only; never raises."""
    if not isinstance(result, dict) or result.get("error"):
        return []
    hints: list[str] = []

    if tool_name == "read_file":
        path = str(args.get("path") or "")
        if path:
            already_read = _read_counts.get(path, 0) > 0
            _read_counts[path] = _read_counts.get(path, 0) + 1
            n = _read_counts[path]
            stale = already_read and _compacted_since_read(path)
            _stamp_read(path)
            if stale and _once(f"recompact:{path}"):
                hints.append(
                    f"[tool-hint] {path} was read before and elided by compaction. Reading it "
                    f"whole again will be elided again — that is a loop. Read only the range you "
                    f"need, or call find_symbol to get the line directly."
                )
            if n >= _READ_HINT_THRESHOLD and _once(f"read:{path}"):
                hints.append(
                    f"[tool-hint] read_file called {n}× on {path}. If you are hunting for a symbol, "
                    f"grep_code(pattern='<name>', path='{path}') returns the line directly; "
                    f"the outline in a truncated read lists the landmarks."
                )

    elif tool_name == "search_code":
        query = str(args.get("query") or "").strip()
        if query and _IDENTIFIER_RE.match(query) and _once("search:identifier"):
            hints.append(
                "[tool-hint] That query is an exact identifier. search_code is fuzzy and can "
                "miss it; grep_code searches raw files and cannot."
            )
        elif _is_empty(result) and _once("search:empty"):
            hints.append(
                "[tool-hint] No index hits. This does not mean the code is absent — the index "
                "may not cover it. Confirm with grep_code before concluding anything."
            )

    elif tool_name == "grep_code":
        pattern = str(args.get("pattern") or "")
        if _STRUCTURAL_PATTERN_RE.match(pattern) and _once("grep:structural"):
            hints.append(
                "[tool-hint] Looking for a definition. graph_context / kb_search answer "
                "'where is X defined, who calls it, what does it depend on' in one call; "
                "grep only finds text that happens to match."
            )
        elif _is_empty(result) and _once("grep:empty"):
            hints.append(
                "[tool-hint] No matches. Try a shorter pattern or fixed_string=true before "
                "widening the search by hand — and search_code for a concept rather than a name."
            )

    elif tool_name == "write_file":
        path = str(args.get("path") or "")
        existing = result.get("existing_lines")
        if existing is None:
            existing = (result.get("replaced_lines") or 0)
        if path and existing and existing >= _REWRITE_HINT_MIN_LINES and _once(f"rewrite:{path}"):
            hints.append(
                f"[tool-hint] Rewrote {path} whole ({existing} lines). edit_file changes just the "
                f"affected region, so unrelated parts of the file cannot be lost in a rewrite."
            )

    return hints
