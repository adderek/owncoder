"""Language-agnostic file outline: the landmarks a model actually navigates by.

Used to answer "where in this file is X" without paging the whole file: a
truncated read returns the outline of the *whole* file alongside its first
window, and edit_file reports it when an anchor is not found.

Deliberately regex-based, not a parser: it must work on a half-written file,
on a language with no parser here, and cost nothing.
"""
from __future__ import annotations

import re

# (kind, pattern) — first capture group is the name.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Section banners: "// === LIGHTING ===", "# --- DOMKI ---", "/* == x == */".
    # Models anchor on these far more than on symbols in markup/script files.
    ("section", re.compile(r"^(?:/[/*]+|#+|--|;+|\*)\s*[-=~*_]{2,}\s*(.+?)\s*[-=~*_]{2,}")),
    ("class", re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)")),
    ("def", re.compile(r"^(?:async\s+)?def\s+(\w+)")),
    ("func", re.compile(r"^(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)")),
    ("func", re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)")),          # go
    ("func", re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")),      # rust
    # const foo = (a) => / const foo = function / let foo = async (
    ("func", re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\w+\s*=>)")),
    ("type", re.compile(r"^(?:export\s+)?(?:type|interface|struct|enum)\s+(\w+)")),
    ("heading", re.compile(r"^(#{1,4})\s+(.+)")),                      # markdown
]

_MAX_NAME = 80


def outline(text: str, max_entries: int = 20) -> list[dict]:
    """File landmarks as [{line, kind, name, indent, text}], in file order.

    Truncation appends a single {"kind": "..."} entry naming how many were
    dropped, matching what edit_file has always returned.
    """
    out: list[dict] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        for kind, pat in _PATTERNS:
            m = pat.match(stripped)
            if not m:
                continue
            name = (m.group(2) if kind == "heading" else m.group(1)).strip()
            if not name:
                break
            out.append({
                "line": lineno,
                "kind": kind,
                "name": name[:_MAX_NAME],
                "indent": len(line) - len(line.lstrip()),
                "text": stripped[:_MAX_NAME],
            })
            break
    if len(out) > max_entries:
        dropped = len(out) - max_entries
        out = out[:max_entries]
        out.append({"line": -1, "kind": "...", "name": f"... ({dropped} more entries truncated)",
                    "indent": 0, "text": ""})
    return out


def format_outline(entries: list[dict]) -> str:
    """One-line-per-landmark rendering for a tool result."""
    parts = []
    for e in entries:
        if e["kind"] == "...":
            parts.append(f"  {e['name']}")
        else:
            parts.append(f"  {e['line']}: {e['kind']} {e['name']}")
    return "\n".join(parts)
