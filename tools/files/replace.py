from __future__ import annotations

import re

from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack, _log_edit


def _find_matches_fuzzy(haystack: str, needle: str) -> list[tuple[int, int]]:
    stripped = needle.strip("\n")
    if not stripped:
        return []
    parts = re.split(r"\s+", stripped)
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if not pattern:
        return []
    return [(m.start(), m.end()) for m in re.finditer(pattern, haystack)]


def _context_for(text: str, start: int, end: int, ctx_lines: int = 3) -> dict:
    before = text[:start].splitlines()
    matched = text[start:end].splitlines()
    start_line = len(before) + (0 if before and not text[:start].endswith("\n") else 1)
    snippet_before = "\n".join(before[-ctx_lines:])
    snippet_after = "\n".join(text[end:].splitlines()[:ctx_lines])
    return {"line": start_line, "before": snippet_before, "match": "\n".join(matched), "after": snippet_after}


def replace_text(path: str, search_block: str, replace_block: str, match_index: int | None = None) -> dict:
    fpath = _resolve(path)

    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    allowed, msg = rules.check_write(rel)
    if not allowed:
        _log_edit("replace_text", path, "blocked", reason=msg)
        return {"error": msg or f"Cannot write to: {path}"}

    if not fpath.exists():
        _log_edit("replace_text", path, "not_found")
        return {"error": f"File not found: {path}"}

    original = fpath.read_text(encoding="utf-8", errors="replace")

    spans: list[tuple[int, int]] = []
    idx = original.find(search_block)
    while idx != -1:
        spans.append((idx, idx + len(search_block)))
        idx = original.find(search_block, idx + 1)

    match_mode = "exact"
    if not spans:
        spans = _find_matches_fuzzy(original, search_block)
        match_mode = "fuzzy"

    if not spans:
        _log_edit("replace_text", path, "no_match")
        return {"error": "search_block not found (tried exact and whitespace-tolerant match). Re-read the file and try again."}

    if len(spans) > 1 and match_index is None:
        candidates = [{"index": i, **_context_for(original, s, e)} for i, (s, e) in enumerate(spans)]
        _log_edit("replace_text", path, "ambiguous", candidates=len(spans))
        return {
            "error": f"search_block matched {len(spans)} locations. Re-call with match_index.",
            "match_count": len(spans),
            "candidates": candidates,
        }

    pick = match_index if match_index is not None else 0
    if pick < 0 or pick >= len(spans):
        return {"error": f"match_index {pick} out of range (0..{len(spans) - 1})"}

    s, e = spans[pick]
    new_content = original[:s] + replace_block + original[e:]

    size_ok, size_msg = rules.check_write_size(new_content)
    if not size_ok:
        return {"error": size_msg}

    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_replace": f"{e - s} chars -> {len(replace_block)} chars", "match_mode": match_mode}

    _undo_stack[path] = original
    fpath.write_text(new_content, encoding="utf-8")
    _log_edit("replace_text", path, "ok", match_mode=match_mode, candidates=len(spans))
    return {"ok": path, "match_mode": match_mode}


@register(
    "replace_symbol",
    {
        "description": (
            "Replace full source of a Python function/class by name. "
            "Immune to whitespace drift and duplicate text. "
            "Dotted names for nested symbols ('MyClass.method'). Python only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Python file path"},
                "symbol": {"type": "string", "description": "Symbol name, dotted for nesting (e.g. 'Foo.bar')"},
                "new_source": {"type": "string", "description": "Full replacement including def/class line. Auto re-indented."},
            },
            "required": ["path", "symbol", "new_source"],
        },
    },
)
def replace_symbol(path: str, symbol: str, new_source: str) -> dict:
    import ast
    import textwrap

    fpath = _resolve(path)
    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    allowed, msg = rules.check_write(rel)
    if not allowed:
        _log_edit("replace_symbol", path, "blocked", reason=msg)
        return {"error": msg or f"Cannot write to: {path}"}

    if not fpath.exists():
        return {"error": f"File not found: {path}"}
    if fpath.suffix != ".py":
        return {"error": "replace_symbol currently supports Python files only."}

    original = fpath.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(original)
    except SyntaxError as e:
        return {"error": f"File has a syntax error; fix it first or use edit_file. ({e})"}

    parts = symbol.split(".")
    node = None
    parent_body = tree.body
    for i, part in enumerate(parts):
        found = None
        for child in parent_body:
            if (
                isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name == part
            ):
                found = child
                break
        if found is None:
            return {"error": f"Symbol not found: {'.'.join(parts[: i + 1])}"}
        node = found
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return {"error": f"{'.'.join(parts[: i + 1])} is not a class; cannot descend into it."}
            parent_body = node.body

    assert node is not None
    lines = original.splitlines(keepends=True)
    start_line = (
        min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    )
    end_line = node.end_lineno
    orig_block = "".join(lines[start_line:end_line])  # noqa: F841

    indent_match = re.match(r"[ \t]*", lines[start_line])
    indent = indent_match.group(0) if indent_match else ""

    new_body = textwrap.dedent(new_source).rstrip("\n")
    new_indented = "\n".join((indent + ln if ln else ln) for ln in new_body.split("\n")) + "\n"

    candidate = "".join(lines[:start_line]) + new_indented + "".join(lines[end_line:])
    try:
        ast.parse(candidate)
    except SyntaxError as e:
        return {"error": f"Replacement would break the file's syntax: {e}. Check indentation and completeness of new_source."}

    size_ok, size_msg = rules.check_write_size(candidate)
    if not size_ok:
        return {"error": size_msg}

    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "symbol": symbol, "old_lines": end_line - start_line, "new_lines": new_indented.count("\n")}

    _undo_stack[path] = original
    fpath.write_text(candidate, encoding="utf-8")
    _log_edit("replace_symbol", path, "ok", symbol=symbol)
    return {"ok": path, "symbol": symbol, "replaced_lines": end_line - start_line}
