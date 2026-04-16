from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools import register
from agent.tools.rules import get_rules

if TYPE_CHECKING:
    from agent.config import Config


_config = None
_undo_stack: dict[str, str] = {}


def _log_edit(tool: str, path: str, outcome: str, **extra) -> None:
    """Append one JSONL record per edit attempt so usage can be audited later."""
    try:
        agent_dir = Path(_config.tools.agent_dir) if _config else Path(".agent")
        if not agent_dir.is_absolute():
            agent_dir = _working_dir() / agent_dir
        agent_dir.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "tool": tool, "path": path, "outcome": outcome, **extra}
        with (agent_dir / "edit_stats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def setup(config) -> None:
    global _config
    _config = config


def _working_dir() -> Path:
    if _config:
        return Path(_config.tools.working_dir).resolve()
    return Path.cwd()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _working_dir() / p
    resolved = p.resolve()
    base = _working_dir().resolve()
    if resolved != base and not str(resolved).startswith(str(base) + "/"):
        raise ValueError(f"Path escapes working directory: {path!r}")
    return resolved


@register("read_file", {
    "description": "Read file contents, optionally limited to a line range. Always use start_line/end_line for large files.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
            "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
            "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
        },
        "required": ["path"],
    },
})
def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    fpath = _resolve(path)

    # Rule check: .agent.ignore — pretend file doesn't exist
    rel = str(fpath.relative_to(_working_dir()))
    allowed, _ = get_rules().check_read(rel)
    if not allowed:
        return {"error": f"File not found: {path}"}

    if not fpath.exists():
        return {"error": f"File not found: {path}"}
    if not fpath.is_file():
        return {"error": f"Not a file: {path}"}

    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": str(e)}

    lines = text.splitlines()
    total = len(lines)

    if start_line is None and end_line is None and total > 500:
        # Return the first 200 lines with a hint rather than nothing.
        head = "\n".join(f"{i + 1}:{l}" for i, l in enumerate(lines[:200]))
        return {
            "warning": f"File has {total} lines; showing first 200. Use start_line/end_line for other ranges.",
            "content": head,
        }

    sl = (start_line or 1) - 1
    el = end_line if end_line else total
    selected = lines[sl:el]

    numbered = "\n".join(f"{sl + i + 1}:{line}" for i, line in enumerate(selected))

    if fpath.suffix == ".py":
        try:
            import ast
            ast.parse(text)
        except SyntaxError as e:
            if path in _undo_stack:
                return {
                    "warning": f"File has syntax error: {e}. It was recently modified. Use undo_file to revert.",
                    "content": numbered,
                }

    return {"content": numbered}


@register("write_file", {
    "description": "Write content to a file. Creates parent dirs if needed. Use only for new files — prefer edit_file for modifying existing files.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "New file content"},
        },
        "required": ["path", "content"],
    },
})
def write_file(path: str, content: str) -> dict:
    import difflib
    fpath = _resolve(path)

    # Rule checks: .agent.ignore, .agent.ro, language allowlist, max files
    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    is_new = not fpath.exists()
    allowed, msg = rules.check_write(rel, is_new=is_new)
    if not allowed:
        return {"error": msg or f"Cannot write to: {path}"}
    size_ok, size_msg = rules.check_write_size(content)
    if not size_ok:
        return {"error": size_msg}
    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_write": f"{len(content)} bytes"}
    if is_new and rules.config.confirm_create:
        return {"error": f"Creating new files requires confirmation: {path}", "requires_confirm": True}

    fpath.parent.mkdir(parents=True, exist_ok=True)

    if fpath.exists():
        original = fpath.read_text(encoding="utf-8", errors="replace")
        _undo_stack[path] = original
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        ))
        diff_summary = "".join(diff_lines[:60])
        if len(diff_lines) > 60:
            diff_summary += f"\n... ({len(diff_lines) - 60} more diff lines)"
    else:
        diff_summary = f"(new file, {len(content.splitlines())} lines)"

    fpath.write_text(content, encoding="utf-8")

    if not is_new and fpath.suffix == ".py":
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            return {"error": f"File written but has syntax error: {e}. Use undo_file to revert.", "path": path}

    return {"ok": path, "diff": diff_summary}


def patch_file(path: str, unified_diff: str) -> dict:
    fpath = _resolve(path)

    # Rule checks: .agent.ignore, .agent.ro, patch size
    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    allowed, msg = rules.check_write(rel)
    if not allowed:
        return {"error": msg or f"Cannot write to: {path}"}
    lines_ok, lines_msg = rules.check_patch_lines(unified_diff)
    if not lines_ok:
        return {"error": lines_msg}
    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_patch": f"{unified_diff.count(chr(10))+1} lines"}

    if not fpath.exists():
        return {"error": f"File not found: {path}"}

    original = fpath.read_text(encoding="utf-8", errors="replace")
    _undo_stack[path] = original

    try:
        patched = _apply_unified_diff(original, unified_diff)
    except Exception as e:
        error_msg = str(e)
        if "patch dry-run failed" in error_msg:
            error_msg = "The provided unified diff does not match the file content (dry-run failed). Check your hunk headers, line numbers, and context lines."
        elif "malformed patch" in error_msg:
            error_msg = "The provided diff is malformed. Ensure it is a valid unified diff with correct hunk headers (@@ -L,C +L,C @@)."
        return {"error": f"Patch failed: {error_msg}"}

    fpath.write_text(patched, encoding="utf-8")
    _log_edit("patch_file", path, "ok")
    return {"ok": path}


def _find_matches_fuzzy(haystack: str, needle: str) -> list[tuple[int, int]]:
    """Return (start, end) match spans using whitespace-tolerant matching.

    Preserves one match per fuzzy hit. Collapses runs of whitespace (incl. newlines)
    in the needle to `\\s+` and escapes the rest. Leading/trailing whitespace on each
    side is matched loosely so indentation drift doesn't break things.
    """
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
    return {
        "line": start_line,
        "before": snippet_before,
        "match": "\n".join(matched),
        "after": snippet_after,
    }


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

    # Exact match first (preserves old behavior + is fastest).
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
        candidates = [
            {"index": i, **_context_for(original, s, e)}
            for i, (s, e) in enumerate(spans)
        ]
        _log_edit("replace_text", path, "ambiguous", candidates=len(spans))
        return {
            "error": f"search_block matched {len(spans)} locations. Re-call with match_index.",
            "match_count": len(spans),
            "candidates": candidates,
        }

    pick = match_index if match_index is not None else 0
    if pick < 0 or pick >= len(spans):
        return {"error": f"match_index {pick} out of range (0..{len(spans)-1})"}

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


@register("replace_symbol", {
    "description": (
        "Replace the full source of a top-level or nested Python function/class by name. "
        "Immune to whitespace drift and duplicate text elsewhere in the file. "
        "Use dotted names for nested symbols (e.g. 'MyClass.method'). Python files only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Python file path"},
            "symbol": {"type": "string", "description": "Symbol name, dotted for nesting (e.g. 'Foo.bar')"},
            "new_source": {"type": "string", "description": "Full replacement source for the symbol, including def/class line. Will be re-indented to match the original."},
        },
        "required": ["path", "symbol", "new_source"],
    },
})
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
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part:
                found = child
                break
        if found is None:
            return {"error": f"Symbol not found: {'.'.join(parts[:i+1])}"}
        node = found
        if i < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return {"error": f"{'.'.join(parts[:i+1])} is not a class; cannot descend into it."}
            parent_body = node.body

    assert node is not None
    lines = original.splitlines(keepends=True)
    # Include preceding decorators in the replaced span.
    start_line = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    end_line = node.end_lineno  # 1-indexed, exclusive when used as slice end
    orig_block = "".join(lines[start_line:end_line])

    indent_match = re.match(r"[ \t]*", lines[start_line])
    indent = indent_match.group(0) if indent_match else ""

    new_body = textwrap.dedent(new_source).rstrip("\n")
    new_indented = "\n".join((indent + ln if ln else ln) for ln in new_body.split("\n")) + "\n"

    # Re-parse the whole file after substitution to catch obviously broken replacements.
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


def _apply_unified_diff(original: str, patch: str) -> str:
    import subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".orig", delete=False, encoding="utf-8") as f:
        f.write(original)
        orig_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
        f.write(patch)
        patch_path = f.name

    try:
        result = subprocess.run(
            ["patch", "--dry-run", orig_path, patch_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Extract conflicting context
            raise RuntimeError(result.stderr.strip() or "patch dry-run failed")

        result = subprocess.run(
            ["patch", orig_path, patch_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        return Path(orig_path).read_text(encoding="utf-8")
    finally:
        os.unlink(orig_path)
        os.unlink(patch_path)
        # patch creates .orig backup
        backup = orig_path + ".orig"
        if os.path.exists(backup):
            os.unlink(backup)


@register("undo_file", {
    "description": "Revert a file to its previous state before the last write/edit operation.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to revert"},
        },
        "required": ["path"],
    },
})
def undo_file(path: str) -> dict:
    """Restore the last pre-write snapshot of a file."""
    if path not in _undo_stack:
        return {"error": f"No undo snapshot for: {path}"}
    try:
        fpath = _resolve(path)
        fpath.write_text(_undo_stack.pop(path), encoding="utf-8")
        return {"ok": path}
    except Exception as e:
        return {"error": str(e)}


def undo_candidates() -> list[str]:
    """Return paths that have undo snapshots."""
    return list(_undo_stack.keys())


def _build_gitignore_spec(base: Path):
    """Return a pathspec.PathSpec from .gitignore (or None if pathspec unavailable/missing)."""
    try:
        import pathspec  # type: ignore
    except ImportError:
        return None
    gi = base / ".gitignore"
    if not gi.exists():
        return None
    lines = [
        line.strip()
        for line in gi.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


@register("list_files", {
    "description": (
        "List files in a directory, respecting .gitignore. Returns relative paths with size. "
        "Capped at max_results (default 500) to keep responses small — narrow with `pattern` "
        "(e.g. 'agent/**/*.py') if you need more."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: working dir)", "default": "."},
            "pattern": {"type": "string", "description": "Glob pattern", "default": "**/*"},
            "ignore_patterns": {"type": "array", "items": {"type": "string"}, "description": "Patterns to ignore"},
            "max_results": {"type": "integer", "description": "Max entries to return (default 500). On overflow, a directory-grouped summary is returned instead of paths."},
        },
        "required": [],
    },
})
def list_files(
    path: str = ".",
    pattern: str = "**/*",
    ignore_patterns: list[str] | None = None,
    max_results: int = 500,
) -> dict:
    import fnmatch
    base = _resolve(path)
    if not base.is_dir():
        return {"error": f"Not a directory: {path}"}

    default_ignore = {".git", "__pycache__", "node_modules", "*.pyc", "build", "dist", ".agent"}
    all_ignore = default_ignore | set(ignore_patterns or [])

    gitignore_spec = _build_gitignore_spec(base)

    cap = max(1, int(max_results))
    rules = get_rules()
    results: list[dict] = []
    dir_counts: dict[str, int] = {}
    total_kept = 0

    for fpath in sorted(base.glob(pattern)):
        if not fpath.is_file():
            continue
        rel = str(fpath.relative_to(base))
        parts = Path(rel).parts

        skip = False
        for part in parts:
            for pat in all_ignore:
                if fnmatch.fnmatch(part, pat):
                    skip = True
                    break
            if skip:
                break

        if not skip and gitignore_spec is not None:
            skip = gitignore_spec.match_file(rel)

        # Rule check: .agent.ignore
        if not skip and rules.ignore.matches(rel):
            skip = True

        if skip:
            continue

        total_kept += 1
        top = parts[0] if len(parts) > 1 else "."
        dir_counts[top] = dir_counts.get(top, 0) + 1
        if len(results) < cap:
            stat = fpath.stat()
            results.append({"path": rel, "size": stat.st_size})

    if total_kept > cap:
        summary = sorted(
            ({"dir": d, "count": n} for d, n in dir_counts.items()),
            key=lambda x: -x["count"],
        )
        return {
            "truncated": True,
            "total": total_kept,
            "returned": 0,
            "by_top_dir": summary,
            "hint": (
                f"{total_kept} files matched (cap={cap}). "
                "Re-call with a narrower `pattern` (e.g. 'src/**/*.py') or a deeper `path`, "
                "or raise `max_results` if you really need everything."
            ),
        }

    return {"files": results, "count": len(results)}
