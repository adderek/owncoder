from __future__ import annotations

from pathlib import Path

from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack


def _build_gitignore_spec(base):
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


@register(
    "read_file",
    {
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
    },
)
def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    fpath = _resolve(path)

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


@register(
    "list_files",
    {
        "description": (
            "List files in a directory, respecting .gitignore. Returns relative paths with size. "
            "Capped at max_results (default 500) to keep responses small — narrow with `pattern` "
            "(e.g. 'src/**/*.py') if you need more."
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
    },
)
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
