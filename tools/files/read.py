from __future__ import annotations

from pathlib import Path

from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack


def _format_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.0f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.0f}TB"


def _count_lines_fast(fpath: Path) -> int:
    """Count lines without reading full file into memory."""
    with open(fpath, "rb") as f:
        return sum(1 for _ in f)


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
    return pathspec.PathSpec.from_lines("gitignore", lines)


@register(
    "read_file",
    {
        "description": (
            "Read file contents, optionally limited to a line range. "
            "Always use start_line/end_line for large files. "
            "When a line range exceeds the file, the range is auto-clamped and "
            "end_of_file=true is returned so you know you've reached the end."
        ),
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
        return {"error": f"File not found: {path}", "resolved": str(fpath)}
    if not fpath.is_file():
        return {"error": f"Not a file: {path}", "resolved": str(fpath)}

    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": str(e)}

    lines = text.splitlines()
    total = len(lines)
    filesize = fpath.stat().st_size

    def _make_header(sl_show: int, el_show: int, clamped: bool = False, past_eof: bool = False) -> str:
        size_str = f" · {_format_size(filesize)}"
        if past_eof:
            return (
                f"[{fpath.name} · {total} lines{size_str} · "
                f"END OF FILE — requested line {sl_show} beyond file's {total} lines. "
                f"Last {el_show - sl_show + 1} lines shown for reference below.]"
            )
        if clamped:
            return (
                f"[{fpath.name} · {total} lines{size_str} · "
                f"showing lines {sl_show}-{el_show}"
                f" · read offset={el_show + 1} for more · "
                f"note: requested up to line {end_line} but file has {total}]"
            )
        if sl_show == 1 and el_show >= total:
            return f"[{fpath.name} · {total} lines{size_str}]"
        return (
            f"[{fpath.name} · {total} lines{size_str} · "
            f"showing lines {sl_show}-{el_show} · "
            f"read offset={el_show + 1} for more]"
        )

    if start_line is None and end_line is None and total > 500:
        head_lines = lines[:200]
        numbered = "\n".join(f"{i + 1}:{l}" for i, l in enumerate(head_lines))
        return {
            "content": _make_header(1, 200) + "\n" + numbered,
            "metadata": {"total_lines": total, "file_size": filesize},
        }

    past_eof = False
    clamped = False

    if start_line is not None and start_line > total:
        # Model requested past EOF — show last 50 lines as reference
        sl = max(0, total - 50)
        el = total
        past_eof = True
    else:
        sl = max(0, (start_line or 1) - 1)
        clamped = end_line is not None and end_line > total
        el = min(end_line if end_line else total, total)

    selected = lines[sl:el]
    numbered = "\n".join(f"{sl + i + 1}:{line}" for i, line in enumerate(selected))

    result: dict = {
        "content": _make_header(sl + 1, el, clamped, past_eof) + "\n" + numbered,
        "metadata": {"total_lines": total, "file_size": filesize},
    }
    if past_eof:
        result["end_of_file"] = True

    if fpath.suffix == ".py":
        try:
            import ast
            ast.parse(text)
        except SyntaxError as e:
            if path in _undo_stack:
                result["warning"] = (
                    f"File has syntax error: {e}. It was recently modified. Use undo_file to revert."
                )

    return result


@register(
    "list_files",
    {
        "description": (
            "List files in a directory, respecting .gitignore. Returns relative paths with size. "
            "Capped at max_results (default 500) to keep responses small — narrow with `pattern` "
            "(e.g. 'src/**/*.py') if you need more. "
            "Set include_lines=true to also get line counts (slower for large trees)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: working dir)", "default": "."},
                "pattern": {"type": "string", "description": "Glob pattern", "default": "**/*"},
                "ignore_patterns": {"type": "array", "items": {"type": "string"}, "description": "Patterns to ignore"},
                "max_results": {"type": "integer", "description": "Max entries to return (default 500). On overflow, a directory-grouped summary is returned instead of paths."},
                "include_lines": {"type": "boolean", "description": "Include line count per file (default: false)", "default": False},
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
    include_lines: bool = False,
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
            entry = {"path": rel, "size": stat.st_size}
            if include_lines:
                entry["lines"] = _count_lines_fast(fpath)
            results.append(entry)

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
