from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

_config = None
_undo_stack: dict[str, str] = {}


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
    return {"content": numbered}


@register("write_file", {
    "description": "Write content to a file. Creates parent dirs if needed. Prefer patch_file for modifying existing files.",
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
    return {"ok": path, "diff": diff_summary}


@register("patch_file", {
    "description": "Apply a unified diff patch to an existing file. Preferred over write_file for modifying code.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to patch"},
            "unified_diff": {"type": "string", "description": "Unified diff string to apply"},
        },
        "required": ["path", "unified_diff"],
    },
})
def patch_file(path: str, unified_diff: str) -> dict:
    fpath = _resolve(path)
    if not fpath.exists():
        return {"error": f"File not found: {path}"}

    original = fpath.read_text(encoding="utf-8", errors="replace")
    _undo_stack[path] = original

    try:
        patched = _apply_unified_diff(original, unified_diff)
    except Exception as e:
        return {"error": f"Patch failed: {e}"}

    fpath.write_text(patched, encoding="utf-8")
    return {"ok": path}


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


def undo_file(path: str) -> dict:
    """Restore the last pre-write snapshot of a file (not a tool — called by UI /undo)."""
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
    "description": "List files in a directory, respecting .gitignore. Returns relative paths with size and mtime.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: working dir)", "default": "."},
            "pattern": {"type": "string", "description": "Glob pattern", "default": "**/*"},
            "ignore_patterns": {"type": "array", "items": {"type": "string"}, "description": "Patterns to ignore"},
        },
        "required": [],
    },
})
def list_files(path: str = ".", pattern: str = "**/*", ignore_patterns: list[str] | None = None) -> dict:
    import fnmatch
    base = _resolve(path)
    if not base.is_dir():
        return {"error": f"Not a directory: {path}"}

    default_ignore = {".git", "__pycache__", "node_modules", "*.pyc", "build", "dist", ".agent"}
    all_ignore = default_ignore | set(ignore_patterns or [])

    gitignore_spec = _build_gitignore_spec(base)

    results = []
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

        if not skip:
            stat = fpath.stat()
            results.append({"path": rel, "size": stat.st_size})

    return {"files": results, "count": len(results)}
