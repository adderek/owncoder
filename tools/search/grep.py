"""grep_code tool — raw text search, works without an index."""
from __future__ import annotations

import fnmatch
import os
import subprocess
from pathlib import Path

from agent.tools import register
from agent.tools.rules import get_rules

_config = None


def _read_deny_globs() -> list[str]:
    """Secret-file globs that must never be surfaced (mirrors security.fs).

    read_file refuses these via safe_open; grep_code must too, or `file_glob='*'`
    would leak .env / key material that read_file blocks.
    """
    try:
        from agent.security import policy as _pol
        if _pol.is_configured():
            g = _pol.get().cfg.read_deny_globs
            if g is not None:
                return g
    except Exception:
        pass
    try:
        from agent.security.fs import _DEFAULT_READ_DENY_GLOBS
        return list(_DEFAULT_READ_DENY_GLOBS)
    except Exception:
        return []


def _is_read_protected(rel: str, deny_globs: list[str]) -> bool:
    if not deny_globs:
        return False
    name = Path(rel).name
    return any(fnmatch.fnmatch(rel, g) or fnmatch.fnmatch(name, g) for g in deny_globs)

_SOURCE_GLOBS = (
    "*.py", "*.c", "*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp",
    "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.rs", "*.java",
    "*.rb", "*.php", "*.cs", "*.swift", "*.kt", "*.lua", "*.zig",
    "*.sh", "*.bash", "*.zsh", "*.fish",
    "*.toml", "*.yaml", "*.yml", "*.json", "*.md",
    "*.S", "*.asm", "*.s",
)

_DEFAULT_MAX = 60
_MAX_LINE_LEN = 300


def setup(config) -> None:
    global _config
    _config = config


@register(
    "grep_code",
    {
        "description": (
            "Grep raw source files — no index needed, always works. "
            "Use for exact matches: names, constants, error codes, hex values. "
            "Also use to verify search_code hits before editing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern (set fixed_string=true for literal match)",
                },
                "path": {
                    "type": "string",
                    "description": "Path to search (default: project root)",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Glob to restrict files, e.g. '*.py'",
                },
                "fixed_string": {
                    "type": "boolean",
                    "description": "Literal string match, not regex (default: false)",
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive (default: false)",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max matches (default: {_DEFAULT_MAX})",
                },
            },
            "required": ["pattern"],
        },
    },
)
def grep_code(
    pattern: str,
    path: str | None = None,
    file_glob: str | None = None,
    fixed_string: bool = False,
    case_insensitive: bool = False,
    max_results: int | None = None,
) -> dict:
    working_dir = (_config.tools.working_dir if _config else None) or os.getcwd()
    search_root = Path(path).expanduser() if path else Path(working_dir)
    if not search_root.is_absolute():
        search_root = Path(working_dir) / search_root
    search_root = search_root.resolve()

    # Confine to project root — reject paths outside working_dir.
    # Uses a manual check (not security.fs.safe_resolve) so the result is always
    # relative to *this tool's* working_dir, not the shared security policy root.
    root = Path(working_dir).resolve()
    try:
        search_root.relative_to(root)
    except ValueError:
        if search_root != root:
            return {
                "error": f"path escapes project root: {path!r} -> {search_root}",
                "pattern": pattern,
            }

    limit = max_results or _DEFAULT_MAX

    cmd = ["grep", "-rn", "--color=never"]
    if fixed_string:
        cmd.append("-F")
    if case_insensitive:
        cmd.append("-i")

    if file_glob:
        cmd += ["--include", file_glob]
    else:
        for g in _SOURCE_GLOBS:
            cmd += ["--include", g]

    for excl in (".git", "__pycache__", "node_modules", ".agent", ".venv", "venv", "build", "dist"):
        cmd += ["--exclude-dir", excl]

    cmd += [pattern, str(search_root)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, errors="replace")
    except subprocess.TimeoutExpired:
        return {"error": "grep timed out", "pattern": pattern}
    except FileNotFoundError:
        return {"error": "grep not found on PATH", "pattern": pattern}

    rules = get_rules()
    deny_globs = _read_deny_globs()
    results = []
    truncated = False

    for raw_line in proc.stdout.splitlines():
        if len(results) >= limit:
            truncated = True
            break
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path, lineno_str, content = parts
        try:
            lineno = int(lineno_str)
        except ValueError:
            continue

        try:
            rel = str(Path(file_path).resolve().relative_to(Path(working_dir).resolve()))
        except ValueError:
            rel = file_path

        if not rules.ignore.empty and rules.ignore.matches(rel):
            continue

        # Never surface secret files read_file would refuse to open.
        if _is_read_protected(rel, deny_globs):
            continue

        results.append({"path": rel, "line": lineno, "content": content[:_MAX_LINE_LEN]})

    return {
        "results": results,
        "count": len(results),
        "truncated": truncated,
        "pattern": pattern,
        "source": "grep",
    }
