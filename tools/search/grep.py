"""grep_code tool — raw text search, works without an index."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent.tools import register
from agent.tools._common import read_deny_globs as _read_deny_globs, is_read_protected as _is_read_protected
from agent.tools.rules import get_rules

_config = None

_SOURCE_GLOBS = (
    "*.py", "*.c", "*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp",
    "*.js", "*.ts", "*.jsx", "*.tsx", "*.go", "*.rs", "*.java",
    "*.rb", "*.php", "*.cs", "*.swift", "*.kt", "*.lua", "*.zig",
    "*.sh", "*.bash", "*.zsh", "*.fish",
    "*.toml", "*.yaml", "*.yml", "*.json", "*.md",
    "*.S", "*.asm", "*.s",
)

_DEFAULT_MAX = 60
_CONTEXT_DEFAULT_MAX = 20   # lower match cap when each hit carries context lines
_MAX_LINE_LEN = 300
_MAX_CONTEXT_CHARS = 2000   # per-match context cap


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
                    "description": f"Max matches (default: {_DEFAULT_MAX}, "
                                   f"or {_CONTEXT_DEFAULT_MAX} when context_lines is set)",
                },
                "context_lines": {
                    "type": "integer",
                    "description": (
                        "Lines of surrounding code attached to each match as `context` "
                        "(max 10). Use when you need to see how a match is used — e.g. "
                        "refactoring — instead of a read_file round trip per hit."
                    ),
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
    context_lines: int = 0,
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

    context_lines = max(0, min(int(context_lines or 0), 10))
    limit = max_results or (_CONTEXT_DEFAULT_MAX if context_lines else _DEFAULT_MAX)

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

    # Attach surrounding lines per match. Reading files here (instead of grep -C)
    # keeps the output parsing unambiguous and reuses the ignore/secret filters
    # already applied above; one read per file, shared across its matches.
    if context_lines and results:
        file_cache: dict[str, list[str] | None] = {}
        for r in results:
            fp = r["path"]
            if fp not in file_cache:
                abs_path = Path(fp)
                if not abs_path.is_absolute():
                    abs_path = Path(working_dir) / fp
                try:
                    file_cache[fp] = abs_path.read_text(
                        encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    file_cache[fp] = None
            lines = file_cache[fp]
            if lines is None:
                continue
            lo = max(0, r["line"] - 1 - context_lines)
            hi = min(len(lines), r["line"] + context_lines)
            block = "\n".join(
                f"{i + 1}{'>' if i + 1 == r['line'] else ':'} {lines[i]}"
                for i in range(lo, hi)
            )
            r["context"] = block[:_MAX_CONTEXT_CHARS]

    return {
        "results": results,
        "count": len(results),
        "truncated": truncated,
        "pattern": pattern,
        "source": "grep",
    }
