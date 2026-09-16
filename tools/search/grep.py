"""grep_code tool — raw text search, works without an index.

Searches ALL text files (any extension — .example, .template, Makefile,
extension-less), not just recognized source extensions: ripgrep when
available (fast, gitignore-aware, binary-skipping), else grep -rI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from agent.security import path_policy
from agent.tools import register
from agent.tools._common import (
    read_deny_globs as _read_deny_globs,
    is_read_protected as _is_read_protected,
    is_path_allowed as _is_path_allowed,
)
from agent.tools.rules import get_rules

_config = None
_GREP_REGEX_FLAG: str | None = None

# graphify-out: generated graph dumps (manifest hashes, node ids) match almost
# any pattern and bury real hits.
_EXCLUDE_DIRS = tuple(sorted(path_policy.hidden_dir_names()))

_DEFAULT_MAX = 60
_CONTEXT_DEFAULT_MAX = 20   # lower match cap when each hit carries context lines
_MAX_LINE_LEN = 300
_MAX_CONTEXT_CHARS = 2000   # per-match context cap
# Whole-call context budget. Past it, hits keep their line but lose context:
# a 16k grep result is read less carefully than the file itself would be, and
# gets summarised away by tool compaction.
_MAX_TOTAL_CONTEXT_CHARS = 8000


def _grep_regex_flag() -> str:
    """``-P`` when the system grep speaks PCRE, else ``-E``. Probed once."""
    global _GREP_REGEX_FLAG
    if _GREP_REGEX_FLAG is None:
        try:
            proc = subprocess.run(["grep", "-P", "-q", "-e", "x"], input="x\n",
                                  text=True, capture_output=True, timeout=5)
            _GREP_REGEX_FLAG = "-P" if proc.returncode == 0 else "-E"
        except Exception:
            _GREP_REGEX_FLAG = "-E"
    return _GREP_REGEX_FLAG


def setup(config) -> None:
    global _config
    _config = config


@register(
    "grep_code",
    {
        "description": (
            "Grep ALL text files (any extension: source, configs, .example/.template, "
            "Makefile, docs) — no index needed. "
            "Exact text only: names, constants, error codes, hex values. "
            "Finds text, not meaning: misses synonyms and concepts (use search_code). "
            "Known symbol's definition/callers → find_symbol. "
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
                        "(max 10, keep it 2-3). Use when you need to see how a match is used — e.g. "
                        "refactoring — instead of a read_file round trip per hit. "
                        "Whole function or file → read_file instead."
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
    try:
        from agent.security import policy as _sec_policy
        if _sec_policy.is_configured():
            # /tmp/... is the scratch, as for the file tools and the shell.
            search_root = _sec_policy.get().map_tmp(search_root)
    except Exception:
        pass
    search_root = search_root.resolve()

    # Confine to *this tool's* working_dir or a path the user granted
    # (`/paths add`, request_path_access) — a manual check, not
    # security.fs.safe_resolve, so the result stays relative to working_dir
    # rather than the shared security policy root.
    if not _is_path_allowed(search_root, working_dir):
        return {
            "error": (f"path escapes project root: {path!r} -> {search_root} "
                      f"(no path grant covers it; request access with "
                      f"request_path_access or /paths add)"),
            "pattern": pattern,
        }

    context_lines = max(0, min(int(context_lines or 0), 10))
    limit = max_results or (_CONTEXT_DEFAULT_MAX if context_lines else _DEFAULT_MAX)

    if shutil.which("rg"):
        # ripgrep: skips binaries and .gitignore'd files natively; much faster.
        # --hidden: dot-dirs like `.press_review/` hold project code. Pruned
        # dirs (.git, .agent, .venv…) are excluded below and secret files are
        # filtered from results, matching the `grep -r` fallback.
        cmd = ["rg", "-n", "--hidden", "--no-heading", "--with-filename", "--color=never", "--no-messages"]
        if fixed_string:
            cmd.append("-F")
        if case_insensitive:
            cmd.append("-i")
        if file_glob:
            cmd += ["-g", file_glob]
        for excl in _EXCLUDE_DIRS:
            cmd += ["-g", f"!{excl}/"]
        cmd += ["--", pattern, str(search_root)]
        tool_name = "ripgrep"
    else:
        # grep -I: search every file but treat binaries as non-matching, so
        # .example/.template/extension-less text files are covered too.
        cmd = ["grep", "-rnI", "--color=never"]
        if fixed_string:
            cmd.append("-F")
        else:
            # Without a flag, grep reads BRE, where `(?:…)`, `\s`, `+` and `|`
            # are literal text: a pattern written for ripgrep then matches
            # nothing and reports no error. PCRE keeps the two backends
            # agreeing; ERE is the fallback where -P is unavailable.
            cmd.append(_grep_regex_flag())
        if case_insensitive:
            cmd.append("-i")
        if file_glob:
            cmd += ["--include", file_glob]
        for excl in _EXCLUDE_DIRS:
            cmd += ["--exclude-dir", excl]
        cmd += ["--", pattern, str(search_root)]
        tool_name = "grep"

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15, errors="replace")
    except subprocess.TimeoutExpired:
        return {"error": f"{tool_name} timed out", "pattern": pattern}
    except FileNotFoundError:
        return {"error": f"{tool_name} not found on PATH", "pattern": pattern}

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
    context_capped = False
    if context_lines and results:
        file_cache: dict[str, list[str] | None] = {}
        context_used = 0
        for r in results:
            if context_used >= _MAX_TOTAL_CONTEXT_CHARS:
                context_capped = True
                break
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
            context_used += len(r["context"])

    out = {
        "results": results,
        "count": len(results),
        "truncated": truncated,
        "pattern": pattern,
        "source": tool_name,
    }
    if context_capped:
        out["context_capped"] = True
        out["hint"] = ("Context budget reached; later hits have no context. "
                       "Narrow path/pattern, or read_file the file if you need most of it.")
    return out
