"""Fast, file-scoped diagnostics run immediately after a successful file edit.

This is deliberately *not* the same thing as the post-edit verify hook
(``VerifyConfig`` / ``_run_verify_command`` in ``core/turn.py``). That one runs a
project-wide command once per turn, at the end, after the model already decided
it was finished. This runs a sub-second, single-file checker and folds the
findings into the tool result the model is about to read — so a syntax error
introduced by an edit is corrected on the *next* step instead of costing a
verify-fail round trip.

Design constraints:

- **Off unless useful.** Disabled by default; when enabled with no explicit
  checkers, only checkers whose binary is actually installed are used.
- **No shell.** Commands are argv lists, never a shell string, so a filename can
  never inject. The path is resolved and confined to the working directory.
- **Budgeted.** Hard per-file timeout; a slow or hanging checker is dropped, it
  never blocks the turn.
- **Advisory.** Findings are attached under a ``_diagnostics`` key on the tool
  result JSON. A tool result that is not a JSON object is left untouched.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# suffix -> (binary, argv template). "{file}" is replaced with the absolute path.
# Only entries whose binary resolves on PATH are used in auto mode. Every command
# here is expected to be file-scoped and fast; project-wide checkers belong in
# [verify], not here.
_AUTO_CHECKERS: list[tuple[tuple[str, ...], str, list[str]]] = [
    ((".py",), "ruff", ["ruff", "check", "--output-format", "concise", "{file}"]),
    ((".sh", ".bash"), "bash", ["bash", "-n", "{file}"]),
    ((".go",), "gofmt", ["gofmt", "-e", "-l", "{file}"]),
    ((".js", ".jsx", ".ts", ".tsx"), "eslint",
     ["eslint", "--no-eslintrc", "--format", "compact", "{file}"]),
]

# Fallback for Python when ruff is absent: the interpreter always is.
_PY_COMPILE = ["python3", "-c",
               "import sys,py_compile;py_compile.compile(sys.argv[1],doraise=True)", "{file}"]


def _resolve_argv(config: "Config") -> list[tuple[tuple[str, ...], list[str]]]:
    """(suffixes, argv) pairs for the active checker set.

    Explicit ``[diagnostics] checkers`` entries win as written — the user is
    responsible for their speed and their argv. Otherwise auto mode picks the
    installed subset.
    """
    cfg = getattr(config, "diagnostics", None)
    explicit = list(getattr(cfg, "checkers", None) or [])
    if explicit:
        out = []
        for entry in explicit:
            suffixes = tuple(getattr(entry, "suffixes", None) or ())
            command = list(getattr(entry, "command", None) or ())
            if suffixes and command:
                out.append((suffixes, command))
        return out

    out = []
    for suffixes, binary, argv in _AUTO_CHECKERS:
        if shutil.which(binary):
            out.append((suffixes, argv))
        elif suffixes == (".py",) and shutil.which("python3"):
            out.append((suffixes, _PY_COMPILE))
    return out


def _confined_path(raw: str, config: "Config") -> Path | None:
    """Absolute path for *raw* if it stays inside the working directory."""
    if not raw:
        return None
    try:
        root = Path(config.tools.working_dir).resolve()
        path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        path.relative_to(root)
    except (ValueError, OSError):
        return None
    return path if path.is_file() else None


async def _run(argv: list[str], path: Path, cwd: str, timeout_s: float) -> str:
    """Run one checker; return its combined output, or "" on clean/failure-to-run.

    A non-zero exit is the normal "found problems" signal for these tools, so the
    exit code is ignored and only the text matters. Anything that times out or
    cannot be spawned yields no findings rather than an error — diagnostics are
    advisory and must never break a turn.
    """
    cmd = [a.replace("{file}", str(path)) for a in argv]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=cwd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as e:
        logger.debug("diagnostics: cannot run %s: %s", cmd[0], e)
        return ""
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.debug("diagnostics: %s timed out after %.1fs", cmd[0], timeout_s)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return ""
    text = (out or b"").decode("utf-8", "replace") + (err or b"").decode("utf-8", "replace")
    return text.strip()


async def check_path(raw_path: str, config: "Config") -> list[str]:
    """Findings for a single edited file. Empty list means clean or not checked."""
    cfg = getattr(config, "diagnostics", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return []
    path = _confined_path(raw_path, config)
    if path is None:
        return []
    suffix = path.suffix.lower()
    checkers = [argv for suffixes, argv in _resolve_argv(config) if suffix in suffixes]
    if not checkers:
        return []

    timeout_s = float(getattr(cfg, "timeout_s", 5.0))
    max_findings = int(getattr(cfg, "max_findings", 10))
    cwd = str(config.tools.working_dir)
    findings: list[str] = []
    for argv in checkers:
        text = await _run(argv, path, cwd, timeout_s)
        if not text:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line and line not in findings:
                findings.append(line)
        if len(findings) >= max_findings:
            break
    return findings[:max_findings]


def _path_from_args(arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except (ValueError, TypeError):
        return ""
    if not isinstance(args, dict):
        return ""
    value = args.get("path") or args.get("file_path") or ""
    return value if isinstance(value, str) else ""


async def annotate(tool_calls, results: list[str], config: "Config") -> list[str]:
    """Attach ``_diagnostics`` to results of successful edits; pass others through.

    One check per distinct path, even when several tool calls touched the same
    file in one batch. Every finding set is attached to every result for that
    path so the model sees it wherever it looks.
    """
    cfg = getattr(config, "diagnostics", None)
    if cfg is None or not getattr(cfg, "enabled", False):
        return results

    parsed: list[dict | None] = []
    paths: list[str] = []
    for tc, result in zip(tool_calls, results):
        obj = None
        path = ""
        try:
            candidate = json.loads(result)
            if isinstance(candidate, dict) and not candidate.get("error"):
                obj = candidate
                path = _path_from_args(tc.function.arguments)
        except (ValueError, TypeError):
            pass
        parsed.append(obj)
        paths.append(path)

    unique = [p for p in dict.fromkeys(paths) if p]
    if not unique:
        return results
    try:
        found = await asyncio.gather(*[check_path(p, config) for p in unique])
    except Exception:
        logger.exception("diagnostics: check batch failed")
        return results
    by_path = dict(zip(unique, found))

    out: list[str] = []
    for result, obj, path in zip(results, parsed, paths):
        findings = by_path.get(path) if obj is not None else None
        if not findings:
            out.append(result)
            continue
        obj["_diagnostics"] = {
            "path": path,
            "findings": findings,
            "note": "Checker output for the file you just edited. Fix real problems "
                    "now; pre-existing unrelated warnings can be left alone.",
        }
        try:
            out.append(json.dumps(obj))
        except (TypeError, ValueError):
            out.append(result)
    return out
