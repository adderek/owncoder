"""`run_tests` tool — discover and run the project's test suites, summarized.

Declaration beats detection. Resolution tiers (first hit wins, unless
`framework` is forced):

1. Declared suites — ``[[tests.suites]]`` in agent.toml (see TestsConfig).
   The right answer for polyglot/monorepo projects: each suite carries its
   own dir, command (interpreter, env) and runs as written. No suite/pattern
   arg → all ``default = true`` suites, aggregated.
2. ``config.verify.command`` — the project's canonical single test command
   (full-suite runs only; it cannot filter).
3. Conventions — ``make test`` / ``just test`` / ``scripts/test.sh``
   (full-suite runs only).
4. Framework heuristics — pytest config or test_*.py files, Cargo.toml,
   go.mod, package.json test script; unittest as forced fallback. Picks the
   project's venv interpreter when present.

Heuristic commands are built as argv lists (no shell). Declared suite
commands, verify.command, and convention scripts are trusted project config
and run through the shell as written.
"""
from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

from agent.tools import register
from agent.tools._common import working_dir

if TYPE_CHECKING:
    from agent.config import Config

_config: "Config | None" = None

_OUTPUT_TAIL_CHARS = 4_000
_DEFAULT_TIMEOUT_S = 300


def setup(config: "Config") -> None:
    global _config
    _config = config


def _project_python(root: str) -> str:
    """Project venv interpreter when present, else system python3."""
    for rel in (".venv/bin/python", "venv/bin/python"):
        p = os.path.join(root, rel)
        if os.access(p, os.X_OK):
            return p
    return "python3"


def _has_pytest_config(root: str) -> bool:
    if os.path.exists(os.path.join(root, "pytest.ini")):
        return True
    for name, needle in (("pyproject.toml", "[tool.pytest"),
                         ("setup.cfg", "[tool:pytest]"),
                         ("tox.ini", "[pytest]")):
        p = os.path.join(root, name)
        try:
            if needle in open(p, encoding="utf-8", errors="replace").read():
                return True
        except OSError:
            continue
    return False


def _has_test_files(root: str) -> bool:
    """Cheap probe: any test_*.py / *_test.py within 3 directory levels."""
    root = root.rstrip(os.sep)
    base_depth = root.count(os.sep)
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox"}
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= 3:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in skip]
        for f in filenames:
            if f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py")):
                return True
    return False


def detect_framework(root: str) -> str | None:
    if _has_pytest_config(root):
        return "pytest"
    if os.path.exists(os.path.join(root, "Cargo.toml")):
        return "cargo"
    if os.path.exists(os.path.join(root, "go.mod")):
        return "go"
    pkg = os.path.join(root, "package.json")
    if os.path.exists(pkg):
        try:
            import json as _json
            scripts = _json.load(open(pkg, encoding="utf-8")).get("scripts", {})
            if scripts.get("test"):
                return "npm"
        except Exception:
            pass
    if _has_test_files(root):
        return "pytest"  # pytest runs bare test_*.py fine; unittest is the forced fallback
    return None


def _build_argv(framework: str, root: str, pattern: str) -> list[str]:
    if framework == "pytest":
        argv = [_project_python(root), "-m", "pytest", "-q", "--no-header"]
        if pattern:
            # A path-ish pattern targets files/dirs; anything else is a -k expression.
            if "/" in pattern or pattern.endswith(".py"):
                argv.append(pattern)
            else:
                argv += ["-k", pattern]
        return argv
    if framework == "unittest":
        argv = [_project_python(root), "-m", "unittest", "discover", "-v"]
        if pattern:
            argv += ["-p", pattern]
        return argv
    if framework == "cargo":
        argv = ["cargo", "test"]
        if pattern:
            argv.append(pattern)
        return argv
    if framework == "go":
        argv = ["go", "test", "./..."]
        if pattern:
            argv += ["-run", pattern]
        return argv
    if framework == "npm":
        argv = ["npm", "test", "--silent"]
        if pattern:
            argv += ["--", pattern]
        return argv
    raise ValueError(f"unsupported framework: {framework}")


# ── Declared suites (tier 1) ─────────────────────────────────────────────────

def _select_suites(suites: list, selector: str) -> list:
    """Pick declared suites. Empty selector → all default ones; otherwise
    exact name match wins, then case-insensitive substring matches."""
    if not selector:
        return [s for s in suites if getattr(s, "default", True)]
    exact = [s for s in suites if s.name == selector]
    if exact:
        return exact
    sel = selector.lower()
    return [s for s in suites if sel in s.name.lower()]


def _guess_parser(suite) -> str:
    p = (getattr(suite, "parser", "auto") or "auto").lower()
    if p != "auto":
        return p
    cmd = suite.command
    if "pytest" in cmd:
        return "pytest"
    if cmd.startswith("go test") or " go test" in cmd:
        return "go"
    if cmd.startswith("cargo") or " cargo " in cmd:
        return "cargo"
    return "none"


async def _run_suites(suites: list, root: str, timeout_s: int) -> dict[str, Any]:
    results = []
    totals = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    all_ok = True
    for s in suites:
        cwd = os.path.abspath(os.path.join(root, s.dir or "."))
        if not os.path.isdir(cwd):
            results.append({"suite": s.name, "ok": False,
                            "error": f"suite dir does not exist: {cwd}"})
            all_ok = False
            continue
        if not (s.command or "").strip():
            results.append({"suite": s.name, "ok": False, "error": "suite has no command"})
            all_ok = False
            continue
        env = {**os.environ, **{str(k): str(v) for k, v in (s.env or {}).items()}}
        t = int(getattr(s, "timeout_s", 0) or 0) or timeout_s
        started = time.monotonic()
        rc, out = await asyncio.to_thread(_run, s.command, cwd, t, True, env)
        parser = _guess_parser(s)
        if parser == "pytest":
            summary = _parse_pytest(out)
        elif parser == "go":
            summary = _parse_go(out)
        elif parser == "cargo":
            summary = _parse_cargo(out)
        else:
            summary = _parse_flexible(out, rc)
        ok = rc == 0
        all_ok = all_ok and ok
        for k in totals:
            totals[k] += summary.get(k, 0)
        results.append({
            "suite": s.name, "dir": s.dir, "command": s.command,
            "returncode": rc, "ok": ok, "timed_out": rc == 124,
            "duration_s": round(time.monotonic() - started, 1),
            **summary,
            "output_tail": out[-_OUTPUT_TAIL_CHARS:] if not ok else out[-500:],
        })
    return {"framework": "declared-suites", "ok": all_ok,
            "suites": results, **totals}


# ── Conventions (tier 3) ─────────────────────────────────────────────────────

def _detect_convention(root: str) -> tuple[str, str] | None:
    """Return (label, shell_command) for make test / just test / scripts/test.sh."""
    mk = os.path.join(root, "Makefile")
    if os.path.exists(mk):
        try:
            content = open(mk, encoding="utf-8", errors="replace").read()
            if re.search(r"^test\s*:", content, flags=re.MULTILINE):
                return "make", "make test"
        except OSError:
            pass
    for jf in ("justfile", "Justfile", ".justfile"):
        p = os.path.join(root, jf)
        if os.path.exists(p):
            try:
                content = open(p, encoding="utf-8", errors="replace").read()
                if re.search(r"^@?test(\s+[\w-]+)*\s*:", content, flags=re.MULTILINE):
                    return "just", "just test"
            except OSError:
                pass
    for rel in ("scripts/test.sh", "script/test", "bin/test"):
        p = os.path.join(root, rel)
        if os.access(p, os.X_OK) and os.path.isfile(p):
            return "script", f"./{rel}"
    return None


# ── Result parsing ───────────────────────────────────────────────────────────

_PYTEST_TAIL_RE = re.compile(
    r"(?:(?P<failed>\d+) failed)?(?:, )?(?:(?P<passed>\d+) passed)?"
    r"(?:, )?(?:(?P<skipped>\d+) skipped)?(?:, )?(?:(?P<errors>\d+) errors?)?"
)


def _parse_pytest(output: str) -> dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for line in reversed(output.splitlines()):
        line = line.strip().strip("=").strip()
        if not line or not any(w in line for w in ("passed", "failed", "error", "skipped")):
            continue
        found = False
        for key in counts:
            m = re.search(rf"(\d+) {key.rstrip('s')}s?\b", line)
            if m:
                counts[key] = int(m.group(1))
                found = True
        if found:
            break
    failures = re.findall(r"^(?:FAILED|ERROR) (\S+)", output, flags=re.MULTILINE)
    return {**counts, "failures": failures[:50]}


def _parse_go(output: str) -> dict[str, Any]:
    failures = re.findall(r"^--- FAIL: (\S+)", output, flags=re.MULTILINE)
    passed = len(re.findall(r"^--- PASS: ", output, flags=re.MULTILINE))
    return {"passed": passed, "failed": len(failures), "skipped": 0, "errors": 0,
            "failures": failures[:50]}


def _parse_cargo(output: str) -> dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed; .*?(\d+) ignored",
                         output):
        counts["passed"] += int(m.group(1))
        counts["failed"] += int(m.group(2))
        counts["skipped"] += int(m.group(3))
    failures = re.findall(r"^---- (\S+) stdout ----", output, flags=re.MULTILINE)
    return {**counts, "failures": failures[:50]}


def _parse_flexible(output: str, returncode: int) -> dict[str, Any]:
    """Best-effort parse for commands of unknown framework (verify.command,
    make/just/script). Tries the pytest summary shape; if nothing matched,
    falls back to returncode-only."""
    s = _parse_pytest(output)
    if any(s[k] for k in ("passed", "failed", "skipped", "errors")) or s["failures"]:
        return s
    return {"passed": 0, "failed": 0 if returncode == 0 else 1,
            "skipped": 0, "errors": 0, "failures": [],
            "parse_note": "output not parsed; see returncode/output_tail"}


def _summarize(framework: str, output: str, returncode: int) -> dict[str, Any]:
    if framework in ("pytest", "unittest"):
        s = _parse_pytest(output) if framework == "pytest" else {
            "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "failures": []}
        if framework == "unittest":
            m = re.search(r"Ran (\d+) tests?", output)
            total = int(m.group(1)) if m else 0
            fails = re.findall(r"^(?:FAIL|ERROR): (\S+)", output, flags=re.MULTILINE)
            s["failed"] = len(fails)
            s["failures"] = fails[:50]
            s["passed"] = max(0, total - s["failed"])
        return s
    if framework == "go":
        return _parse_go(output)
    if framework == "cargo":
        return _parse_cargo(output)
    # npm and anything else: no reliable universal format.
    return {"passed": 0, "failed": 0 if returncode == 0 else 1,
            "skipped": 0, "errors": 0, "failures": [],
            "parse_note": "framework output not parsed; see output_tail + returncode"}


def _run(argv_or_cmd, cwd: str, timeout_s: int, shell: bool,
         env: dict | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            argv_or_cmd, shell=shell, cwd=cwd, env=env,
            capture_output=True, text=True, timeout=timeout_s,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return 124, f"[run_tests] timed out after {timeout_s}s\n{out}{err}"
    except FileNotFoundError as e:
        return 127, f"[run_tests] command not found: {e}"


@register(
    "run_tests",
    {
        "description": (
            "Run the project's test suites and return a parsed summary (pass/fail/skip "
            "counts + failing test ids) instead of raw output. Resolution: declared "
            "[[tests.suites]] from agent.toml (polyglot/monorepo aware — each suite has "
            "its own dir/command/env) → configured verify command → make/just/script "
            "conventions → framework auto-detect (pytest/cargo/go/npm/unittest, using "
            "the project venv when present). No args = full default suite set. "
            "Prefer this over run_argv for tests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suite": {
                    "type": "string",
                    "description": (
                        "Name of a declared suite to run (exact or substring). "
                        "Empty = all default suites (or auto-detect when none declared)."
                    ),
                },
                "pattern": {
                    "type": "string",
                    "description": (
                        "Test subset for detected frameworks: a path (tests/unit/test_x.py) "
                        "or keyword/name expression (pytest -k / go -run / cargo filter). "
                        "Also matches declared suite names."
                    ),
                },
                "framework": {
                    "type": "string",
                    "enum": ["auto", "pytest", "unittest", "cargo", "go", "npm"],
                    "description": "Force a framework, bypassing suites/conventions. Default auto.",
                },
                "path": {
                    "type": "string",
                    "description": "Project directory. Default: working dir.",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": f"Kill each run after this many seconds (default {_DEFAULT_TIMEOUT_S}).",
                },
            },
            "required": [],
        },
    },
)
async def run_tests(suite: str = "", pattern: str = "", framework: str = "auto",
                    path: str = "", timeout_s: int = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(path.strip() or working_dir(_config)))
    if not os.path.isdir(root):
        return {"error": f"not a directory: {root}"}
    timeout_s = max(5, min(int(timeout_s or _DEFAULT_TIMEOUT_S), 1800))
    suite = (suite or "").strip()
    pattern = (pattern or "").strip()

    fw = (framework or "auto").strip().lower()
    shell = False
    if fw == "auto":
        # ── Tier 1: declared suites ─────────────────────────────────────────
        declared = list(getattr(getattr(_config, "tests", None), "suites", None) or [])
        if declared:
            selected = _select_suites(declared, suite or pattern)
            if suite and not selected:
                return {"error": f"no declared suite matches {suite!r}",
                        "available_suites": [s.name for s in declared]}
            if selected:
                return await _run_suites(selected, root, timeout_s)
            # pattern matched no suite name → fall through to detection tiers

        # ── Tier 2: verify.command (full runs only; it cannot filter) ──────
        verify_cmd = getattr(getattr(_config, "verify", None), "command", "") or ""
        if verify_cmd and not pattern:
            started = time.monotonic()
            rc, out = await asyncio.to_thread(_run, verify_cmd, root, timeout_s, True)
            summary = _parse_flexible(out, rc)
            return {
                "framework": "verify.command",
                "command": verify_cmd,
                "returncode": rc,
                "ok": rc == 0,
                "duration_s": round(time.monotonic() - started, 1),
                **summary,
                "output_tail": out[-_OUTPUT_TAIL_CHARS:],
            }

        # ── Tier 3: conventions (full runs only) ────────────────────────────
        if not pattern:
            conv = _detect_convention(root)
            if conv is not None:
                label, cmd = conv
                started = time.monotonic()
                rc, out = await asyncio.to_thread(_run, cmd, root, timeout_s, True)
                summary = _parse_flexible(out, rc)
                return {
                    "framework": f"convention:{label}",
                    "command": cmd,
                    "returncode": rc,
                    "ok": rc == 0,
                    "timed_out": rc == 124,
                    "duration_s": round(time.monotonic() - started, 1),
                    **summary,
                    "output_tail": out[-_OUTPUT_TAIL_CHARS:],
                }

        # ── Tier 4: framework heuristics ────────────────────────────────────
        fw = detect_framework(root)
        if fw is None:
            return {"error": "no test setup found: no [[tests.suites]] declared, no "
                             "verify command, no make/just/script convention, and no "
                             "pytest config, test_*.py, Cargo.toml, go.mod, or npm test "
                             "script detected. Declare suites in agent.toml or force "
                             "framework=..."}

    try:
        argv = _build_argv(fw, root, pattern.strip())
    except ValueError as e:
        return {"error": str(e)}

    started = time.monotonic()
    rc, out = await asyncio.to_thread(_run, argv, root, timeout_s, shell)
    summary = _summarize(fw, out, rc)
    return {
        "framework": fw,
        "command": " ".join(argv),
        "returncode": rc,
        "ok": rc == 0,
        "timed_out": rc == 124,
        "duration_s": round(time.monotonic() - started, 1),
        **summary,
        "output_tail": out[-_OUTPUT_TAIL_CHARS:],
    }
