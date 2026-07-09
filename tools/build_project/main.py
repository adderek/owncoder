"""`build_project` tool — run the project's build and return parsed errors.

The invocation itself is trivial (build systems are self-describing — the
Makefile/build.gradle IS the declaration, so unlike run_tests there is no
config section). The value is deterministic error extraction: a gradle or
meson log is hundreds of lines; the model needs `{file, line, message}`
tuples it can feed straight into edit_file, not a wall of output.

Detection order (first hit wins, unless `system` is forced):
Makefile → justfile → meson.build → CMakeLists.txt → build.gradle(.kts) /
gradlew → Cargo.toml → go.mod → package.json build script.
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

_DEFAULT_TIMEOUT_S = 600
_OUTPUT_TAIL_CHARS = 4_000
_MAX_ERRORS = 50


def setup(config: "Config") -> None:
    global _config
    _config = config


def detect_build_system(root: str) -> tuple[str, list[str]] | None:
    """Return (label, argv) for the first matching build system."""
    def has(name: str) -> bool:
        return os.path.exists(os.path.join(root, name))

    if has("Makefile") or has("makefile"):
        return "make", ["make"]
    for jf in ("justfile", "Justfile", ".justfile"):
        if has(jf):
            return "just", ["just", "build"]
    if has("meson.build"):
        # Requires a configured build dir; standard convention is ./build.
        if os.path.isdir(os.path.join(root, "build")):
            return "meson", ["ninja", "-C", "build"]
        return "meson", ["meson", "setup", "build"]
    if has("CMakeLists.txt"):
        if os.path.isdir(os.path.join(root, "build")):
            return "cmake", ["cmake", "--build", "build"]
        return "cmake", ["cmake", "-B", "build"]
    if has("gradlew"):
        return "gradle", ["./gradlew", "assemble", "-q"]
    if has("build.gradle") or has("build.gradle.kts"):
        return "gradle", ["gradle", "assemble", "-q"]
    if has("Cargo.toml"):
        return "cargo", ["cargo", "build"]
    if has("go.mod"):
        return "go", ["go", "build", "./..."]
    pkg = os.path.join(root, "package.json")
    if os.path.exists(pkg):
        try:
            import json as _json
            if _json.load(open(pkg, encoding="utf-8")).get("scripts", {}).get("build"):
                return "npm", ["npm", "run", "build", "--silent"]
        except Exception:
            pass
    return None


# ── Error extraction ─────────────────────────────────────────────────────────
# gcc/clang/rustc/vala/meson: file:line:col: error: msg   (col optional)
_GCC_RE = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+)(?::\d+)?:\s*"
    r"(?P<kind>error|warning|fatal error)[:,]\s*(?P<msg>.+)$", re.MULTILINE)
# javac / gradle-java: file.java:12: error: msg
_JAVAC_RE = re.compile(
    r"^(?P<file>\S+\.(?:java|kt|kts)):(?P<line>\d+):\s*"
    r"(?P<kind>error|warning):\s*(?P<msg>.+)$", re.MULTILINE)
# kotlin daemon style: e: file:12:5 msg   /  w: file:...
_KOTLIN_RE = re.compile(
    r"^(?P<kind>[ew]): (?P<file>\S+?):(?P<line>\d+):\d+ (?P<msg>.+)$", re.MULTILINE)
# go: file.go:12:5: msg (no error keyword)
_GO_RE = re.compile(
    r"^(?P<file>\S+\.go):(?P<line>\d+)(?::\d+)?: (?P<msg>.+)$", re.MULTILINE)


def extract_errors(output: str) -> list[dict]:
    seen: set[tuple] = set()
    errors: list[dict] = []

    def add(file: str, line: str, kind: str, msg: str) -> None:
        kind = {"e": "error", "w": "warning", "fatal error": "error"}.get(kind, kind)
        key = (file, line, msg[:80])
        if key in seen:
            return
        seen.add(key)
        errors.append({"file": file, "line": int(line), "kind": kind,
                       "message": msg.strip()[:300]})

    for rx in (_GCC_RE, _JAVAC_RE, _KOTLIN_RE, _GO_RE):
        for m in rx.finditer(output):
            add(m.group("file"), m.group("line"),
                m.groupdict().get("kind", "error"), m.group("msg"))
    # Errors first, then warnings; stable within each group.
    errors.sort(key=lambda e: e["kind"] != "error")
    return errors[:_MAX_ERRORS]


def _run(argv: list[str], cwd: str, timeout_s: int) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout_s)
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace")
        err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
        return 124, f"[build_project] timed out after {timeout_s}s\n{out}{err}"
    except FileNotFoundError as e:
        return 127, f"[build_project] command not found: {e}"


@register(
    "build_project",
    {
        "description": (
            "Build the project and return compiler errors as {file, line, message} "
            "tuples instead of a raw log — feed them straight into edit_file. "
            "Auto-detects the build system (make, just, meson/ninja, cmake, gradle, "
            "cargo, go, npm build script). Use `target` for a specific make/just/"
            "gradle target. Prefer this over run_argv for builds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Build target (make/just/gradle). Empty = default target.",
                },
                "system": {
                    "type": "string",
                    "enum": ["auto", "make", "just", "meson", "cmake", "gradle",
                             "cargo", "go", "npm"],
                    "description": "Force a build system. Default auto-detect.",
                },
                "path": {
                    "type": "string",
                    "description": "Project directory. Default: working dir.",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": f"Kill the build after this many seconds (default {_DEFAULT_TIMEOUT_S}).",
                },
            },
            "required": [],
        },
    },
)
async def build_project(target: str = "", system: str = "auto",
                        path: str = "", timeout_s: int = _DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(path.strip() or working_dir(_config)))
    if not os.path.isdir(root):
        return {"error": f"not a directory: {root}"}
    timeout_s = max(10, min(int(timeout_s or _DEFAULT_TIMEOUT_S), 3600))
    target = (target or "").strip()

    detected = detect_build_system(root)
    sysname = (system or "auto").strip().lower()
    if sysname == "auto":
        if detected is None:
            return {"error": "no build system detected (Makefile, justfile, meson.build, "
                             "CMakeLists.txt, gradle, Cargo.toml, go.mod, npm build script). "
                             "Force one with system=..."}
        label, argv = detected
    else:
        if detected is not None and detected[0] == sysname:
            label, argv = detected
        else:
            fallbacks = {
                "make": ["make"], "just": ["just", "build"],
                "meson": ["ninja", "-C", "build"], "cmake": ["cmake", "--build", "build"],
                "gradle": ["gradle", "assemble", "-q"], "cargo": ["cargo", "build"],
                "go": ["go", "build", "./..."], "npm": ["npm", "run", "build", "--silent"],
            }
            if sysname not in fallbacks:
                return {"error": f"unsupported build system: {sysname}"}
            label, argv = sysname, fallbacks[sysname]

    if target:
        if label in ("make", "just"):
            argv = argv[:1] + [target]
        elif label == "gradle":
            argv = [argv[0], target, "-q"]
        elif label == "cargo":
            argv = ["cargo", "build", "-p", target]
        else:
            return {"error": f"`target` is not supported for {label}; "
                             "run the default build or use run_argv"}

    started = time.monotonic()
    rc, out = await asyncio.to_thread(_run, argv, root, timeout_s)
    errors = extract_errors(out)
    n_err = sum(1 for e in errors if e["kind"] == "error")
    return {
        "system": label,
        "command": " ".join(argv),
        "returncode": rc,
        "ok": rc == 0,
        "timed_out": rc == 124,
        "duration_s": round(time.monotonic() - started, 1),
        "error_count": n_err,
        "errors": errors,
        "output_tail": out[-_OUTPUT_TAIL_CHARS:] if rc != 0 else out[-500:],
    }
