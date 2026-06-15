"""LLM deep-read vulnerability audit (Tier-2 #21).

The rest of the suite is a deterministic FLOOR (regex/SAST) plus an LLM that only
*triages* what the scanners already found. That floor is blind to whole vulnerability
classes — memory-safety bugs in C, logic flaws, subtle injection — because there is no
pattern to match. This module fills that gap: it feeds source code DIRECTLY to the local
LLM and asks it to find vulnerabilities, the way a human auditor reads code.

This is the capability that got Mythos pulled ("analyze code for vulnerabilities"). Here
it is scoped to DEFENSIVE self-audit of your own / authorized code: the model reports
weaknesses so they can be fixed, it does not write exploits. Findings are clearly tagged
LLM-REPORTED (unverified) — a weak local model misses subtle bugs and invents some, so
each hit should be confirmed (e.g. `/security verify`) before trusting it.

Code is reviewed in overlapping line windows so a large file (libyaml's scanner.c) is
covered, not just its head. Bounded by a window cap to keep cost finite. Offline if the
LLM endpoint is local; air-gap aware.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SRC_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".go", ".rs", ".py", ".js", ".ts",
    ".tsx", ".jsx", ".java", ".kt", ".rb", ".php", ".vala", ".swift", ".m", ".mm",
}
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build",
              "dist", "target", "vendor", ".agent", "test", "tests", "testdata"}

_WINDOW_LINES = 320
_OVERLAP = 30
_MAX_WINDOWS = 50          # hard cap on LLM calls per review
_MAX_OUTPUT_TOKENS = 1100

_SYSTEM = (
    "You are a security auditor reading source code to find real, exploitable "
    "vulnerabilities so the owner can FIX them. Focus on high-impact classes: memory "
    "safety (buffer/heap/stack overflow, OOB read/write, use-after-free, integer "
    "overflow leading to undersized allocation), injection (command/SQL/path), unsafe "
    "deserialization, missing bounds/length checks, signedness bugs, format strings, "
    "and auth/crypto misuse.\n\n"
    "You are shown a NUMBERED window of one file. Report only concrete issues you can "
    "point to a line for. Do NOT write exploit code. Output STRICT JSON: a list of "
    '{"line": <int>, "severity": "critical|high|medium|low", "class": "<short>", '
    '"detail": "<one sentence: the bug and why it is exploitable>"}. '
    "If you see nothing credible, output []. JSON only, no prose, no markdown fences."
)


def _select_files(target: str) -> list[Path]:
    root = Path(target)
    if root.is_file():
        return [root]
    out = []
    for dp, dirnames, filenames in __import__("os").walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for fn in filenames:
            if Path(fn).suffix.lower() in _SRC_EXTS:
                out.append(Path(dp) / fn)
    # Largest files first — more code, likelier to hold the interesting bug.
    out.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return out


def _windows(lines: list[str]):
    n = len(lines)
    if n <= _WINDOW_LINES:
        yield 1, lines
        return
    start = 0
    while start < n:
        end = min(n, start + _WINDOW_LINES)
        yield start + 1, lines[start:end]
        if end == n:
            break
        start = end - _OVERLAP


def _parse(out: str) -> list[dict]:
    out = out.strip()
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n", "", out)
        out = re.sub(r"\n```$", "", out).strip()
    try:
        data = json.loads(out)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        # Best-effort: grab the first JSON array in the text.
        m = re.search(r"\[.*\]", out, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        return []


async def _review_window(client, model, rel: str, base_line: int, chunk: list[str]) -> list[dict]:
    numbered = "\n".join(f"{base_line + i}: {ln}" for i, ln in enumerate(chunk))
    user = f"File: {rel} (lines {base_line}-{base_line + len(chunk) - 1})\n```\n{numbered}\n```"
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        max_tokens=_MAX_OUTPUT_TOKENS,
        temperature=0.1,
    )
    raw = (resp.choices[0].message.content or "") if resp.choices else ""
    issues = []
    for it in _parse(raw):
        if not isinstance(it, dict) or "line" not in it:
            continue
        issues.append({
            "file": rel,
            "line": it.get("line"),
            "severity": str(it.get("severity", "medium")).lower(),
            "class": str(it.get("class", "?"))[:40],
            "detail": str(it.get("detail", ""))[:300],
        })
    return issues


def estimate(target: str) -> tuple[int, int]:
    """Cheap pre-scan (no LLM): (source file count, windows that will be sent).

    Windows are capped at _MAX_WINDOWS — the returned count reflects what will
    actually run, so the UI banner is honest.
    """
    files = _select_files(target)
    windows = 0
    for f in files:
        try:
            n = len(f.read_text(errors="ignore").splitlines())
        except OSError:
            continue
        windows += sum(1 for _ in _windows([""] * max(n, 1)))
        if windows >= _MAX_WINDOWS:
            return len(files), _MAX_WINDOWS
    return len(files), windows


def _state_path(config) -> Path:
    root = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()
    ad = Path(getattr(getattr(config, "tools", None), "agent_dir", ".agent") or ".agent")
    base = ad if ad.is_absolute() else root / ad
    return base / "security" / "review_state.json"


def _load_state(config) -> dict:
    p = _state_path(config)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(config, state: dict) -> None:
    p = _state_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=0))


async def review(config, target: str, *, incremental: bool = False, on_progress=None) -> str:
    """Deep-read audit of *target* (file or dir). Returns Markdown. Never raises.

    incremental: review only files new or modified since the last review (state in
    .agent/security/review_state.json). on_progress(msg): optional per-window callback.
    """
    import time as _time
    from contextlib import asynccontextmanager
    from openai import AsyncOpenAI
    from agent.config import make_registry
    from agent.security import airgap

    # Drive the bottom 'sec' status indicator while the model works. Guarded:
    # importing agent.core pulls modules that need a real openai; fall back to a
    # no-op so review still runs if that import chain is unavailable.
    try:
        from agent.core.model_status import track_async as _track
    except Exception:  # noqa: BLE001
        @asynccontextmanager
        async def _track(role):  # type: ignore
            yield

    def _emit(msg: str) -> None:
        if on_progress:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001
                pass

    try:
        entry = make_registry(config).default
    except Exception as e:  # noqa: BLE001
        return f"(review unavailable: {e})"
    if airgap.is_enabled(config) and not airgap.is_local_url(entry.base_url):
        return "# air-gap: refused — LLM endpoint is non-local"

    files = _select_files(target)
    if not files:
        return f"No source files under {target} (exts: {', '.join(sorted(_SRC_EXTS))})."

    state = _load_state(config) if incremental else {}
    if incremental:
        fresh = []
        for f in files:
            try:
                mt = int(f.stat().st_mtime)
            except OSError:
                continue
            if state.get(str(f.resolve()), {}).get("mtime", -1) < mt:
                fresh.append(f)
        skipped = len(files) - len(fresh)
        files = fresh
        if not files:
            return (f"Incremental review: all {skipped} source file(s) already reviewed "
                    f"since last run. Nothing changed. (Use `/security review .` to force.)")

    # Plan total windows (capped) for an honest w/W denominator.
    planned = 0
    for f in files:
        try:
            n = len(f.read_text(errors="ignore").splitlines())
        except OSError:
            continue
        planned += sum(1 for _ in _windows([""] * max(n, 1)))
    planned = min(planned, _MAX_WINDOWS)

    root = Path(target)
    base = root if root.is_dir() else root.parent
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    findings: list[dict] = []
    windows_used = 0
    files_done = 0
    truncated = False
    state_now = _load_state(config)
    try:
        for f in files:
            if windows_used >= _MAX_WINDOWS:
                truncated = True
                break
            try:
                lines = f.read_text(errors="ignore").splitlines()
            except OSError:
                continue
            rel = str(f.relative_to(base)) if base in f.parents or base == f.parent else str(f)
            before = len(findings)
            for bl, chunk in _windows(lines):
                if windows_used >= _MAX_WINDOWS:
                    truncated = True
                    break
                windows_used += 1
                _emit(f"[{windows_used}/{planned}] {rel}:{bl}  ({len(findings)} issue(s) so far)")
                try:
                    async with _track("sec"):
                        findings += await _review_window(client, entry.model, rel, bl, chunk)
                except Exception:  # noqa: BLE001 - one bad window must not abort the run
                    continue
            files_done += 1
            # Mark this file reviewed (mtime now) so incremental skips it next time.
            try:
                state_now[str(f.resolve())] = {
                    "mtime": int(f.stat().st_mtime),
                    "reviewed_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    "issues": len(findings) - before,
                }
            except OSError:
                pass
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001
            pass
        _save_state(config, state_now)

    # Dedupe + sort by severity.
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    seen, uniq = set(), []
    for it in findings:
        k = (it["file"], it["line"], it["class"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)
    uniq.sort(key=lambda x: (order.get(x["severity"], 9), x["file"], x["line"] or 0))

    lines_out = [
        f"# LLM vulnerability review — {Path(target).resolve()}",
        f"- files reviewed: {files_done}/{len(files)}  windows: {windows_used}"
        + ("  (window cap hit — not all code seen)" if truncated else ""),
        f"- reported issues: {len(uniq)}",
        "",
        "> **LLM-REPORTED, UNVERIFIED.** A local model misses real bugs and invents "
        "some. Treat as leads, not facts — confirm each with code review or "
        "`/security verify`. This complements the deterministic scan; it does not "
        "replace it.",
        "",
    ]
    if not uniq:
        lines_out.append("No vulnerabilities reported by the model (NOT proof of safety).")
        return "\n".join(lines_out)
    lines_out += ["| sev | class | location | detail |", "|---|---|---|---|"]
    for it in uniq:
        loc = f"{it['file']}:{it['line']}"
        detail = it["detail"].replace("|", "/").replace("\n", " ")
        lines_out.append(f"| {it['severity']} | {it['class']} | {loc} | {detail} |")
    return "\n".join(lines_out)


def _resolve_target(config, arg: str):
    """Resolve a review target, confined to granted paths. Returns (path, error).

    Empty / '.' → project working_dir. Relative → resolved against working_dir.
    Anything outside a path grant (default = project root) is refused, so the agent
    cannot be pointed at '/' or arbitrary trees without explicit acceptance.
    """
    import os
    workdir = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".").resolve()
    raw = arg.strip()
    if raw in ("", "."):
        return workdir, None
    p = Path(os.path.expanduser(raw))
    if not p.is_absolute():
        p = workdir / p
    try:
        p = p.resolve()
    except OSError:
        return None, f"bad path: {raw}"
    if not p.exists():
        return None, f"path not found: {raw}"
    try:
        from agent.security import policy, path_grants
        if policy.is_configured() and path_grants.grant_for(p) is None:
            return None, (f"'{p}' is outside the project and not granted.\n"
                          f"Grant read access first:  /paths add {p} ro")
    except Exception:  # noqa: BLE001 - policy not set up (e.g. tests): allow
        pass
    return p, None


def run_review_command(config, arg: str, on_progress=None) -> str:
    """Sync handler for `/security review [<path>]` (safe from a running event loop).

    No arg → incremental review of the project (only files changed since last run).
    '.' or a path → full review of that location (confined to granted paths).
    """
    import asyncio

    incremental = arg.strip() == ""
    target, err = _resolve_target(config, arg)
    if err:
        return f"Error: {err}"
    return asyncio.run(review(config, str(target), incremental=incremental, on_progress=on_progress))
