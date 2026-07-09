"""`review_changes` tool — pre-commit second opinion on the working diff.

Composes two existing layers into one report:

1. Static floor: ``agent.security.secaudit`` diff-only scan (secrets, hygiene,
   SAST when semgrep/bandit are on PATH). Offline, deterministic.
2. LLM review: the diff (plus static findings) is sent to the strongest
   *available* model entry — preferably a different endpoint than the model
   currently driving the turn, so the review is a genuine second opinion
   rather than self-review.

If no reviewer endpoint is reachable the tool still returns the static
findings, marked ``degraded: true``.

Storage/network: read-only git commands in the working dir; one chat
completion to a configured model entry. Nothing is written.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from typing import TYPE_CHECKING, Any

from agent.tools import register
from agent.tools._common import working_dir

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_config: "Config | None" = None

# Diff budget sent to the reviewer. Beyond this the diff is truncated
# file-by-file (whole files dropped, never mid-hunk) and the report says so.
_MAX_DIFF_CHARS = 30_000
# Cap on untracked-file content included as pseudo-diff.
_MAX_UNTRACKED_CHARS = 4_000

_REVIEW_SYSTEM_PROMPT = """\
You are a strict senior code reviewer. You receive a unified diff of pending
changes plus findings from a static security scanner. Review ONLY the changed
code. Look for:
- bugs: logic errors, off-by-one, wrong operator, inverted condition, races
- edge cases: empty input, None, error paths, resource leaks
- API misuse and violated invariants visible in the diff context
- anti-patterns: dead code, copy-paste drift, swallowed exceptions

Do NOT restate the static findings; they are already in the report.
Do NOT comment on style, formatting, or naming unless it hides a bug.
If the diff looks correct, say so — do not invent findings.

Respond with ONLY a JSON object, no prose, no code fences:
{"summary": "<1-3 sentence verdict>",
 "findings": [{"file": "<path>", "line": <int or null>,
               "severity": "high|medium|low",
               "kind": "bug|edge-case|api-misuse|anti-pattern",
               "note": "<problem and concrete fix, one line>"}]}
"""


def setup(config: "Config") -> None:
    global _config
    _config = config


def _run_git(*args: str, cwd: str, timeout: float = 30.0) -> tuple[str, int]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        result = subprocess.run(
            ["git"] + list(args),
            cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        return "", 124
    return result.stdout, result.returncode


def _collect_diff(cwd: str, staged: bool) -> tuple[str, list[str]]:
    """Return (diff_text, notes). Unstaged mode appends untracked files as
    pseudo-diffs so brand-new files are reviewed too."""
    notes: list[str] = []
    if staged:
        out, rc = _run_git("diff", "--cached", cwd=cwd)
        return (out if rc == 0 else ""), notes

    out, rc = _run_git("diff", cwd=cwd)
    diff = out if rc == 0 else ""
    untracked, rc = _run_git("ls-files", "--others", "--exclude-standard", cwd=cwd)
    if rc == 0:
        for rel in untracked.splitlines():
            p = os.path.join(cwd, rel)
            try:
                if os.path.getsize(p) > 200_000:
                    notes.append(f"untracked file skipped (too large): {rel}")
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    content = fh.read(_MAX_UNTRACKED_CHARS + 1)
            except OSError:
                continue
            truncated = len(content) > _MAX_UNTRACKED_CHARS
            body = content[:_MAX_UNTRACKED_CHARS]
            diff += (
                f"\n--- /dev/null\n+++ b/{rel} (untracked, new file"
                f"{', truncated' if truncated else ''})\n"
                + "".join(f"+{line}\n" for line in body.splitlines())
            )
    return diff, notes


def _truncate_by_file(diff: str, budget: int) -> tuple[str, list[str]]:
    """Trim the diff to *budget* chars by dropping whole per-file sections."""
    if len(diff) <= budget:
        return diff, []
    sections: list[str] = []
    current: list[str] = []
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git") or line.startswith("--- /dev/null"):
            if current:
                sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    for sec in sections:
        if used + len(sec) <= budget:
            kept.append(sec)
            used += len(sec)
        else:
            first = sec.splitlines()[0] if sec else "?"
            dropped.append(f"diff section not reviewed (size budget): {first.strip()}")
    return "".join(kept), dropped


def _static_findings(cwd: str) -> tuple[list[dict], list[str]]:
    from dataclasses import asdict
    try:
        from agent.security import secaudit
        res = secaudit.scan(cwd, diff_only=True)
        return [asdict(f) for f in res.findings], list(res.scanners_missing)
    except Exception as e:
        logger.warning("review_changes: secaudit failed: %s", e)
        return [], []


def _pick_reviewer(config: "Config") -> tuple[str, Any] | None:
    """Pick the reviewing entry. Preference: a live entry on a DIFFERENT
    endpoint that is at least as strong as the active model. A weaker reviewer
    critiquing a stronger author generates noise findings, so when no equal-or-
    stronger alternative exists we fall back to the strongest entry overall —
    usually the active model itself, which the caller flags as self_review.
    Returns (entry_name, ModelEntry) or None."""
    from agent.core.model_tier import build_ladder
    try:
        ladder = build_ladder(config)  # weakest → strongest
    except Exception:
        return None
    if not ladder:
        return None
    active = (config.llm.base_url, config.llm.model)

    def _is_active(e) -> bool:
        return (e.base_url, e.model or config.llm.model) == active

    active_power = 0.0
    for name, p in ladder:
        e = config.model_entries.get(name)
        if e is not None and _is_active(e):
            active_power = max(active_power, p)
    for name, p in reversed(ladder):  # strongest first
        e = config.model_entries.get(name)
        if e is None or _is_active(e):
            continue
        if p >= active_power:
            return name, e
        break  # strongest non-active is weaker than the author — don't use it
    # Flagged self-review beats a weaker second opinion.
    name = ladder[-1][0]
    e = config.model_entries.get(name)
    return (name, e) if e is not None else None


def _parse_review(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj = json.loads(text[start:text.rfind("}") + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    findings = obj.get("findings")
    obj["findings"] = findings if isinstance(findings, list) else []
    return obj


async def _llm_review(entry, diff: str, static: list[dict]) -> tuple[dict | None, str | None]:
    """Return (parsed_report, error). One completion, low temperature."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key or "local")
    static_block = ""
    if static:
        brief = [
            f"- {f['severity']}: {f['path']}:{f.get('line') or '?'} {f['message']}"
            for f in static[:30]
        ]
        static_block = "\n\nStatic scanner findings (context, do not restate):\n" + "\n".join(brief)
    try:
        resp = await client.chat.completions.create(
            model=entry.model,
            temperature=0.2,
            max_tokens=min(int(entry.max_output_tokens or 2048), 4096),
            messages=[
                {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"Diff to review:\n\n{diff}{static_block}"},
            ],
        )
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    parsed = _parse_review(text)
    if parsed is None:
        # Model ignored the JSON contract — keep its prose as the summary.
        return {"summary": text.strip()[:2000], "findings": [],
                "format_note": "reviewer did not return valid JSON"}, None
    return parsed, None


@register(
    "review_changes",
    {
        "description": (
            "Pre-commit second opinion on pending changes. Collects the git diff "
            "(staged or unstaged+untracked), runs the deterministic security scan on "
            "changed files, then has the strongest available OTHER model review the "
            "diff for bugs, edge cases, API misuse, and anti-patterns. Use BEFORE "
            "committing non-trivial edits. Returns a structured report; degrades to "
            "static-only findings when no reviewer endpoint is reachable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "Review `git diff --cached` instead of unstaged changes. Default false.",
                },
                "path": {
                    "type": "string",
                    "description": "Repo directory to review. Default: working dir.",
                },
            },
            "required": [],
        },
    },
)
async def review_changes(staged: bool = False, path: str = "") -> dict[str, Any]:
    cwd = os.path.abspath(os.path.expanduser(path.strip() or working_dir(_config)))
    if not os.path.isdir(os.path.join(cwd, ".git")):
        probe, rc = await asyncio.to_thread(_run_git, "rev-parse", "--git-dir", cwd=cwd)
        if rc != 0:
            return {"error": f"not a git repository: {cwd}"}

    diff, notes = await asyncio.to_thread(_collect_diff, cwd, bool(staged))
    if not diff.strip():
        which = "staged" if staged else "unstaged"
        return {"summary": f"No {which} changes to review.", "findings": [],
                "static_findings": [], "degraded": False}

    diff, dropped = _truncate_by_file(diff, _MAX_DIFF_CHARS)
    notes.extend(dropped)

    static, scanners_missing = await asyncio.to_thread(_static_findings, cwd)

    report: dict[str, Any] = {
        "mode": "staged" if staged else "unstaged",
        "static_findings": static,
        "scanners_missing": scanners_missing,
        "notes": notes,
        "degraded": False,
    }

    if _config is None:
        report.update(degraded=True, summary="No config; static findings only.", findings=[])
        return report

    picked = _pick_reviewer(_config)
    if picked is None:
        report.update(
            degraded=True, findings=[],
            summary="No reviewer model reachable; static findings only.",
        )
        return report

    entry_name, entry = picked
    self_review = (entry.base_url, entry.model or _config.llm.model) == \
                  (_config.llm.base_url, _config.llm.model)
    parsed, err = await _llm_review(entry, diff, static)
    if parsed is None:
        report.update(
            degraded=True, findings=[],
            summary=f"Reviewer call failed ({err}); static findings only.",
        )
        return report

    report.update(
        summary=parsed.get("summary", ""),
        findings=parsed.get("findings", []),
        reviewed_by=entry_name,
        self_review=self_review,
    )
    if "format_note" in parsed:
        report["notes"] = report["notes"] + [parsed["format_note"]]
    if self_review:
        report["notes"] = report["notes"] + [
            "reviewer is the same endpoint as the active model — weaker signal"
        ]
    return report
