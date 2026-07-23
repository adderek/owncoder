"""LLM-as-judge scorer for the eval harness.

Scores the *quality* of a passing (or failing) solution along dimensions the
mechanical checks can't see: minimal diff, style match, no drive-by changes.

Design rules (see PLAN_STRONG S2):

- Mechanical pass/fail stays primary. The judged 0-10 score is secondary and
  never turns a mechanical fail into a pass.
- The judge sees ONLY the task prompt and the unified diff of the workspace —
  never the candidate agent's own transcript or commentary, so a candidate
  cannot talk its way to a higher score.
- "Touched a forbidden file" is checked mechanically from the diff's file
  list, never judged.
- Oversized diffs are clamped mechanically (score cap) before the judge ever
  runs — a length-penalty the judge can't be argued out of.
- The LLM call goes through ``call_role_with_failover(config, "judge", …)``
  so judging is failover-safe and pinnable to the strongest configured model
  via ``[model_roles] judge = "<entry>"``.

This module lazily imports the agent package inside the functions that need
it, so the harness itself stays runnable fully offline (``--agent-cmd`` smoke
tests) without an agent config present.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

# Diffs longer than this are scored at most OVERSIZED_SCORE_CAP regardless of
# what the judge says. Tasks can override via judge.max_diff_lines.
DEFAULT_MAX_DIFF_LINES = 400
OVERSIZED_SCORE_CAP = 4

# Hard cap on diff text handed to the judge model (chars).
_DIFF_PROMPT_CAP = 20_000

JUDGE_SYSTEM_PROMPT = """\
You are a strict code-review judge scoring an automated coding agent's work.
You are given the task the agent was asked to do and the unified diff of every
change it made to the workspace. You see nothing else — judge the diff, not
intentions.

Score 0-10 using the rubric provided. Respond with ONLY a JSON object:
{"score": <int 0-10>, "rationale": "<one or two short sentences>"}

Scoring principles that apply to every rubric:
- A correct, minimal, idiomatic change scores high.
- Unrelated "drive-by" edits (reformatting untouched lines, renaming things
  the task didn't ask about, adding features nobody requested) lose points
  even if harmless.
- Deleting or weakening tests/checks to make things pass scores 0-2.
- An empty diff for a task that requires changes scores 0.
- Do not reward verbosity: added comments, docstrings, or defensive code the
  task didn't ask for are drive-by changes, not bonuses."""

# Rubric anchors + two compact calibration examples per task type. The
# examples are deliberately small so they calibrate without dominating the
# context of weak judge models.
RUBRICS: dict[str, str] = {
    "fix": """\
Task type: FIX — repair a specific defect.
10: defect fixed at its root cause; diff touches only the faulty logic; style
    matches surrounding code.
 8: correct fix, one minor blemish (slightly wider edit or small style drift).
 5: defect suppressed rather than fixed (symptom patch, broad try/except), or
    correct fix buried in unrelated edits.
 3: partially fixed, or fix plus a plausible new defect.
 0: not fixed, tests weakened/deleted, or empty diff.

Calibration A (score 9): task "fix off-by-one dropping last element"; diff is
one line, `range(len(xs) - 1)` -> `range(len(xs))`. Minimal, root cause.
Calibration B (score 4): same task; diff wraps the loop body in try/except,
appends the last element manually after the loop, and reformats the whole
file. Works, but symptom-patched and full of drive-by churn.""",
    "edit": """\
Task type: EDIT — add or change functionality to a spec.
10: implements exactly the spec; smallest reasonable diff; matches file's
    existing naming/style; nothing beyond the ask.
 8: spec met with minor excess (an extra helper or comment not asked for).
 5: spec met but with substantial unrequested additions or restructuring.
 3: spec partially met, or met while breaking existing conventions.
 0: spec not met, or existing behavior the task relied on was removed.

Calibration A (score 9): task "add a --quiet flag suppressing the banner";
diff adds one argparse line and one `if not args.quiet:` guard.
Calibration B (score 5): same task; flag works but the diff also rewrites
argument parsing to click, renames two functions, and adds a config file
loader nobody asked for.""",
    "locate": """\
Task type: LOCATE — find/answer something about the code; the deliverable is
an answer artifact (file/text), with little or no code change expected.
10: answer exactly correct and specific; no code modified beyond the required
    answer artifact.
 8: answer correct with minor imprecision or harmless extra content.
 5: answer partially correct, or correct but source files were edited
    needlessly.
 3: answer vague/mostly wrong.
 0: wrong answer or unrelated destructive edits.

Calibration A (score 10): task "name the function that loads config, write it
to ANSWER.txt"; diff adds ANSWER.txt containing `load_config`. Nothing else.
Calibration B (score 5): ANSWER.txt has the right name inside a paragraph of
speculation, and the diff also 'cleaned up' imports in the source file.""",
}

DEFAULT_TASK_TYPE = "fix"


# ── Mechanical diff handling ──────────────────────────────────────────────


def compute_diff(fixture_dir: Path, workspace: Path) -> str:
    """Unified diff between the pristine fixture and the finished workspace.

    ``.agent`` (harness-created project marker) is excluded. Returns "" when
    nothing changed. Uses ``diff -ruN`` so added/removed files appear too.
    """
    proc = subprocess.run(
        ["diff", "-ruN", "-x", ".agent", "-x", "__pycache__",
         str(fixture_dir), str(workspace)],
        capture_output=True, text=True, timeout=60,
    )
    # diff exits 0 = same, 1 = differences, 2 = trouble
    if proc.returncode >= 2:
        raise RuntimeError(f"diff failed: {proc.stderr.strip()[:300]}")
    out = proc.stdout
    # Normalize temp paths so the judge (and stored results) see stable
    # workspace-relative names instead of /tmp/eval-xyz noise.
    out = out.replace(str(workspace) + "/", "b/").replace(str(fixture_dir) + "/", "a/")
    out = out.replace(str(workspace), "b").replace(str(fixture_dir), "a")
    return out


def diff_stats(diff: str) -> dict:
    """Mechanical stats: files touched and +/- line counts."""
    files: list[str] = []
    added = removed = 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].split("\t")[0]
            path = path[2:] if path.startswith("b/") else path
            if path != "/dev/null":
                files.append(path)
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"files": files, "added": added, "removed": removed,
            "total_lines": added + removed}


def check_forbidden(diff: str, forbid_paths: list[str]) -> list[str]:
    """Return the forbidden paths the diff touched (fnmatch globs). Mechanical
    — this is never delegated to the judge."""
    from fnmatch import fnmatch
    touched = diff_stats(diff)["files"]
    hits = []
    for pattern in forbid_paths:
        for f in touched:
            if fnmatch(f, pattern):
                hits.append(f)
    return sorted(set(hits))


# ── Judge prompt + response parsing ───────────────────────────────────────


def build_judge_messages(task_type: str, prompt: str, diff: str) -> list[dict]:
    """Messages for the judge. Deliberately contains ONLY the task prompt and
    the diff — no candidate commentary, no agent transcript."""
    rubric = RUBRICS.get(task_type, RUBRICS[DEFAULT_TASK_TYPE])
    diff_shown = diff
    if len(diff_shown) > _DIFF_PROMPT_CAP:
        diff_shown = diff_shown[:_DIFF_PROMPT_CAP] + "\n…[diff clipped]…"
    user = (
        f"{rubric}\n\n"
        f"## Task given to the agent\n{prompt}\n\n"
        f"## Unified diff of all changes\n```diff\n{diff_shown}\n```\n\n"
        'Respond with ONLY the JSON object {"score": ..., "rationale": ...}.'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_score(text: str) -> tuple[int, str]:
    """Extract (score, rationale) from a judge response.

    Accepts a bare JSON object, JSON inside a code fence, or falls back to a
    ``"score": N`` regex. Raises ValueError when no score is recoverable.
    Scores are clamped to 0-10.
    """
    candidate = text.strip()
    m = re.search(r"\{.*\}", candidate, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            score = int(data["score"])
            rationale = str(data.get("rationale", "")).strip()
            return max(0, min(10, score)), rationale
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+)', candidate)
    if m:
        return max(0, min(10, int(m.group(1)))), ""
    raise ValueError(f"unparseable judge response: {candidate[:200]!r}")


# ── Judging one task ──────────────────────────────────────────────────────


async def judge_task(config, task: dict, diff: str) -> dict:
    """Score one task's diff. Returns a dict for TaskResult.judge:

    {score, rationale, entry, oversized, forbidden, diff_stats, error}

    Mechanical guards run first; the LLM only refines within what they allow.
    Never raises — an unusable judge shows up as ``error`` so a broken judge
    endpoint can't take down a mechanical eval run.
    """
    jcfg = task.get("judge") or {}
    task_type = jcfg.get("type", DEFAULT_TASK_TYPE)
    max_lines = int(jcfg.get("max_diff_lines", DEFAULT_MAX_DIFF_LINES))
    forbid = jcfg.get("forbid_paths") or []

    stats = diff_stats(diff)
    result: dict = {
        "score": None, "rationale": "", "entry": "",
        "oversized": stats["total_lines"] > max_lines,
        "forbidden": check_forbidden(diff, forbid) if forbid else [],
        "diff_stats": stats,
        "error": "",
    }

    if result["forbidden"]:
        result["score"] = 0
        result["rationale"] = (
            "touched forbidden path(s): " + ", ".join(result["forbidden"]))
        return result
    if not diff.strip():
        result["score"] = 0
        result["rationale"] = "empty diff"
        return result

    try:
        from agent.core.llm_retry import call_role_with_failover
        from agent.security.airgap import is_enabled as _airgap_enabled

        messages = build_judge_messages(task_type, task.get("prompt", ""), diff)
        resp, name, _entry = await call_role_with_failover(
            config, "judge",
            messages=messages,
            max_tokens=300,
            temperature=0.0,
            metrics_role="judge",
            local_only=_airgap_enabled(config),
        )
        text = resp.choices[0].message.content or ""
        score, rationale = parse_score(text)
        result["entry"] = name
        result["score"] = score
        result["rationale"] = rationale
    except Exception as exc:  # noqa: BLE001 - judging must never kill the run
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    if result["oversized"] and result["score"] > OVERSIZED_SCORE_CAP:
        result["score"] = OVERSIZED_SCORE_CAP
        result["rationale"] = (
            f"[clamped: diff {stats['total_lines']} lines > "
            f"{max_lines} cap] " + result["rationale"])
    return result


def load_agent_config():
    """Load the agent's config for judge calls. Separate function so tests
    can monkeypatch it and offline runs fail with a clear message."""
    from agent.config.loader import load_config
    return load_config()


# ── Baseline regression compare ───────────────────────────────────────────

JUDGE_REGRESSION_DROP = 2  # judged score drop strictly greater than this


def compare_to_baseline(current: dict, baseline: dict) -> list[str]:
    """Regressions of *current* vs *baseline* (both ``--json`` payloads).

    A regression is: mechanical pass -> fail flip, or a judged score drop of
    more than JUDGE_REGRESSION_DROP points. New tasks and improvements are
    not regressions. Returns human-readable regression strings (empty = ok).
    """
    def _index(payload: dict) -> dict[str, dict]:
        return {r.get("id"): r for r in payload.get("results", [])}

    cur, base = _index(current), _index(baseline)
    regressions: list[str] = []
    for tid, b in base.items():
        c = cur.get(tid)
        if c is None:
            continue  # task removed: not a regression signal
        if b.get("status") == "pass" and c.get("status") != "pass":
            regressions.append(f"{tid}: mechanical pass -> {c.get('status')}")
        b_j = (b.get("judge") or {}).get("score")
        c_j = (c.get("judge") or {}).get("score")
        if b_j is not None and c_j is not None and (b_j - c_j) > JUDGE_REGRESSION_DROP:
            regressions.append(f"{tid}: judged score {b_j} -> {c_j}")
    return regressions
