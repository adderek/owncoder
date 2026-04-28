"""LLM-based compaction for old completed plan steps.

Summarizes completed/skipped steps into plan.notes, then removes them to
reduce token cost when the plan is injected into agent context.
Call compact_plan() after a batch of steps finish, or on demand via /plan compact.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config
    from agent.planning.plan import Plan

logger = logging.getLogger(__name__)

_COMPACT_PROMPT = (
    "You are summarizing completed steps from an agent's work plan.\n"
    "Given a list of completed/skipped steps with their notes, produce a concise\n"
    "summary (≤5 sentences) capturing: what was accomplished, key decisions made,\n"
    "and any caveats for future steps. Be factual. Output plain text only."
)


async def compact_plan(
    plan: "Plan",
    client: "AsyncOpenAI",
    config: "Config",
    *,
    min_completed: int = 3,
) -> tuple[bool, str]:
    """Summarize and drop completed/skipped steps from plan.

    Returns (changed, message). No-op if fewer than min_completed steps qualify.
    Saves plan after compaction.
    """
    from agent.planning.plan import save_plan

    done_steps = [s for s in plan.steps if s.status in ("completed", "skipped")]
    if len(done_steps) < min_completed:
        return False, f"Only {len(done_steps)} completed steps — threshold {min_completed} not reached."

    step_lines = "\n".join(
        f"- [{s.id}] {s.description}" + (f" (notes: {s.notes})" if s.notes else "")
        for s in done_steps
    )
    user_msg = f"Goal: {plan.goal}\n\nCompleted steps:\n{step_lines}"

    model = getattr(config.llm, "model", "")
    max_tokens = getattr(config.token_limits, "commit_summary_tokens", 512)

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _COMPACT_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        summary = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("compact_plan: LLM call failed: %s", exc)
        return False, f"LLM call failed: {exc}"

    summary = summary.strip()
    if not summary:
        return False, "LLM returned empty summary."

    done_ids = {s.id for s in done_steps}
    plan.steps = [s for s in plan.steps if s.id not in done_ids]

    prior = plan.notes.strip()
    plan.notes = (prior + "\n\n" + summary).strip() if prior else summary

    save_plan(plan)
    return True, f"Compacted {len(done_steps)} steps. Summary appended to plan.notes."


def compact_plan_sync(
    plan: "Plan",
    *,
    min_completed: int = 3,
) -> tuple[bool, str]:
    """Drop completed/skipped steps without LLM summary (no client needed).

    Concatenates step descriptions + notes into plan.notes as plain text.
    Useful when no LLM client is available (e.g. CLI tooling, tests).
    """
    from agent.planning.plan import save_plan

    done_steps = [s for s in plan.steps if s.status in ("completed", "skipped")]
    if len(done_steps) < min_completed:
        return False, f"Only {len(done_steps)} completed steps — threshold {min_completed} not reached."

    lines = [f"Completed steps (compacted):"]
    for s in done_steps:
        line = f"  [{s.id}] {s.description}"
        if s.notes:
            line += f" — {s.notes}"
        lines.append(line)
    summary = "\n".join(lines)

    done_ids = {s.id for s in done_steps}
    plan.steps = [s for s in plan.steps if s.id not in done_ids]

    prior = plan.notes.strip()
    plan.notes = (prior + "\n\n" + summary).strip() if prior else summary

    save_plan(plan)
    return True, f"Compacted {len(done_steps)} steps (no LLM)."
