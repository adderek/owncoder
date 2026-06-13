"""Session-end skill distillation — procedural memory that self-evolves.

Complements promoter.py (project facts) and reflector.py (behavioral rules).
Those persist *what* the agent learned; this persists *how* — reusable
multi-step workflows saved as loadable skills under .agent/skills/.

The LLM is shown the current skill index so it can either:
  - create a brand-new skill for a workflow it just executed, or
  - refine an existing one (same name → SkillLoader.save bumps the version),

which is the "skills get better over time" loop. Saving is deliberately
conservative: capped per session, gated by confidence, and skipped when the
proposed body is effectively identical to the current skill.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.memory.facts_store import FactsStore

logger = logging.getLogger(__name__)

_MAX_SKILLS_PER_SESSION = 2
_MIN_CONFIDENCE = 0.8

_DISTILL_PROMPT = """\
You distill reusable SKILLS from a coding session for an AI coding agent.

A skill is a durable, multi-step procedure the agent can reload in a future
session to repeat a workflow without rediscovering it. NOT a one-line rule,
NOT a project fact — a concrete how-to (e.g. "run the integration suite",
"cut a release", "add a new tool to the registry").

You are given the session knowledge and the list of EXISTING skills.
- If the session demonstrates a new reusable workflow → propose a new skill.
- If it improves/corrects an EXISTING skill → reuse that skill's exact name so
  it gets refined (versioned), and provide the full improved body.

Each skill body: Markdown, imperative steps, self-contained, generic (no
task-specific paths/values). Skip anything trivial or one-off.

Output JSON only:
{"skills": [{"name": "kebab-name", "description": "one line", "content": "## Steps\\n1. ...", "confidence": 0.9}]}

Max 2 skills. confidence 0.0-1.0, propose only >= 0.8. Output {"skills": []} if
nothing durable and procedural was learned.
"""


def distill_session_skills(
    session_id: str,
    config: "Config",
    facts_store: "FactsStore | None" = None,
) -> int:
    """Distill reusable skills from a session; save to project skills dir.

    Returns number of skills created/updated (0 on skip or error).
    """
    if not getattr(config.agent, "distill_skills", True):
        return 0
    if facts_store is None:
        return 0

    latest = facts_store.latest_round()
    source_text = ""
    if latest is not None:
        source_text = latest.knowledge_draft or latest.summary or ""
    if len(source_text) < 80:
        return 0

    try:
        from agent.skills import SkillLoader
        loader = SkillLoader(config)
    except Exception as e:
        logger.debug("distill_session_skills: loader init failed: %s", e)
        return 0

    existing = loader.available()
    index_text = "\n".join(f"- {n}: {d}" for n, d in existing) or "(none yet)"
    prompt_input = (
        f"[EXISTING SKILLS]\n{index_text}\n\n[SESSION KNOWLEDGE]\n{source_text[:5000]}"
    )

    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": _DISTILL_PROMPT},
                {"role": "user", "content": prompt_input},
            ],
            max_tokens=900,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("distill_session_skills: LLM call failed: %s", e)
        return 0

    skills_list = _parse_skills(raw)
    if not skills_list:
        return 0

    existing_names = {n for n, _ in existing}
    saved = 0
    for item in skills_list[:_MAX_SKILLS_PER_SESSION]:
        name = (item.get("name") or "").strip()
        content = (item.get("content") or "").strip()
        description = (item.get("description") or "").strip()
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if not name or not content or confidence < _MIN_CONFIDENCE:
            continue

        # Skip churn: if the skill already exists with an effectively identical
        # body, don't bump the version for nothing.
        from agent.skills import normalize_name
        slug = normalize_name(name)
        if slug in existing_names:
            try:
                cur_body = loader.load([slug])
                if _normalize_body(cur_body).endswith(_normalize_body(content)):
                    continue
            except Exception:
                pass

        try:
            meta = loader.save(
                name,
                content,
                description=description or None,
                origin=f"distill:{session_id[:8]}",
            )
            saved += 1
            logger.debug(
                "distill_session_skills: %s skill %s v%d",
                "updated" if meta["version"] > 1 else "created",
                meta["name"],
                meta["version"],
            )
        except Exception as e:
            logger.debug("distill_session_skills: save failed for %r: %s", name, e)

    if saved:
        logger.info(
            "distill_session_skills: %d skill(s) saved for session %s",
            saved,
            session_id[:16],
        )
    return saved


def _normalize_body(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _parse_skills(raw: str) -> list[dict]:
    """Parse LLM JSON output, tolerating markdown fences and bare arrays."""

    def _extract(data) -> list[dict]:
        if isinstance(data, dict):
            return data.get("skills") or []
        if isinstance(data, list):
            return [s for s in data if isinstance(s, dict)]
        return []

    try:
        return _extract(json.loads(raw))
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        try:
            return _extract(json.loads(m.group(1)))
        except Exception:
            pass

    return []
