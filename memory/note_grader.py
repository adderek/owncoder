"""Grade injected notes for usefulness — feeds the notes relevance loop.

After a turn that had saved notes injected into context, this module asks a
cheap background model which of those notes the assistant's answer actually
relied on. Results land in the MemoryStore usage counters (inject_count /
used_count); Agent._refresh_notes_context demotes notes that keep getting
injected without ever being useful.

Runs from the idle deferred-action queue (agent.core.idle_tasks), fail-soft.
When no model is reachable, falls back to a token-overlap heuristic so the
counters still move.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You judge which background notes were actually useful for answering a "
    "user's message. You get the user message, the injected notes (each with "
    "an id), and the assistant's answer.\n"
    "Return ONLY a JSON array of the ids of notes whose content the answer "
    "visibly relied on. Unused or merely-related notes are excluded. An empty "
    "array is a normal, common result. Output JSON only."
)

_MAX_OUTPUT_TOKENS = 120
_JSON_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _overlap_heuristic(notes: list[dict], answer: str) -> list[str]:
    """Fallback: a note counts as used when enough of its distinctive tokens
    appear in the answer."""
    ans_tokens = {t.lower() for t in re.findall(r"\w{4,}", answer or "")}
    if not ans_tokens:
        return []
    used = []
    for n in notes:
        text = f"{n.get('title') or ''} {n.get('body') or ''}"
        toks = {t.lower() for t in re.findall(r"\w{4,}", text)}
        if not toks:
            continue
        if len(toks & ans_tokens) / len(toks) >= 0.3:
            used.append(n["id"])
    return used


async def grade_notes(
    query: str, notes: list[dict], answer: str, config: "Config"
) -> list[str]:
    """Return ids of *notes* the *answer* actually used. Never raises."""
    notes = [n for n in (notes or []) if n.get("id")]
    if not notes or not (answer or "").strip():
        return []
    try:
        from openai import AsyncOpenAI
        from agent.config import make_registry

        entry = make_registry(config).role("background")
        client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
        try:
            from agent.metrics import model_calls
            model_calls.record_entry(entry, role="note-grader")
        except Exception:
            pass
        note_lines = "\n".join(
            f"- id={n['id']} | {n.get('title') or ''}: {(n.get('body') or '')[:300]}"
            for n in notes
        )
        user = (
            f"User message:\n{query[:1500]}\n\nInjected notes:\n{note_lines}\n\n"
            f"Assistant answer:\n{(answer or '')[:3000]}"
        )
        try:
            resp = await client.chat.completions.create(
                model=entry.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
        finally:
            await client.close()
        m = _JSON_RE.search(raw)
        if not m:
            return _overlap_heuristic(notes, answer)
        ids = json.loads(m.group(0))
        valid = {n["id"] for n in notes}
        return [str(i) for i in ids if str(i) in valid]
    except Exception as e:
        logger.debug("note grading model call failed (%s) — using heuristic", e)
        return _overlap_heuristic(notes, answer)
