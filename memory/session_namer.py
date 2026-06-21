"""Auto-generate session metadata (name, description, tags, classification).

A session's id is a timestamp; this module gives it a human name and searchable
metadata. Runs on the summarizer model, fail-soft (never raises), and is meant
to be driven from the idle deferred-action queue (see agent.core.idle_tasks).
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.memory.session import Session

logger = logging.getLogger(__name__)

_CLASSES = ("feature", "bugfix", "refactor", "research", "docs", "ops", "other")

_SYSTEM = (
    "You name and classify a coding-assistant session from its transcript.\n"
    "Return ONLY a JSON object, no prose, with keys:\n"
    '  "name": 1-3 word human title (may use spaces, Title Case)\n'
    '  "description": one concise sentence describing the session\n'
    '  "tags": array of up to 5 short lowercase ascii tags\n'
    f'  "classification": one of {list(_CLASSES)}\n'
    '  "summary": 1-3 sentence summary of what happened\n'
    "Be specific and concrete. Output JSON only."
)

_MAX_INPUT_CHARS = 12000
_MAX_OUTPUT_TOKENS = 400
_MIN_MESSAGES = 2

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def needs_meta(session: "Session") -> bool:
    """True when the session lacks generated metadata worth filling in."""
    if session is None:
        return False
    name = (getattr(session, "name", "") or "").strip()
    desc = (getattr(session, "description", "") or "").strip()
    tags = getattr(session, "tags", None) or []
    classification = (getattr(session, "classification", "") or "").strip()
    return not (name and desc and tags and classification)


def _format_transcript(messages: list[dict]) -> str:
    """Flatten user/assistant turns into a compact transcript string."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.strip().startswith("{system}"):
            continue
        lines.append(f"{role}: {content.strip()[:1000]}")
    return "\n".join(lines)


async def _call_llm(config: "Config", user_content: str) -> str:
    from openai import AsyncOpenAI
    from agent.config import make_registry

    try:
        from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
    except Exception:  # pragma: no cover - fallback when status unavailable
        def _ms_inc(_role: str) -> None: ...
        def _ms_dec(_role: str) -> None: ...

    entry = make_registry(config).summarizer
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    _ms_inc("name")
    try:
        resp = await client.chat.completions.create(
            model=entry.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_content[:_MAX_INPUT_CHARS]},
            ],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    finally:
        _ms_dec("name")
        await client.close()


def _coerce_meta(raw: str) -> dict | None:
    """Parse the model's JSON reply into a clean metadata dict, or None."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    from agent.memory.session import _sanitize_short_name

    name = str(data.get("name", "") or "").strip()[:80]
    description = str(data.get("description", "") or "").strip()[:500]
    summary = str(data.get("summary", "") or "").strip()[:1000]

    classification = str(data.get("classification", "") or "").strip().lower()
    if classification not in _CLASSES:
        classification = "other"

    raw_tags = data.get("tags") or []
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for t in raw_tags:
            t = re.sub(r"[^a-z0-9_-]", "", str(t).strip().lower())[:24]
            if t and t not in tags:
                tags.append(t)
            if len(tags) >= 5:
                break

    if not name:
        return None

    short_name = _sanitize_short_name(name.replace(" ", "-").lower())

    return {
        "name": name,
        "short_name": short_name,
        "description": description,
        "tags": tags,
        "classification": classification,
        "summary": summary,
    }


async def generate_session_meta(
    session: "Session", messages: list[dict], config: "Config"
) -> dict | None:
    """Return generated metadata dict for *session*, or None on failure/skip.

    Never raises — logs and returns None on any error.
    """
    try:
        convo = [m for m in (messages or []) if m.get("role") in ("user", "assistant")]
        if len(convo) < _MIN_MESSAGES:
            return None
        transcript = _format_transcript(messages)
        if not transcript.strip():
            return None
        raw = await _call_llm(config, transcript)
        return _coerce_meta(raw)
    except Exception:
        logger.debug("generate_session_meta failed", exc_info=True)
        return None


def apply_meta(session: "Session", meta: dict, *, overwrite: bool = False) -> bool:
    """Apply generated *meta* onto *session*. Returns True if anything changed.

    By default only fills empty fields (respects a user-set name). With
    overwrite=True, replaces all fields.
    """
    if not meta:
        return False
    changed = False

    def _set(attr: str, value) -> None:
        nonlocal changed
        if value in (None, "", []):
            return
        current = getattr(session, attr, None)
        if overwrite or not current:
            if current != value:
                setattr(session, attr, value)
                changed = True

    _set("name", meta.get("name"))
    _set("short_name", meta.get("short_name"))
    _set("description", meta.get("description"))
    _set("tags", meta.get("tags"))
    _set("classification", meta.get("classification"))
    _set("summary", meta.get("summary"))
    return changed
