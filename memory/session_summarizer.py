"""Session-level Q/A dialogue summarizer.

Generates a persistent Markdown summary of all Q or A turns from a session.
Stored in session_dir/session_{q|a}_summary.json with a turn watermark.
Re-summarization is incremental: previous summary + new turns only.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_Q_SYSTEM = (
    "Summarize the user questions from this coding-assistant session.\n"
    "Produce a concise Markdown document covering: what topics were explored, "
    "the user's goals and intent, any recurring themes or unresolved issues.\n"
    "Output Markdown only."
)

_A_SYSTEM = (
    "Summarize the agent responses from this coding-assistant session.\n"
    "Produce a concise Markdown document covering: what was accomplished, "
    "which tools were used, files modified, key decisions and findings.\n"
    "Output Markdown only."
)

_Q_UPDATE_SYSTEM = (
    _Q_SYSTEM + "\nYou receive a previous summary and additional new turns. "
    "Integrate the new turns into the summary, keeping the result concise."
)

_A_UPDATE_SYSTEM = (
    _A_SYSTEM + "\nYou receive a previous summary and additional new turns. "
    "Integrate the new turns into the summary, keeping the result concise."
)

_MAX_INPUT_CHARS = 12000
_MAX_OUTPUT_TOKENS = 900


def summary_path(session_dir: Path, scope: str) -> Path:
    return session_dir / f"session_{scope}_summary.json"


def load_stored(session_dir: Path, scope: str) -> dict:
    p = summary_path(session_dir, scope)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(session_dir: Path, scope: str, content: str, up_to_turn: int) -> None:
    p = summary_path(session_dir, scope)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "content": content,
                "summarized_up_to_turn": up_to_turn,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _fmt_q(entries: list) -> str:
    lines = []
    for i, (tid, q, _a) in enumerate(entries, 1):
        text = (q or {}).get("content", "")
        if text:
            lines.append(f"{i}. [turn {tid}] {text[:800]}")
    return "\n".join(lines)


def _fmt_a(entries: list) -> str:
    lines = []
    for i, (tid, _q, a) in enumerate(entries, 1):
        if not a:
            continue
        text = (a.get("content") or "")[:800]
        tools = a.get("tool_calls") or []
        files = a.get("modified_files") or []
        meta_parts = []
        if tools:
            meta_parts.append("tools: " + ", ".join(str(x) for x in tools[:6]))
        if files:
            meta_parts.append("files: " + ", ".join(files[:4]))
        meta = f" [{', '.join(meta_parts)}]" if meta_parts else ""
        if text or meta:
            lines.append(f"{i}. [turn {tid}]{meta}\n   {text}")
    return "\n".join(lines)


async def _call_llm(config: "Config", system: str, user_content: str) -> str:
    from openai import AsyncOpenAI
    from agent.config import make_registry
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
    from agent.core.streaming import _clean_output

    entry = make_registry(config).background
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    _ms_inc("sum")
    parts: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=entry.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content[:_MAX_INPUT_CHARS]},
            ],
            max_tokens=_MAX_OUTPUT_TOKENS,
            temperature=0.3,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                parts.append(delta.content)
    finally:
        _ms_dec("sum")
        await client.close()

    return _clean_output("".join(parts)).strip()


async def generate(
    session_dir: Path,
    entries: list,
    scope: str,
    config: "Config",
    *,
    force: bool = False,
) -> str:
    """Return a session-level Markdown summary for scope 'q' or 'a'.

    Loads from disk if current (i.e. watermark >= max turn). Generates and
    persists otherwise. Incremental: only new turns are sent when updating.
    """
    if not entries:
        return ""

    max_turn = max(tid for tid, _q, _a in entries)
    stored = load_stored(session_dir, scope)
    watermark = stored.get("summarized_up_to_turn", -1)

    if not force and watermark >= max_turn and stored.get("content"):
        return stored["content"]

    fmt = _fmt_q if scope == "q" else _fmt_a
    sys_fresh = _Q_SYSTEM if scope == "q" else _A_SYSTEM
    sys_update = _Q_UPDATE_SYSTEM if scope == "q" else _A_UPDATE_SYSTEM

    prev = stored.get("content", "")
    if prev and watermark >= 0:
        # Incremental: pass previous summary + only the new turns
        new_entries = [(tid, q, a) for tid, q, a in entries if tid > watermark]
        if not new_entries and not force:
            return prev
        new_text = fmt(new_entries)
        user_content = (
            f"Previous summary (covers turns up to {watermark}):\n{prev}\n\n"
            f"New turns to incorporate:\n{new_text}"
        )
        system = sys_update
    else:
        user_content = fmt(entries)
        system = sys_fresh

    if not user_content.strip():
        return prev

    result = await _call_llm(config, system, user_content)
    if result:
        await asyncio.to_thread(_save, session_dir, scope, result, max_turn)
    return result
