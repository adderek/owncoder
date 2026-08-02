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

from agent.security import vault

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
    data = vault.read_json(summary_path(session_dir, scope))
    return data if isinstance(data, dict) else {}


def _save(session_dir: Path, scope: str, content: str, up_to_turn: int) -> None:
    # A summary is the conversation in condensed form — same privacy gate.
    vault.write_json(summary_path(session_dir, scope), {
        "content": content,
        "summarized_up_to_turn": up_to_turn,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


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
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
    from agent.core.streaming import _clean_output
    from agent.core.llm_retry import open_stream_with_failover

    # Best-effort status label: the primary "background" entry, even though
    # failover may end up serving the call from a different one.
    try:
        from agent.config import make_registry
        _label_model = make_registry(config).background.model
    except Exception:
        _label_model = None
    _ms_inc("sum", None, _label_model)
    parts: list[str] = []
    client = None
    try:
        stream, _name, _entry, client = await open_stream_with_failover(
            config, "background",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content[:_MAX_INPUT_CHARS]},
            ],
            max_tokens=_MAX_OUTPUT_TOKENS, temperature=0.3,
            metrics_role="qa-summary",
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                parts.append(delta.content)
    finally:
        _ms_dec("sum", None, _label_model)
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass

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
