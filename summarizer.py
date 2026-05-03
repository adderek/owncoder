from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_Q_SYSTEM = (
    "Summarise the following user message in one concise sentence that captures the intent. "
    "Output only the summary sentence — no labels, no punctuation other than a period."
)
_A_SYSTEM = (
    "Summarise the following agent response in one concise sentence that captures the outcome. "
    "Output only the summary sentence — no labels, no punctuation other than a period."
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Strip Gemma 4 / thinking-mode special tokens leaked into output
_LEAK_RE = re.compile(r"<[^>]*\|[^>]*>")


async def _call_llm_one_line(
    config: "Config",
    system_prompt: str,
    content: str,
) -> str:
    """Stream a one-line summary using the summarizer model (falls back to default LLM)."""
    from openai import AsyncOpenAI
    from agent.config import make_registry
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
    entry = make_registry(config).summarizer
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    _ms_inc("sum")
    try:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        stream = await client.chat.completions.create(
            model=entry.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content[:4000]},
            ],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta:
                if delta.content:
                    content_parts.append(delta.content)
                if getattr(delta, "reasoning_content", None):
                    reasoning_parts.append(delta.reasoning_content)
    finally:
        _ms_dec("sum")
        await client.close()

    def _clean(text: str) -> str:
        """Strip Gemma 4 thinking-mode prefix artifacts from streamed output."""
        text = _THINK_RE.sub("", text)
        text = _LEAK_RE.sub("", text)
        text = re.sub(
            r"^(?:\s*(?:<[^>]*\|[^>]*>|\b(?:thought|user|assistant|system|tool)\b)\s*)+",
            "", text, flags=re.IGNORECASE,
        )
        return text.strip()

    raw_content = "".join(content_parts)
    full = _clean(raw_content)
    if not full:
        full = _clean("".join(reasoning_parts))
    if not full:
        full = raw_content.strip()
    return full


def _update_json_file(path: Path, updates: dict) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(updates)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("summarizer: failed to update %s: %s", path, e)


async def summarize_turn_background(
    config: "Config",
    q_path: Path,
    a_path: Path,
) -> None:
    """Background task: add summary_q / summary_a to the Q and A JSON files.

    Runs in isolation — never touches the agent's primary conversation context.
    Errors are silently logged and never propagate to the caller.
    """
    if not config.ui.q_summaries:
        return
    try:
        q_content = json.loads(q_path.read_text(encoding="utf-8")).get("content", "")
        a_content = json.loads(a_path.read_text(encoding="utf-8")).get("content", "")

        if not q_content and not a_content:
            return

        summary_q, summary_a = await asyncio.gather(
            _call_llm_one_line(config, _Q_SYSTEM, q_content) if q_content else asyncio.sleep(0, result=""),
            _call_llm_one_line(config, _A_SYSTEM, a_content) if a_content else asyncio.sleep(0, result=""),
        )

        if summary_q:
            await asyncio.to_thread(_update_json_file, q_path, {"summary_q": summary_q})
        if summary_a:
            await asyncio.to_thread(_update_json_file, a_path, {"summary_a": summary_a})

        logger.debug(
            "summarize_turn_background: done — Q=%r A=%r",
            summary_q[:60] if summary_q else "",
            summary_a[:60] if summary_a else "",
        )
    except asyncio.CancelledError:
        logger.debug("summarize_turn_background: cancelled")
        raise
    except Exception:
        logger.exception("summarize_turn_background: unexpected error (ignored)")
