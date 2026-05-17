from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_Q_SYSTEM = (
    "Summarise the following user message to capture the intent. "
    "Length limits: keep text between tool calls to ≤25 words. Keep final responses to ≤100 words unless the task requires more detail. "
    "Output only the summary — no labels, no extra punctuation."
)
_A_SYSTEM = (
    "Summarise the following agent response to capture the outcome. "
    "Length limits: keep text between tool calls to ≤25 words. Keep final responses to ≤100 words unless the task requires more detail. "
    "Output only the summary — no labels, no extra punctuation."
)


def _pick_summarizer_entry(config: "Config", content: str) -> tuple:
    """Smart summarizer routing: GPU when free+fits, CPU fallback otherwise.

    Returns (entry, used_gpu) where entry is the chosen ModelEntry and
    used_gpu indicates whether it's the GPU pool model.
    """
    from agent.config import make_registry
    from agent.core.model_status import get_counts

    cpu_entry = make_registry(config).summarizer
    gpu_pool = config.concurrency.gpu_pool

    # If no GPU pool configured, always use CPU
    if not gpu_pool:
        return cpu_entry, False

    # Find the default model entry — it's the primary GPU model
    default_name = config.model_roles.get("default", "default")
    if default_name not in gpu_pool:
        return cpu_entry, False  # default is not GPU, nothing to route

    default_entry = config.model_entries.get(default_name)
    if default_entry is None:
        return cpu_entry, False

    # Check if GPU is currently busy (main role is running)
    counts = get_counts()
    if counts.get("main", 0) > 0:
        return cpu_entry, False  # GPU busy with main chat

    # Estimate whether the content fits in GPU context (leave 25% headroom)
    from agent.memory.compactor import _count_tokens_approx
    total_tokens = _count_tokens_approx([
        {"role": "system", "content": _Q_SYSTEM},
        {"role": "user", "content": content[:4000]},
    ])
    # Add some margin for the second summary if both run in parallel
    total_tokens = total_tokens * 2 + 200
    ctx = default_entry.ctx_window
    if ctx and total_tokens < ctx * 0.75:
        return default_entry, True  # fits on GPU

    return cpu_entry, False


from contextlib import asynccontextmanager


@asynccontextmanager
async def _noop():
    yield


async def _call_llm_one_line(
    config: "Config",
    system_prompt: str,
    content: str,
) -> str:
    """Stream a one-line summary using the summarizer model (falls back to default LLM)."""
    from openai import AsyncOpenAI
    from agent.config import make_registry
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec, gpu_slot as _gpu_slot
    entry, used_gpu = _pick_summarizer_entry(config, content)
    client = AsyncOpenAI(base_url=entry.base_url, api_key=entry.api_key)
    _ms_inc("sum" if not used_gpu else "main")
    try:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        async with _gpu_slot() if used_gpu else _noop():
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
        _ms_dec("sum" if not used_gpu else "main")
        await client.close()

    from agent.core.streaming import _clean_output

    raw_content = "".join(content_parts)
    full = _clean_output(raw_content)
    if not full:
        full = _clean_output("".join(reasoning_parts))
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
    except Exception as _exc:
        logger.warning("summarize_turn_background: error (ignored): %s", _exc)
