from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from .prompts import _inject_think_hint, _log_llm_request, _build_call_kwargs
from .tool_calls import _FakeToolCall

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_NARRATION_PHRASES = [
    "i'll apply", "i will apply", "let me apply",
    "i'll patch", "i will patch", "let me patch",
    "i'll write", "i will write", "let me write",
    "i'll modify", "i will modify", "let me modify",
    "i'll update", "i will update", "let me update",
    "i'll change", "i will change", "let me change",
    "i'll create", "i will create", "let me create",
    "i need to write", "i need to create", "i need to modify", "i need to patch",
    "i should write", "i should create", "i should modify", "i should patch",
    "using patch_file", "using write_file",
]


def _is_narrating_tool_use(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _NARRATION_PHRASES)


def _strip_tool_blocks(text: str) -> str:
    from .tool_calls import _TAG_RE
    text = _TAG_RE.sub("", text)
    return text.strip()


async def _stream_response(client, config: "Config", api_messages, tools, on_token, on_usage=None, on_reasoning=None):
    from agent._tokens import count_tokens_approx
    from agent.memory.compactor import _count_tokens_approx
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_arg_chars = 0
    tc_acc: dict[int, dict] = {}
    finish_reason = "stop"

    api_messages = _inject_think_hint(api_messages, config)
    _log_llm_request(api_messages, tools, config)
    t_start = time.monotonic()
    t_first_token: float | None = None
    server_usage: dict | None = None

    _ms_inc("main")
    try:
        stream = await client.chat.completions.create(
            messages=api_messages,
            tools=tools if tools else None,
            stream=True,
            stream_options={"include_usage": True},
            **_build_call_kwargs(config),
        )

        async for chunk in stream:
            u = getattr(chunk, "usage", None)
            if u is not None:
                server_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }

            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = choice.delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                reasoning_parts.append(rc)
                if on_reasoning is not None:
                    try:
                        on_reasoning(rc)
                    except Exception:
                        logger.exception("on_reasoning callback failed")
            if delta.content:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                content_parts.append(delta.content)
                on_token(delta.content)
            for tc_delta in (delta.tool_calls or []):
                if t_first_token is None:
                    t_first_token = time.monotonic()
                if tc_delta.function and tc_delta.function.arguments:
                    tool_arg_chars += len(tc_delta.function.arguments)
                idx = tc_delta.index
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if tc_delta.id:
                    tc_acc[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc_acc[idx]["function"]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc_acc[idx]["function"]["arguments"] += tc_delta.function.arguments
    finally:
        _ms_dec("main")

    full_content = "".join(content_parts)
    if tc_acc:
        tool_calls = []
        for idx in sorted(tc_acc):
            raw = tc_acc[idx]
            try:
                args = json.loads(raw["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(_FakeToolCall(raw["function"]["name"], args))
            tool_calls[-1].id = raw["id"] or tool_calls[-1].id
    else:
        tool_calls = None

    t_end = time.monotonic()
    full_reasoning = "".join(reasoning_parts)
    if on_usage is not None:
        content_tokens = count_tokens_approx(full_content) if full_content else 0
        reasoning_tokens = count_tokens_approx(full_reasoning) if full_reasoning else 0
        tool_tokens = tool_arg_chars // 4
        output_tokens = (
            server_usage["completion_tokens"]
            if server_usage and server_usage["completion_tokens"]
            else content_tokens + reasoning_tokens + tool_tokens
        )
        input_tokens = (
            server_usage["prompt_tokens"]
            if server_usage and server_usage["prompt_tokens"]
            else _count_tokens_approx(api_messages)
        )
        stream_seconds = max(1e-6, t_end - t_start)
        gen_seconds = max(1e-6, t_end - (t_first_token or t_start))
        on_usage({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "content_tokens": content_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_tokens": tool_tokens,
            "stream_seconds": stream_seconds,
            "gen_seconds": gen_seconds,
            "ttft": (t_first_token - t_start) if t_first_token else None,
        })

    return finish_reason, full_content, tool_calls, full_reasoning
