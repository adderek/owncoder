from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from .prompts import _inject_think_hint, _log_llm_request, _build_call_kwargs
from .tool_calls import _FakeToolCall, _parse_text_tool_calls, _parse_qwen_function_xml

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

# Repetition guard: break stream if last N content/reasoning chunks are identical
_REPEAT_WINDOW = 10
_REPEAT_THRESHOLD = 6

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

# Strip thinking-mode special tokens leaked into content by some models (Gemma 4, DeepSeek, etc.)
_LEAK_RE = re.compile(r"<[^>]*\|[^>]*>")
# ChatML-style tokens like <|im_start|>, <|im_end|>, <|imend>, <|imendend>
_CHATML_TOKEN_RE = re.compile(r"<\|[^>]*>")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _repetition_guard(content: str, threshold: int = _REPEAT_THRESHOLD) -> bool:
    """Check if last third of text repeats same word/phrase (model stuck in loop).

    Splits tail into word tokens, counts runs of identical tokens.
    Returns True when same token appears >threshold times consecutively.
    """
    words = content.split()
    if len(words) < threshold:
        return False
    # Check the trailing chunk for repeated token runs
    tail = words[-threshold:]
    if len(set(tail)) == 1:
        return True
    # Also check runs of the same token
    run_count = 1
    for i in range(len(words) - 1, 0, -1):
        if words[i] == words[i - 1]:
            run_count += 1
            if run_count >= threshold:
                return True
        else:
            run_count = 1
    return False


def _strip_text_tool_calls(text: str) -> str:
    """Strip call:function_name{...} fragments, keeping surrounding text.

    Uses the same bracket-matching logic as _parse_text_tool_calls so nested
    braces (rare in args) are handled correctly.
    """
    import re as _re
    parts = []
    last_end = 0
    for m in _re.finditer(r"call:\w+\s*\{", text):
        start = m.start()
        brace_start = text.index("{", start)
        depth = 1
        i = brace_start + 1
        while depth > 0 and i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        parts.append(text[last_end:start])
        last_end = i
    parts.append(text[last_end:])
    return "".join(parts).strip()


def _strip_qwen_function_xml(text: str) -> str:
    """Strip <function=name>...</function> blocks, keeping surrounding text."""
    import re as _re
    return _re.sub(r"<function=\w+>.*?</function>", "", text, flags=_re.DOTALL).strip()


def _clean_output(text: str) -> str:
    """Strip leaked control tokens and thinking artifacts from model output."""
    text = _THINK_TAG_RE.sub("", text)
    text = _LEAK_RE.sub("", text)
    text = _CHATML_TOKEN_RE.sub("", text)
    # Strip role words concatenated with actual content (e.g. "thoughtAdd login")
    text = re.sub(
        r"\b(?:thought|user|assistant|system|tool)\s*(?=[A-Z])",
        " ", text, flags=re.IGNORECASE,
    )
    # Strip orphaned role word at end (remaining after token stripping)
    text = re.sub(r"\s*\b(?:thought|user|assistant|system|tool)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _is_narrating_tool_use(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _NARRATION_PHRASES)


def _strip_tool_blocks(text: str) -> str:
    from .tool_calls import _TAG_RE
    text = _TAG_RE.sub("", text)
    return text.strip()


def _gpu_slot(config):
    """Context manager: acquire GPU semaphore only when the model is GPU-bound."""
    from agent.core.model_status import gpu_slot as _gs
    if config.llm.gpu:
        return _gs()
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def _noop():
        yield
    return _noop()


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
        async with _gpu_slot(config):
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
                if _repetition_guard("".join(reasoning_parts)):
                    logger.warning("Repeating reasoning content detected — breaking stream")
                    break
            if delta.content:
                if t_first_token is None:
                    t_first_token = time.monotonic()
                content_parts.append(delta.content)
                on_token(delta.content)
                if _repetition_guard("".join(content_parts)):
                    logger.warning("Repeating content detected — breaking stream")
                    break
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

    raw_content = "".join(content_parts)
    full_content = _clean_output(raw_content)
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
    elif raw_content:
        # No native function calls — try text-based tool calls from raw content
        text_calls = _parse_text_tool_calls(raw_content)
        if text_calls:
            tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in text_calls]
            full_content = _strip_text_tool_calls(full_content)
        else:
            # Fallback: Qwen3 <function=name>...<parameter>...</parameter> XML
            text_calls = _parse_qwen_function_xml(raw_content)
            if text_calls:
                tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in text_calls]
                full_content = _strip_qwen_function_xml(full_content)
            else:
                tool_calls = None
    else:
        tool_calls = None

    t_end = time.monotonic()
    full_reasoning = "".join(reasoning_parts)
    stream_seconds = max(1e-6, t_end - t_start)
    gen_seconds = max(1e-6, t_end - (t_first_token or t_start))
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

    n_tool_calls = len(tool_calls) if tool_calls else 0
    logger.debug(
        "_stream_response: finish=%r content=%dch reasoning=%dch tools=%d stream=%.1fs gen=%.1fs",
        finish_reason, len(full_content), len(full_reasoning),
        n_tool_calls, stream_seconds, gen_seconds,
    )

    return finish_reason, full_content, tool_calls, full_reasoning
