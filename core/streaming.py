from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import prompt_cache
from .prompts import apply_prompt_hints, _log_llm_request, _build_call_kwargs
from .tool_calls import _FakeToolCall, _parse_text_tool_calls, _parse_qwen_function_xml, _parse_agent_exec_xml
from .tool_discovery import CORE_TOOLS as _CORE_TOOLS

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


class StreamStalledError(Exception):
    """Raised when an LLM stream produces no chunk within the stall window.

    Signals a wedged backend (e.g. a GPU/HSA lost-wakeup on the llama.cpp side)
    so the turn loop can abort the request — closing the connection frees the
    server slot — and retry instead of hanging forever."""


@dataclass
class StreamedMessage:
    """OpenAI-message-shaped wrapper around a streamed response."""
    content: str | None
    tool_calls: list | None
    reasoning_content: str = ""


@dataclass
class StreamedChoice:
    """OpenAI-choice-shaped wrapper so streaming and non-streaming paths converge."""
    message: StreamedMessage
    finish_reason: str | None


def build_streamed_choice(finish_reason, full_content, raw_tool_calls, reasoning) -> StreamedChoice:
    return StreamedChoice(
        message=StreamedMessage(
            content=full_content,
            tool_calls=raw_tool_calls or None,
            reasoning_content=reasoning or "",
        ),
        finish_reason=finish_reason,
    )

# Repetition guard: break stream if last N content/reasoning chunks are identical
_REPEAT_WINDOW = 10
_REPEAT_THRESHOLD = 6
# Degenerate single-char runs (e.g. a URL followed by thousands of '0's) have no
# whitespace, so the word-based repetition guard never fires; 120 stays above
# legitimate horizontal rules / separator lines.
_CHAR_RUN_LIMIT = 120

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

# One source for both narration patterns below. Two hand-maintained copies of
# this list had drifted: `explore`, `find_symbol`, `find_tools` and `grep_code`
# are in CORE_TOOLS and were in neither, so a fabricated <grep_code> reached the
# user as a real search result with no nudge. CORE_TOOLS is imported so adding a
# core tool covers it automatically; _EXTRA_NARRATABLE holds the on-demand tools
# that are not core. `tool_discovery` imports nothing but __future__, so this is
# import-safe.
_EXTRA_NARRATABLE: frozenset[str] = frozenset({
    "patch_file", "search_archive", "web_fetch", "web_search",
    "git_diff", "git_log", "git_status", "git_blame", "git_related_files",
    "replace_symbol", "undo_file",
})

#: Every tool name a model might narrate instead of calling. Longest-first so the
#: alternation cannot match a prefix of a longer name (e.g. `git_log` before
#: `git_logs` would truncate; ordering removes the class of bug entirely).
_NARRATABLE_NAMES: tuple[str, ...] = tuple(
    sorted(_CORE_TOOLS | _EXTRA_NARRATABLE, key=lambda n: (-len(n), n))
)
_NARRATABLE_ALT = "|".join(_NARRATABLE_NAMES)

# Models that output bare Python-style function calls as text instead of invoking the tool API.
# Matched against the stripped start of the response (first non-whitespace line).
_BARE_TOOL_CALL_RE = re.compile(
    r"^(" + _NARRATABLE_ALT + r")\s*\(",
    re.MULTILINE,
)

# Tool names written as XML tags in prose, e.g.
#   <web_search query="...">...fabricated results...</web_search>
#   <write_file path="..." content="...">
# The parser never produced a call from these (it knows <agent_exec>,
# <function=name> and JSON forms), so the tag — and any result the model wrote
# inside it — is pure narration. Seen on heavily quantised Qwen3.5 (IQ2_S).
_PSEUDO_TOOL_TAG_RE = re.compile(
    r"<(" + _NARRATABLE_ALT + r")(?=[\s>])"
)



# ── prose vs code, shared by every narration check ──────────────────────────
# All three checks below used `text.split("```")` and looked at even segments.
# That knows exactly one code syntax, so four things read as prose and produced
# spurious nudges: inline `backticks`, ~~~ fences, indented blocks, and the model
# quoting a tag while explaining it. The last one matters most, because the nudge
# text itself names <agent_exec> and <tool_name ...>: once such a line is in
# history the detector can feed itself. The deadloop corpus has
# `<agent_exec tool="grep_code" ...>` repeated 138 times in one message.
_CODE_REGION_RE = re.compile(
    r"```.*?(?:```|\Z)"      # fenced block; tolerate an unterminated one
    r"|~~~.*?(?:~~~|\Z)"
    r"|`[^`\n]*`",            # inline code, single line only
    re.DOTALL,
)
_INDENTED_LINE_RE = re.compile(r"^(?:[ ]{4,}|\t)", re.MULTILINE)


def _split_prose_code(text: str) -> list[tuple[bool, str]]:
    """Alternating (is_prose, segment). Concatenating the segments rebuilds
    `text` exactly, so a caller may rewrite prose and leave code verbatim."""
    out: list[tuple[bool, str]] = []
    pos = 0
    for m in _CODE_REGION_RE.finditer(text):
        if m.start() > pos:
            out.extend(_split_indented(text[pos:m.start()]))
        out.append((False, m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.extend(_split_indented(text[pos:]))
    return out


def _split_indented(chunk: str) -> list[tuple[bool, str]]:
    """Indented lines are code; keep line endings with their line."""
    out: list[tuple[bool, str]] = []
    for line in chunk.splitlines(keepends=True):
        is_code = bool(_INDENTED_LINE_RE.match(line))
        if out and out[-1][0] == (not is_code):
            out[-1] = (not is_code, out[-1][1] + line)
        else:
            out.append((not is_code, line))
    return out


def _prose_only(text: str) -> str:
    return "".join(s for is_prose, s in _split_prose_code(text) if is_prose)


# A quotation is not a call. A narrated call carries an attribute (`name="v"`) or
# closes immediately; `<grep_code ...>` with a literal ellipsis is someone talking
# *about* the tag, which is exactly what the nudge text does.
_PSEUDO_TOOL_CALL_RE = re.compile(
    r"<(" + _NARRATABLE_ALT + r")(?:\s*>|\s+(?!\.\.\.)[a-zA-Z_][\w.-]*\s*=)"
)

def _has_pseudo_tool_tag(text: str) -> bool:
    """True if a tool name is written as a tag in prose, outside any code."""
    return bool(_PSEUDO_TOOL_CALL_RE.search(_prose_only(text)))


# Role labels that leak as residue right after a control token (e.g. "<|im_start|>thought").
# Only stripped when adjacent to a control token — a bare occurrence in prose is real text.
_ROLE_ALT = r"thought|user|assistant|system|tool"
# Strip thinking-mode special tokens leaked into content by some models (Gemma 4, DeepSeek, etc.),
# plus an optional role label trailing the token (the actual leak signal).
_LEAK_RE = re.compile(r"<[^>]*\|[^>]*>(?:\s*(?i:" + _ROLE_ALT + r")\b)?")
# ChatML-style tokens like <|im_start|>, <|im_end|>, <|imend>, <|imendend>
_CHATML_TOKEN_RE = re.compile(r"<\|[^>]*>(?:\s*(?i:" + _ROLE_ALT + r")\b)?")
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# A deadloop repeats a whole LINE, not a token, so the token-run check below is
# blind to it. Measured against the real loops in the agent logs: "I'll read from
# 370 to 550." x177 -> False, `self.last_request_time = time.time()` x64 -> False.
# Both are caught by counting repeated lines in a bounded tail instead.
#
# Bounds matter: this runs on the whole accumulated content for every streamed
# token, so it is already O(n^2) in the response. The line scan therefore looks at
# a fixed tail and a fixed number of lines, adding a constant, not another factor.
_LINE_TAIL_CHARS = 4000      # how far back to look
_LINE_WINDOW = 40            # qualifying lines considered
_LINE_REPEAT_THRESHOLD = 6   # identical lines in that window to call it a loop
_LINE_MIN_LEN = 20           # shorter lines legitimately repeat in code: }, pass, return


def _line_repetition_guard(content: str) -> bool:
    """True when one substantial line dominates the tail — the deadloop shape.

    Counts rather than requiring adjacency: real loops interleave. The log case
    alternated "I'll read from 370 to 550." with "Wait, I'll also check ...", which
    a run-length check would miss. The length floor keeps `}`, `pass` and `return
    None` from tripping it, since those repeat legitimately in generated code.
    """
    lines = [ln.strip() for ln in content[-_LINE_TAIL_CHARS:].splitlines()]
    lines = [ln for ln in lines if len(ln) >= _LINE_MIN_LEN][-_LINE_WINDOW:]
    if len(lines) < _LINE_REPEAT_THRESHOLD:
        return False
    from collections import Counter
    return Counter(lines).most_common(1)[0][1] >= _LINE_REPEAT_THRESHOLD


def _repetition_guard(content: str, threshold: int = _REPEAT_THRESHOLD) -> bool:
    """Check if last third of text repeats same word/phrase (model stuck in loop).

    Splits tail into word tokens, counts runs of identical tokens.
    Returns True when same token appears >threshold times consecutively.
    """
    tail = content[-_CHAR_RUN_LIMIT:]
    if len(tail) >= _CHAR_RUN_LIMIT and len(set(tail)) == 1:
        return True
    if _line_repetition_guard(content):
        return True
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
    parts = []
    last_end = 0
    for m in re.finditer(r"call:\w+\s*\{", text):
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
    return re.sub(r"<function=\w+>.*?</function>", "", text, flags=re.DOTALL).strip()


def _strip_agent_exec_xml(text: str) -> str:
    """Strip <agent_exec .../> and <agent_exec ...>...</agent_exec> blocks.

    Also handles malformed tags where args attribute is unclosed (no terminating ">)
    and tags whose attribute values contain > (e.g. Python -> syntax).
    """
    # Attribute content: handles quoted values that may contain >
    _AV = r'(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^>\'"])*'
    text = re.sub(r'<agent_exec\s+' + _AV + r'\s*/>', "", text, flags=re.DOTALL)
    text = re.sub(r'<agent_exec\s+' + _AV + r'\s*>.*?</agent_exec>', "", text, flags=re.DOTALL)
    # Malformed: args=" is never closed — strip from tag start to end of string
    text = re.sub(r'<agent_exec\s+tool="\w+"\s+args="(?:[^"\\]|\\.)*$', "", text, flags=re.DOTALL)
    return text.strip()


# No literal "<agent_exec" in the note — the marking regexes would re-match it.
_UNEXECUTED_EXEC_NOTE = "[removed: agent_exec tag written as text — this tool was NOT executed]"


def _mark_unexecuted_agent_exec(text: str) -> str:
    """Replace leftover <agent_exec> tags with an explicit not-executed marker.

    Tags still present at the end of a turn were rejected by the parser
    (unparseable tool name like "run_argv (extracted)", malformed args) and so
    never executed. Leaving them verbatim shows the model's fabricated result
    to the user as if real, and re-seeds the format for future imitation.
    """
    return _mark_unexecuted_tool_tags(text)


def _mark_unexecuted_tool_tags(text: str) -> str:
    """Mark every unexecuted tool tag: <agent_exec> and bare tool-name tags
    (<write_file ...>, <web_search ...>) alike. Neither form ever ran, so any
    result written inside one is fabricated."""
    _AV = r'(?:"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^>\'"])*'
    note = _UNEXECUTED_EXEC_NOTE

    def _sub(seg: str, tag: str) -> str:
        seg = re.sub(r'<' + tag + r'\b' + _AV + r'\s*/>', note, seg, flags=re.DOTALL)
        seg = re.sub(r'<' + tag + r'\b' + _AV + r'\s*>.*?</' + tag + r'>', note, seg, flags=re.DOTALL)
        # Malformed tail: tag opened but never closed before end of segment
        seg = re.sub(r'<' + tag + r'\b' + _AV + r'\s*>?(?:(?!</' + tag + r'>).)*$', note, seg, flags=re.DOTALL)
        return seg

    # Code stays verbatim — a tag quoted there is content, not a call.
    out = []
    for is_prose, seg in _split_prose_code(text):
        if is_prose:
            if "<agent_exec" in seg:
                seg = _sub(seg, "agent_exec")
            for m in set(_PSEUDO_TOOL_CALL_RE.findall(seg)):
                seg = _sub(seg, m)
        out.append(seg)
    return "".join(out).strip()


def _clean_output(text: str) -> str:
    """Strip leaked control tokens and thinking artifacts from model output."""
    text = _THINK_TAG_RE.sub("", text)
    text = _LEAK_RE.sub("", text)
    text = _CHATML_TOKEN_RE.sub("", text)
    # Strip role words glued to capitalized content (e.g. "thoughtAdd login").
    # No \s* and a case-sensitive [A-Z] lookahead: only the no-space concatenation
    # is stripped. A global re.IGNORECASE made [A-Z] match lowercase too, so
    # "systemu" -> "system"+"u" matched and got mangled to " u" mid-word.
    text = re.sub(
        r"\b(?i:" + _ROLE_ALT + r")(?=[A-Z])",
        " ", text,
    )
    # Strip orphaned role word at end (remaining after token stripping)
    text = re.sub(r"\s*\b(?i:" + _ROLE_ALT + r")\s*$", "", text)
    return text.strip()


def _has_unexecuted_agent_exec(text: str) -> bool:
    """True if an <agent_exec> tag appears outside ``` code fences.

    A tag surviving to this point was rejected by the parser (unparseable tool
    name, malformed args) and never executed. Fenced occurrences are excluded:
    quoting the tag in a code block (e.g. when working on this codebase) is
    legitimate content, not a hallucinated call.
    """
    return "<agent_exec" in _prose_only(text)


def _is_narrating_tool_use(text: str) -> bool:
    # A leftover <agent_exec> tag means the parser rejected it (bad tool name /
    # malformed args) — the model wrote a tool call as text, often with a
    # fabricated result. Treat as narration so the turn loop re-prompts.
    if _has_unexecuted_agent_exec(text):
        return True
    # Same failure, different syntax: <write_file ...>, <web_search ...> etc.
    if _has_pseudo_tool_tag(text):
        return True
    lower = text.lower()
    if any(phrase in lower for phrase in _NARRATION_PHRASES):
        return True
    # Bare function-call syntax: model outputs `write_file(...)` as text instead of API call.
    # Seen on Qwen2.5-14B-Instruct-1M K_M quants — correct task understanding, wrong output format.
    return bool(_BARE_TOOL_CALL_RE.search(text))


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


class _StreamStopped(Exception):
    """Internal: stop_event was set while waiting on a quiet stream."""


async def _safe_close(stream) -> None:
    try:
        await stream.close()
    except Exception:
        pass


async def _next_chunk(stream_it, *, budget_s: int, heartbeat_s: int, waiting_for: str,
                      stop_event=None, on_heartbeat=None):
    """Await the next stream chunk while keeping the wait observable and interruptible.

    Unlike a flat ``wait_for``, this distinguishes a slow-but-alive backend from a
    wedged one: it polls in ``heartbeat_s`` slices, emits a heartbeat on each slice
    (so a long prefill never looks frozen), honours ``stop_event`` mid-wait, and only
    declares a stall (``TimeoutError``) once the whole ``budget_s`` elapses with no chunk.
    ``budget_s``/``heartbeat_s`` of 0 disable the limit / the heartbeat respectively.
    """
    pending = asyncio.ensure_future(stream_it.__anext__())

    def _abandon() -> None:
        # Cancel and retrieve any late exception, otherwise asyncio logs
        # "exception was never retrieved" for a task that failed after we
        # gave up on it.
        pending.cancel()
        pending.add_done_callback(
            lambda t: t.cancelled() or t.exception())

    waited = 0.0
    while True:
        if budget_s <= 0 and heartbeat_s <= 0:
            return await pending
        slice_s = heartbeat_s if heartbeat_s > 0 else budget_s
        if budget_s > 0:
            slice_s = min(slice_s, max(0.01, budget_s - waited))
        # asyncio.wait (not wait_for+shield): a timeout leaves `pending`
        # untouched, so a chunk or error landing between slices is picked up
        # on the next pass instead of tripping asyncio's "exception in
        # shielded future" handler.
        done, _ = await asyncio.wait({pending}, timeout=slice_s)
        if done:
            return pending.result()
        waited += slice_s
        if stop_event is not None and stop_event.is_set():
            _abandon()
            raise _StreamStopped
        if budget_s > 0 and waited >= budget_s:
            _abandon()
            raise asyncio.TimeoutError
        if on_heartbeat is not None:
            try:
                # The budget travels with the heartbeat so the consumer can say
                # "40s of 260s" instead of a bare elapsed count that means
                # nothing without knowing what this model normally needs.
                on_heartbeat(waiting_for, int(waited), int(budget_s))
            except Exception:
                logger.exception("on_heartbeat callback failed")


async def _stream_response(client, config: "Config", api_messages, tools, on_token, on_usage=None, on_reasoning=None, stop_event=None, on_stall_progress=None):
    from agent._tokens import count_tokens_approx
    from agent.memory.compactor import _count_tokens_approx
    from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec, provider_label

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_arg_chars = 0
    tc_acc: dict[int, dict] = {}
    finish_reason = "stop"

    api_messages = apply_prompt_hints(api_messages, config)
    _log_llm_request(api_messages, tools, config)
    # Last step before the wire: hint injection rewrites the system message, so
    # breakpoints have to be placed after it or they mark stale content.
    api_messages = prompt_cache.prepare(api_messages, config)
    t_start = time.monotonic()
    t_first_token: float | None = None
    server_usage: dict | None = None

    _endpoint = provider_label(str(getattr(client, "base_url", "") or getattr(config.llm, "base_url", "")))
    _model = getattr(config.llm, "model", None)
    _ms_inc("main", _endpoint, _model)
    try:
        async with _gpu_slot(config):
            stream = await client.chat.completions.create(
            messages=api_messages,
            tools=tools if tools else None,
            stream=True,
            stream_options={"include_usage": True},
            **_build_call_kwargs(config),
        )

        # Two distinct fuses: prefill (no first token yet) emits no chunks while the
        # backend chews a big prompt — that is slow, not wedged, so it gets a generous
        # TTFT budget. Once tokens flow, a gap means the backend actually stalled, so
        # the tighter inter-chunk budget applies. Conflating them false-trips on long
        # prefills, and the retry then doubles backend load.
        ttft_s = int(getattr(config.llm, "stream_ttft_seconds", 0) or 0)
        stall_s = int(getattr(config.llm, "stream_stall_seconds", 0) or 0)
        heartbeat_s = int(getattr(config.llm, "stream_heartbeat_seconds", 0) or 0)
        # Prefill budget from this model's own measured history, scaled to the
        # prompt actually being sent: the fixed setting has to cover a cold box
        # chewing 200k tokens, which makes it far too loose for a normal turn.
        # Only tightens/loosens the FIRST-token fuse — once tokens flow, a gap
        # is a wedge regardless of how big the prompt was.
        if ttft_s > 0 and getattr(config.llm, "stream_ttft_adaptive", True):
            try:
                from agent.metrics.ttft_expect import expect_for_config
                exp = expect_for_config(config, _count_tokens_approx(api_messages))
                if exp.adaptive:
                    logger.debug("ttft budget %.0fs (was %ds) from %d samples",
                                 exp.budget_s, ttft_s, exp.samples)
                    ttft_s = int(exp.budget_s)
            except Exception:
                logger.debug("adaptive ttft budget unavailable", exc_info=True)
        stream_it = stream.__aiter__()
        while True:
            # Cooperative interrupt: a hung `async for` never yields back to the
            # turn loop, so check the stop flag on every chunk boundary.
            if stop_event is not None and stop_event.is_set():
                logger.info("stream: stop_event set — closing stream")
                finish_reason = "stop"
                await _safe_close(stream)
                break
            before_first = t_first_token is None
            budget_s = ttft_s if before_first else stall_s
            waiting_for = "first token (prefill)" if before_first else "next token"
            try:
                chunk = await _next_chunk(
                    stream_it,
                    budget_s=budget_s,
                    heartbeat_s=heartbeat_s,
                    waiting_for=waiting_for,
                    stop_event=stop_event,
                    on_heartbeat=on_stall_progress,
                )
            except StopAsyncIteration:
                break
            except _StreamStopped:
                logger.info("stream: stop_event set mid-wait — closing stream")
                finish_reason = "stop"
                await _safe_close(stream)
                break
            except asyncio.TimeoutError:
                await _safe_close(stream)
                why = (
                    "no first token — prefill exceeded budget or backend wedged"
                    if before_first else
                    "stopped emitting mid-stream — backend likely wedged"
                )
                raise StreamStalledError(
                    f"LLM stream stalled: no chunk for {budget_s}s ({why})"
                )
            u = getattr(chunk, "usage", None)
            if u is not None:
                server_usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                    # Part of prompt_tokens that the endpoint served from cache.
                    # 0 means "not reported", which is not the same as a miss.
                    "cached_tokens": prompt_cache.extract_cached_tokens(u),
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
        _ms_dec("main", _endpoint, _model)

    raw_content = "".join(content_parts)
    full_content = _clean_output(raw_content)
    if tc_acc:
        tool_calls = []
        for idx in sorted(tc_acc):
            raw = tc_acc[idx]
            raw_args = raw["function"]["arguments"] or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # Don't silently drop malformed args to {}. Try the flat-arg
                # recovery parser first; if that fails too, preserve the raw
                # string verbatim so execute_tool reports it as invalid JSON
                # and failure_report captures raw_arguments (matches the
                # non-streaming path, which keeps the raw string).
                from .tool_calls import _try_parse_flat_args
                args = _try_parse_flat_args(raw_args)
            fc = _FakeToolCall(raw["function"]["name"], args or {})
            if args is None:
                fc.function.arguments = raw_args
            fc.id = raw["id"] or fc.id
            tool_calls.append(fc)
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
                # Fallback: <agent_exec tool="name" args="key='val', ..."/> format
                text_calls = _parse_agent_exec_xml(raw_content)
                if text_calls:
                    tool_calls = [_FakeToolCall(c["name"], c["arguments"]) for c in text_calls]
                    full_content = _strip_agent_exec_xml(full_content)
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
            "cached_input_tokens": (server_usage or {}).get("cached_tokens", 0),
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
