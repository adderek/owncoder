"""Tests for agent/core/streaming.py — leak cleaning and stream handling."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.config import Config
from agent.core.streaming import _clean_output, _stream_response, StreamStalledError


class TestCleanOutput:
    """_clean_output strips leaked control tokens and thinking artifacts."""

    def test_channel_tokens_stripped(self):
        """<|channel|> and <|im_start|> removed, real content preserved."""
        dirty = "<|channel|><|im_start|>thought <channel|>add login button"
        assert _clean_output(dirty) == "add login button"

    def test_think_blocks_stripped(self):
        """<think>...</think> entirely removed."""
        dirty = "Here's the fix.<think>I need to check edge cases</think> Added validation."
        assert _clean_output(dirty) == "Here's the fix. Added validation."

    def test_tool_call_fragment_preserved(self):
        """call:func{...} preserved — parsed as text-based tool call."""
        dirty = "Looking at code.<|tool_call|>call:search_code{term: 'x'}"
        assert _clean_output(dirty) == "Looking at code.call:search_code{term: 'x'}"

    def test_orphaned_role_word_at_end_stripped(self):
        """Standalone 'thought' at end stripped after token cleanup."""
        dirty = "<|channel|>thought"
        assert _clean_output(dirty) == ""

    def test_mixed_real_content_preserved(self):
        """Real content survives — only tokens stripped."""
        dirty = "Add login button.\n<|channel|>Extra noise."
        assert _clean_output(dirty) == "Add login button.\nExtra noise."

    def test_clean_input_unchanged(self):
        """Normal text passes through unmodified."""
        assert _clean_output("Fix bug in auth middleware.") == "Fix bug in auth middleware."

    def test_empty_input(self):
        assert _clean_output("") == ""

    def test_role_words_mid_text_handled(self):
        """'thought' before uppercase word gets space, not removed."""
        dirty = "thoughtLet me fix this"
        assert _clean_output(dirty) == "Let me fix this"

    def test_role_word_prefix_of_real_word_not_mangled(self):
        """Words that merely start with a role word + lowercase are left intact.

        Regression: global re.IGNORECASE made the [A-Z] lookahead match lowercase
        too, so 'systemu' -> 'system'+'u' was mangled to ' u' mid-word.
        """
        assert _clean_output("Architektura systemu jest") == "Architektura systemu jest"
        assert _clean_output("toolkit username userland") == "toolkit username userland"
        assert _clean_output("system design now") == "system design now"

    def test_call_fragment_mid_text_not_stripped(self):
        """call:pattern mid-text is NOT stripped — only trailing fragments."""
        text = "You can call:search() with a query."
        assert _clean_output(text) == text

    def test_chatml_tokens_stripped(self):
        """<|imend>, <|imendend> stripped from output."""
        assert _clean_output("Done.<|imend>") == "Done."
        assert _clean_output("Done.<|imendend>") == "Done."
        assert _clean_output("Done.<|im_end|>") == "Done."


class TestStreamResponseClean:
    """_stream_response returns cleaned full_content from leaky chunks."""

    def _mock_chunk(self, content: str = "", reasoning: str = "",
                    tool_calls: list | None = None, finish: str | None = None):
        """Build a mock stream chunk."""
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.finish_reason = finish
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        if tool_calls:
            delta.tool_calls = tool_calls
        else:
            delta.tool_calls = []
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    async def _make_async_iter(self, *chunks):
        """Turn chunks into async generator."""
        for c in chunks:
            yield c

    def _make_client(self, *chunks):
        """Return a client whose create() yields the given chunks."""
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=self._make_async_iter(*chunks)
        )
        return client

    def _make_config(self):
        """Minimal config for streaming."""
        c = Config()
        c.llm.think_level = "off"
        return c

    async def test_channel_tokens_cleaned_from_stream(self):
        """Leaky stream chunks have channel tokens stripped, text tool calls parsed."""
        client = self._make_client(
            self._mock_chunk(content="Add login"),
            self._mock_chunk(content=" button.\n"),
            self._mock_chunk(content="<|channel|><|im_start|>thought"),
            self._mock_chunk(content="<|tool_call|>call:search{}"),
        )
        tokens: list[str] = []
        _, content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: tokens.append(t),
        )
        # Content still has thought+tool call text (no longer stripped)
        assert "Add login button" in content
        # Text-based tool call detected
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function.name == "search"

    async def test_think_blocks_cleaned_from_stream(self):
        """<think> blocks spread across chunks are stripped."""
        client = self._make_client(
            self._mock_chunk(content="Result.\n"),
            self._mock_chunk(content="<think>Deep analysis here"),
            self._mock_chunk(content=" more thinking</think>"),
            self._mock_chunk(content=" Done."),
        )
        _, content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert content == "Result.\n Done."

    async def test_clean_stream_unchanged(self):
        """Stream without leaks returns full content as-is."""
        client = self._make_client(
            self._mock_chunk(content="Fix "),
            self._mock_chunk(content="the "),
            self._mock_chunk(content="bug."),
        )
        _, content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert content == "Fix the bug."

    @pytest.mark.parametrize(
        "text,expected_repeat",
        [
            ("de-facto de-facto de-facto de-facto de-facto de-facto de-facto", True),
            ("de-facto " * 10, True),
            ("a b c d e f g h", False),
            ("de-facto de-facto something-else de-facto de-facto", False),
            ("repeat repeat repeat repeat repeat repeat repeat", True),
            # Degenerate single-char run with no whitespace (session ec19:
            # URL followed by thousands of '0's formed one giant "word")
            ("https://www.polsatnews.pl/wiadomosc/202607081625" + "0" * 8000, True),
            ("0" * 120, True),
            ("0" * 119, False),
            ("normal text then " + "=" * 80 + " separator", False),
        ],
    )
    def test_repetition_guard_detects_loops(self, text, expected_repeat):
        from agent.core.streaming import _repetition_guard
        assert _repetition_guard(text) == expected_repeat

    def _tc_delta(self, index: int, name: str = "", arguments: str = "", tc_id: str = ""):
        """Build a mock streamed tool-call delta fragment."""
        tc = MagicMock()
        tc.index = index
        tc.id = tc_id
        fn = MagicMock()
        fn.name = name
        fn.arguments = arguments
        tc.function = fn
        return tc

    async def test_malformed_tool_args_preserved_not_dropped(self):
        """Unparseable streamed tool-call args keep the raw string instead of {}."""
        bad = '{this is not valid json at all}'  # fails json.loads and flat-arg parse
        client = self._make_client(
            self._mock_chunk(tool_calls=[self._tc_delta(0, name="edit_file", tc_id="call_1")]),
            self._mock_chunk(tool_calls=[self._tc_delta(0, arguments=bad)]),
        )
        _, _content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert calls is not None and len(calls) == 1
        # Raw malformed string preserved verbatim — not silently emptied to "{}".
        assert calls[0].function.arguments == bad
        assert calls[0].function.arguments != "{}"

    async def test_valid_tool_args_parsed(self):
        """Well-formed streamed tool-call args round-trip to canonical JSON."""
        client = self._make_client(
            self._mock_chunk(tool_calls=[self._tc_delta(0, name="read_file", tc_id="call_2")]),
            self._mock_chunk(tool_calls=[self._tc_delta(0, arguments='{"path": "a.py"}')]),
        )
        import json as _json
        _, _content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert calls is not None and len(calls) == 1
        assert calls[0].function.name == "read_file"
        assert _json.loads(calls[0].function.arguments) == {"path": "a.py"}

    async def test_stream_breaks_on_repeated_content(self):
        """Stream breaks when same word repeats many times."""
        chunks = [self._mock_chunk(content="de-facto ")] * 15
        client = self._make_client(*chunks)
        _, content, calls, _ = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert "de-facto" in content
        # Should have fewer than all 15 (broken early)
        assert content.count("de-facto") < 15

    async def test_repeated_reasoning_breaks_stream(self):
        """Stream breaks when same reasoning word repeats many times."""
        chunks = [self._mock_chunk(reasoning="de-facto ")] * 15
        client = self._make_client(*chunks)
        _, content, calls, reasoning = await _stream_response(
            client, self._make_config(), [], [], on_token=lambda t: None,
        )
        assert "de-facto" in reasoning
        assert reasoning.count("de-facto") < 15  # should not have all 15
        tokens: list[str] = []
        client = self._make_client(
            self._mock_chunk(content="hello "),
            self._mock_chunk(content="<|channel|>noise"),
        )
        await _stream_response(client, self._make_config(), [], [], on_token=tokens.append)
        # on_token sees raw chunks; final content is cleaned
        assert tokens == ["hello ", "<|channel|>noise"]


class _HangingStream:
    """Async stream that yields one chunk then blocks forever — simulates a
    wedged backend (GPU/HSA lost-wakeup) mid-generation."""

    def __init__(self, first_chunk):
        self._first = first_chunk
        self._served = False
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._served:
            self._served = True
            return self._first
        await asyncio.Event().wait()  # never set → hang

    async def close(self):
        self.closed = True


class TestStreamStallWatchdog:
    """The per-chunk stall watchdog aborts a wedged stream instead of hanging."""

    def _config(self, stall_s):
        c = Config()
        c.llm.think_level = "off"
        c.llm.stream_stall_seconds = stall_s
        # The watchdog now has two fuses (pre-first-token TTFT vs mid-stream gap);
        # the hanging-stream tests trip before any real token, so cap TTFT too.
        c.llm.stream_ttft_seconds = stall_s
        return c

    async def test_stall_raises_after_timeout(self):
        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = []
        stream = _HangingStream(chunk)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream)

        cfg = self._config(stall_s=1)  # 1s window — fires fast, bounded test time
        with pytest.raises(StreamStalledError):
            await _stream_response(client, cfg, [], [], on_token=lambda t: None)
        assert stream.closed is True  # stream closed → server slot freed

    async def test_no_watchdog_when_disabled(self):
        # stall_seconds=0 disables the watchdog; a normal finite stream still works
        c = self._config(stall_s=0)
        chunk = self._finished_chunk()
        client = MagicMock()

        async def _iter():
            yield chunk
        client.chat.completions.create = AsyncMock(return_value=_iter())
        finish, content, calls, _ = await _stream_response(
            client, c, [], [], on_token=lambda t: None,
        )
        assert finish == "stop"

    def _finished_chunk(self):
        chunk = MagicMock()
        chunk.usage = None
        choice = MagicMock()
        choice.finish_reason = "stop"
        delta = MagicMock()
        delta.content = "ok"
        delta.reasoning_content = ""
        delta.tool_calls = []
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    async def test_ttft_fuse_independent_of_inter_chunk(self):
        # Before the first token, the generous TTFT budget governs — NOT the tight
        # inter-chunk stall. A long prefill must not false-trip the mid-stream fuse.
        c = Config()
        c.llm.think_level = "off"
        c.llm.stream_stall_seconds = 100   # huge: would never fire in test time
        c.llm.stream_ttft_seconds = 1      # the fuse that should actually trip
        c.llm.stream_heartbeat_seconds = 0
        chunk = MagicMock(); chunk.usage = None; chunk.choices = []
        stream = _HangingStream(chunk)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream)
        with pytest.raises(StreamStalledError):
            await _stream_response(client, c, [], [], on_token=lambda t: None)
        assert stream.closed is True

    async def test_heartbeat_fires_while_waiting(self):
        c = Config()
        c.llm.think_level = "off"
        c.llm.stream_stall_seconds = 100
        c.llm.stream_ttft_seconds = 3
        c.llm.stream_heartbeat_seconds = 1  # ticks before the 3s budget trips
        chunk = MagicMock(); chunk.usage = None; chunk.choices = []
        stream = _HangingStream(chunk)
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=stream)
        beats: list[tuple[str, int, int]] = []
        with pytest.raises(StreamStalledError):
            await _stream_response(
                client, c, [], [], on_token=lambda t: None,
                on_stall_progress=lambda waiting_for, secs, budget: beats.append(
                    (waiting_for, secs, budget)),
            )
        assert beats, "expected at least one heartbeat while the stream was quiet"
        assert "first token" in beats[0][0]
        # The budget travels with the beat so the UI can show "Ns of Ms".
        assert beats[0][2] == 3


# ── narration-pattern coverage ──────────────────────────────────────────────
# Two hand-maintained copies of the tool-name list had drifted out of sync with
# the registry: explore, find_symbol, find_tools and grep_code were in
# CORE_TOOLS and in neither pattern, so a fabricated <grep_code> reached the user
# as a genuine search result and fired no nudge. The patterns are now derived
# from CORE_TOOLS; these tests fail if a future core tool escapes them.

def test_every_core_tool_is_caught_as_a_pseudo_tag():
    from agent.core.streaming import _PSEUDO_TOOL_TAG_RE
    from agent.core.tool_discovery import CORE_TOOLS
    missed = sorted(n for n in CORE_TOOLS
                    if not _PSEUDO_TOOL_TAG_RE.search(f'<{n} arg="x">'))
    assert not missed, f"tool names narratable as XML but not detected: {missed}"


def test_every_core_tool_is_caught_as_a_bare_call():
    from agent.core.streaming import _BARE_TOOL_CALL_RE
    from agent.core.tool_discovery import CORE_TOOLS
    missed = sorted(n for n in CORE_TOOLS
                    if not _BARE_TOOL_CALL_RE.search(f'{n}("x")'))
    assert not missed, f"tool names narratable as bare calls but not detected: {missed}"


def test_fabricated_search_result_is_marked_not_passed_through():
    """The regression that motivated this: a search tool narrated with invented
    output must be replaced, not handed to the user as if the tool had run."""
    from agent.core.streaming import _mark_unexecuted_tool_tags, _has_pseudo_tool_tag
    text = 'Here is what I found:\n<grep_code pattern="login">src/auth.py:42: def login()</grep_code>\nDone.'
    assert _has_pseudo_tool_tag(text)
    out = _mark_unexecuted_tool_tags(text)
    assert "src/auth.py:42" not in out, "fabricated tool output survived"


def test_tags_inside_code_fences_are_left_alone():
    from agent.core.streaming import _has_pseudo_tool_tag
    assert not _has_pseudo_tool_tag('Example:\n```\n<grep_code pattern="x">\n```\n')


# ── prose vs code: the false-nudge classes ──────────────────────────────────
# All three narration checks used to split on "```" alone, so four things read
# as prose and produced spurious nudges. The fourth matters most: the nudge text
# itself names <agent_exec> and <tool_name ...>, so once such a line reaches
# history the detector can feed itself. The deadloop corpus in the agent logs has
# `<agent_exec tool="grep_code" ...>` repeated 138 times in a single message.

import pytest


@pytest.mark.parametrize("label,text", [
    ("inline backticks",  'Use `<grep_code pattern="x">` to search.'),
    ("tilde fence",       'Example:\n~~~\n<grep_code pattern="x">\n~~~\n'),
    ("indented block",    'Example:\n\n    <grep_code pattern="x">\n'),
    ("quoted in prose",   'Do not write <grep_code ...> as text; call the tool.'),
    ("backtick fence",    '```\n<grep_code pattern="x">\n```'),
])
def test_code_and_quotation_are_not_narration(label, text):
    from agent.core.streaming import _has_pseudo_tool_tag
    assert not _has_pseudo_tool_tag(text), f"false positive on {label}"


@pytest.mark.parametrize("text", [
    '<grep_code pattern="x">src/a.py:1: hit</grep_code>',
    '<grep_code>hit</grep_code>',
    '<write_file path="/tmp/x" content="y">',
])
def test_real_narration_is_still_caught(text):
    from agent.core.streaming import _has_pseudo_tool_tag
    assert _has_pseudo_tool_tag(text)


def test_split_prose_code_rebuilds_the_input_exactly():
    """_mark_unexecuted_tool_tags rewrites prose and must leave code verbatim,
    so the segmentation has to be lossless."""
    from agent.core.streaming import _split_prose_code
    for t in ['a `b` c', '```\nx\n```', '~~~\ny\n~~~\n', 'p\n\n    indented\nq',
              'unterminated ```fence', '']:
        assert "".join(s for _, s in _split_prose_code(t)) == t


def test_fenced_tag_survives_the_rewrite():
    from agent.core.streaming import _mark_unexecuted_tool_tags
    text = 'Docs:\n```\n<write_file path="x" content="y">\n```\nDone.'
    assert '<write_file path="x" content="y">' in _mark_unexecuted_tool_tags(text)


# ── deadloop: a repeated LINE, not a repeated token ─────────────────────────
# _repetition_guard only ever detected runs of one identical token (`//////`,
# `the the the`), because it compares words[i] to words[i-1]. A deadloop repeats a
# whole line, whose adjacent tokens differ, so none of the real loops in the agent
# logs tripped it: "I'll read from 370 to 550." x177 -> False,
# `self.last_request_time = time.time()` x64 -> False.
#
# Replayed over 648 real assistant messages from the logs, the line check fires on
# 14 of 14 messages with >=6 repeated lines in the tail and on none of the other
# 634. Note an offline replay UNDERSTATES a streaming guard: it sees only the final
# text, while during streaming the loop passes through the tail.

_LOOP_LINE = "I'll read from 370 to 550."
_LOOP_CODE = "self.last_request_time = time.time()"


@pytest.mark.parametrize("label,text", [
    ("177x prose line, from the logs", "\n".join([_LOOP_LINE] * 177)),
    ("64x code line, from the logs",   "\n".join([_LOOP_CODE] * 64)),
    # The logged loop alternated two lines, so adjacency-based detection misses it.
    ("interleaved pair",  "\n".join([_LOOP_LINE, "Wait, I'll also check the definition."] * 8)),
    ("at the threshold",  "Analysis.\n" + "\n".join([_LOOP_LINE] * 6)),
])
def test_repeated_lines_are_caught(label, text):
    from agent.core.streaming import _repetition_guard
    assert _repetition_guard(text), label


@pytest.mark.parametrize("label,text", [
    ("just below the threshold", "Analysis.\n" + "\n".join([_LOOP_LINE] * 5)),
    # Short lines repeat legitimately in generated code; the length floor covers it.
    ("short code lines",  "\n".join(["    pass", "    return None", "    }"] * 20)),
    ("distinct env reads", "\n".join(f'    {k} = os.environ.get("{k}")' for k in
                                     "host port user password timeout retries region".split())),
    ("ordinary prose",    "A normal answer with several sentences. Each one differs. "
                          "Nothing repeats here at all."),
])
def test_legitimate_repetition_is_not_a_loop(label, text):
    from agent.core.streaming import _line_repetition_guard
    assert not _line_repetition_guard(text), label


def test_the_line_scan_is_bounded():
    """The guard runs on the whole accumulated content for every streamed token, so
    it is already O(n^2); the line scan must add a constant, not another factor."""
    from agent.core.streaming import (_LINE_TAIL_CHARS, _LINE_WINDOW,
                                      _line_repetition_guard)
    # A loop far outside the tail must not be found: proof the scan is bounded.
    buried = "\n".join([_LOOP_LINE] * 50) + "\n" + ("unique filler line number %d\n" % 0) \
             + "\n".join(f"unique filler line number {i}" for i in range(1, 400))
    assert len(buried) > _LINE_TAIL_CHARS
    assert not _line_repetition_guard(buried)
    assert _LINE_WINDOW > 0
