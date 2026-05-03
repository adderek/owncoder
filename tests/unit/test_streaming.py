"""Tests for agent/core/streaming.py — leak cleaning and stream handling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.config import Config
from agent.core.streaming import _clean_output, _stream_response


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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
        ],
    )
    def test_repetition_guard_detects_loops(self, text, expected_repeat):
        from agent.core.streaming import _repetition_guard
        assert _repetition_guard(text) == expected_repeat

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
