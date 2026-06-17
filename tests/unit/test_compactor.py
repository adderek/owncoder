"""Unit tests for agent/memory/compactor.py — token counting and parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agent.memory.compactor import (
    _count_tokens_approx,
    _parse_compaction_output,
    _parse_synthesis_output,
    _truncate_tool_results_in,
    compact,
    CompactionError,
)
from agent.memory.facts_store import FactsStore
from agent._test_helpers import make_response, make_client


class TestCountTokensApprox:
    def test_nonzero_for_content(self):
        msgs = [{"role": "user", "content": "hello world"}]
        assert _count_tokens_approx(msgs) > 0

    def test_empty_messages(self):
        assert _count_tokens_approx([]) == 0

    def test_tool_calls_counted(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"x.py"}',
                        },
                    }
                ],
            }
        ]
        assert _count_tokens_approx(msgs) > 0

    def test_none_content(self):
        msgs = [{"role": "assistant", "content": None}]
        count = _count_tokens_approx(msgs)
        assert count >= 0


class TestParseCompactionOutput:
    def test_full_output(self):
        text = (
            '<facts>{"files_modified": ["a.py"]}</facts><summary>Did stuff.</summary>'
        )
        facts, summary = _parse_compaction_output(text)
        assert facts["files_modified"] == ["a.py"]
        assert summary == "Did stuff."

    def test_missing_facts(self):
        text = "<summary>Just a summary.</summary>"
        facts, summary = _parse_compaction_output(text)
        assert facts == {}
        assert summary == "Just a summary."

    def test_invalid_json_in_facts(self):
        text = "<facts>not json</facts><summary>Ok.</summary>"
        facts, summary = _parse_compaction_output(text)
        assert facts == {}
        assert summary == "Ok."

    def test_no_tags_at_all(self):
        text = "Just plain text without any tags."
        facts, summary = _parse_compaction_output(text)
        assert facts == {}
        assert summary == ""


class TestTruncateToolResults:
    def test_short_messages_unchanged(self):
        msgs = [
            {"role": "tool", "content": "short", "tool_call_id": "1"},
            {"role": "user", "content": "hello"},
        ]
        result = _truncate_tool_results_in(msgs, max_chars=100)
        assert result[0]["content"] == "short"

    def test_long_tool_result_truncated(self):
        long_content = "x" * 5000
        msgs = [{"role": "tool", "content": long_content, "tool_call_id": "1"}]
        result = _truncate_tool_results_in(msgs, max_chars=100)
        assert len(result[0]["content"]) < 200
        assert "truncated" in result[0]["content"]

    def test_non_tool_messages_preserved(self):
        long_content = "x" * 5000
        msgs = [{"role": "user", "content": long_content}]
        result = _truncate_tool_results_in(msgs, max_chars=100)
        assert result[0]["content"] == long_content


class TestParseSynthesisOutput:
    def test_all_three_blocks(self):
        text = (
            '<facts>{"files_modified":["a.py"]}</facts>'
            "<summary>Did X.</summary>"
            "<q>User wants Y.</q>"
        )
        facts, summary, q = _parse_synthesis_output(text)
        assert facts == {"files_modified": ["a.py"]}
        assert summary == "Did X."
        assert q == "User wants Y."

    def test_missing_q_is_empty(self):
        text = "<facts>{}</facts><summary>s</summary>"
        _, _, q = _parse_synthesis_output(text)
        assert q == ""


def _mk_messages(n_pairs: int = 12, size_chars: int = 400) -> list[dict]:
    """Build a system message + `n_pairs` user/assistant exchanges."""
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        msgs.append({"role": "user", "content": f"question {i}: " + "x" * size_chars})
        msgs.append(
            {"role": "assistant", "content": f"answer {i}: " + "y" * size_chars}
        )
    return msgs


def _client_returning(*outputs: str):
    """Async client whose chat.completions.create yields the given outputs in order."""
    client = MagicMock()
    responses = [make_response(content=o) for o in outputs]
    client.chat.completions.create = AsyncMock(side_effect=responses)
    return client


class TestTwoStageCompact:
    @pytest.mark.asyncio
    async def test_single_round_persists_tier2(self, cfg, tmp_path):
        """First compaction writes round-0001 with a knowledge draft,
        and the compacted message carries the stage-2 summary."""
        store = FactsStore("sess-1", base_dir=tmp_path)

        stage1_draft = "### Intent\nUser wants foo.\n### Files\nedited foo.py."
        stage2 = (
            '<facts>{"files_modified":["foo.py"]}</facts>'
            "<summary>Edited foo.py to add foo().</summary>"
            "<q>User needs foo() working.</q>"
        )
        client = _client_returning(stage1_draft, stage2)

        messages = _mk_messages()
        result = await compact(
            messages, cfg, client, keep_last=2, facts_store=store, turn_index=12
        )

        # Two LLM calls made: stage 1 + stage 2.
        assert client.chat.completions.create.await_count == 2

        # Round persisted.
        latest = store.latest_round()
        assert latest is not None
        assert latest.round_id == 1
        assert latest.knowledge_draft == stage1_draft
        assert latest.summary == "Edited foo.py to add foo()."
        assert latest.q_view == "User needs foo() working."
        assert latest.facts == {"files_modified": ["foo.py"]}

        # Compacted message contains the stage-2 summary, NOT the draft.
        compacted = [m for m in result if m.get("role") == "assistant"][0]
        assert "Edited foo.py" in compacted["content"]
        assert "recall_facts" in compacted["content"]
        # Draft content should stay Tier-2 only — never in the active message.
        assert stage1_draft not in compacted["content"]

    @pytest.mark.asyncio
    async def test_incremental_round_feeds_previous_forward(self, cfg, tmp_path):
        """Second compaction round sees the prior draft + summary in the
        Stage-1 prompt, so facts accumulate instead of being re-derived (and gradually lost) each time."""
        store = FactsStore("sess-2", base_dir=tmp_path)

        # Seed a prior round manually.
        store.new_round(
            from_turn=0,
            to_turn=3,
            knowledge_draft="### Intent\nOld goal was bar().",
            summary="Earlier: added bar().",
            q_view="User wanted bar().",
            facts={"files_modified": ["bar.py"]},
        )

        stage1 = "### Intent\nOld goal was bar(). New addition: baz()."
        stage2 = (
            '<facts>{"files_modified":["bar.py","baz.py"]}</facts>'
            "<summary>Now both bar() and baz() exist.</summary>"
            "<q>Needs baz() tested.</q>"
        )
        client = _client_returning(stage1, stage2)

        messages = _mk_messages()
        await compact(
            messages, cfg, client, keep_last=2, facts_store=store, turn_index=15
        )

        # Inspect what the stage-1 call received — the prior draft must be in there.
        stage1_call = client.chat.completions.create.await_args_list[0]
        stage1_messages = stage1_call.kwargs["messages"]
        user_content = stage1_messages[-1]["content"]
        assert "Previous knowledge draft" in user_content
        assert "Old goal was bar()" in user_content
        assert "Earlier: added bar()" in user_content

        # New round links to previous.
        latest = store.latest_round()
        assert latest.round_id == 2
        assert latest.prev_round_id == 1
        assert latest.prev_summary == "Earlier: added bar()."
        assert latest.knowledge_draft == stage1
        assert "baz.py" in latest.facts.get("files_modified", [])

    @pytest.mark.asyncio
    async def test_compact_without_facts_store_still_works(self, cfg):
        """Back-compat: calls with no facts_store behave like before."""
        stage1 = "draft"
        stage2 = "<facts>{}</facts><summary>ok.</summary><q></q>"
        client = _client_returning(stage1, stage2)
        messages = _mk_messages()
        result = await compact(messages, cfg, client, keep_last=2)
        assert any("ok." in (m.get("content") or "") for m in result)


class TestCompactionRobustness:
    @pytest.mark.asyncio
    async def test_looks_complete(self):
        from agent.memory.compactor import _looks_complete

        assert _looks_complete("<facts>{}</facts><summary>s</summary><q>q</q>") is True
        assert _looks_complete("<facts>{}</facts><summary>s</summary>") is False
        assert _looks_complete("<summary>s</summary><q>q</q>") is False

    @pytest.mark.asyncio
    async def test_synthesize_summary_retry_on_length(self, cfg):
        from agent.memory.compactor import _synthesize_summary

        client = make_client(
            make_response(
                content="<facts>{}</facts><summary>s</summary><q>q</q>",
                finish_reason="length",
            ),
            make_response(content='<facts>{"a":1}</facts><summary>s</summary><q>q</q>'),
        )

        facts, summary, q = await _synthesize_summary("draft", cfg, client)
        assert facts == {"a": 1}
        assert summary == "s"
        assert q == "q"
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_synthesize_summary_retry_on_incomplete(self, cfg):
        from agent.memory.compactor import _synthesize_summary

        client = make_client(
            make_response(content="<facts>{}</facts><summary>s</summary>"),
            make_response(content='<facts>{"a":1}</facts><summary>s</summary><q>q</q>'),
        )

        facts, summary, q = await _synthesize_summary("draft", cfg, client)
        assert facts == {"a": 1}
        assert client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_synthesize_summary_fails_after_retry(self, cfg):
        from agent.memory.compactor import _synthesize_summary, CompactionError

        client = make_client(
            make_response(content="<facts>{}</facts><summary>s</summary>"),
            make_response(content="<facts>{}</facts><summary>s</summary>"),
        )

        with pytest.raises(
            CompactionError, match="stage 2 output incomplete after retry"
        ):
            await _synthesize_summary("draft", cfg, client)

    @pytest.mark.asyncio
    async def test_analyze_transcript_stage1_failure_returns_placeholder(self, cfg):
        """Stage 1 failure must return a safe placeholder, not the raw transcript."""
        from agent.memory.compactor import _analyze_transcript

        raw_transcript = "user: hello\nassistant: hi there"
        client = make_client()
        client.chat.completions.create = AsyncMock(side_effect=Exception("conn error"))

        # Use a long transcript so the short-transcript shortcut doesn't fire.
        long_transcript = raw_transcript + (" x" * 2000)
        result = await _analyze_transcript(long_transcript, None, cfg, client)

        assert "[COMPACTION_ERROR" in result
        assert long_transcript not in result

    @pytest.mark.asyncio
    async def test_analyze_transcript_stage1_failure_preserves_prev_knowledge(self, cfg):
        """Previous round's knowledge_draft is kept as prefix even when Stage 1 fails."""
        from agent.memory.compactor import _analyze_transcript
        from agent.memory.facts_store import FactsRound

        prev = FactsRound(
            round_id=1,
            timestamp="2026-01-01T00:00:00Z",
            from_turn=0,
            to_turn=5,
            knowledge_draft="Previous knowledge here.",
        )
        client = make_client()
        client.chat.completions.create = AsyncMock(side_effect=Exception("conn error"))

        long_transcript = "user: new turn\n" + ("x " * 2000)
        result = await _analyze_transcript(long_transcript, prev, cfg, client)

        assert result.startswith("Previous knowledge here.")
        assert "[COMPACTION_ERROR" in result
        assert long_transcript not in result

    @pytest.mark.asyncio
    async def test_compact_fallback_on_compaction_error(self, cfg, tmp_path):
        from agent.memory.compactor import compact, CompactionError
        from agent.memory.facts_store import FactsStore

        store = FactsStore("sess-fallback", base_dir=tmp_path)

        client = make_client()
        client.chat.completions.create = AsyncMock(side_effect=Exception("LLM error"))

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
            {"role": "user", "content": "third"},
        ]

        result = await compact(
            messages, cfg, client, keep_last=1, facts_store=store, turn_index=3
        )

        assert len(result) == len(messages)
        assert (
            result[0]["content"]
            == "[SESSION SUMMARY ERROR: stage 2 call failed: LLM error]"
        )

    @pytest.mark.asyncio
    async def test_verbatim_tail_does_not_start_with_orphan_tool(self, cfg, tmp_path):
        """If the verbatim window would begin on a tool result whose assistant
        (tool_calls) got compacted away, that result must be pushed into the
        compacted half — never left as a leading orphan tool message."""
        store = FactsStore("sess-orphan", base_dir=tmp_path)
        stage2 = "<facts>{}</facts><summary>s</summary><q>q</q>"
        client = _client_returning("draft", stage2)

        big = "x" * 400
        # Tail of 4 (keep_last=2) starts on a tool result; a user message later
        # in the tail keeps the last-user adjustment from moving the split.
        messages = [{"role": "system", "content": "sys"}]
        for i in range(8):
            messages.append({"role": "user", "content": f"q{i} {big}"})
            messages.append({"role": "assistant", "content": f"a{i} {big}"})
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": "t1", "type": "function",
                                         "function": {"name": "read_file", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": "t1", "content": "file body"})
        messages.append({"role": "user", "content": f"follow-up {big}"})
        messages.append({"role": "assistant", "content": f"reply {big}"})
        messages.append({"role": "assistant", "content": f"more {big}"})

        result = await compact(messages, cfg, client, keep_last=2,
                               facts_store=store, turn_index=20)

        # No tool message may appear without a preceding assistant carrying tool_calls.
        for idx, m in enumerate(result):
            if m.get("role") == "tool":
                prev = result[idx - 1] if idx > 0 else {}
                assert prev.get("tool_calls"), f"orphan tool message at result[{idx}]"
