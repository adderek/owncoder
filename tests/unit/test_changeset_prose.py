"""One-line model-generated intent summary for a round's changeset.

If this regresses: the "off" default could start making silent background
model calls on every round; a diff body could leak into a prompt sent to a
background model (the one hard requirement of this feature); a model failure
could raise instead of degrading to ""; or an over-long reply could blow past
the ≤15-word cap this module promises callers.
"""
from __future__ import annotations

import asyncio
import json

import pytest

import agent.core.changeset as cs
import agent.core.changeset_prose as csp
from agent.config import Config
from agent.core.agent import Agent

_DIFF_BODY_MARKER = "SECRET_TOKEN_abc123_this_must_never_leave_the_diff_body"


def _changeset_with_diff() -> cs.Changeset:
    fc = cs.FileChange(
        path="agent/core/thing.py", added=10, removed=3, status="modified",
        diff=f"--- a/thing.py\n+++ b/thing.py\n@@ -1,1 +1,1 @@\n-old\n+{_DIFF_BODY_MARKER}\n",
    )
    fc2 = cs.FileChange(path="agent/core/new_thing.py", added=20, removed=0, status="added")
    return cs.Changeset(turn_id=1, files=[fc, fc2], tier="list")


class _StubLLM:
    def __init__(self, reply="", exc=None, captured=None):
        self._reply = reply
        self._exc = exc
        self._captured = captured if captured is not None else []

    async def __call__(self, config, system_prompt, content):
        self._captured.append((system_prompt, content))
        if self._exc is not None:
            raise self._exc
        return self._reply


class TestSummarizeNeverLeaksDiffBodies:
    async def test_prompt_carries_file_names_and_counts(self, monkeypatch):
        captured = []
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM("did a thing", captured=captured))

        out = await csp.summarize(Config(), _changeset_with_diff())

        assert out == "did a thing"
        assert len(captured) == 1
        _, content = captured[0]
        assert "thing.py" in content
        assert "new_thing.py" in content
        assert "+10" in content and "-3" in content

    async def test_prompt_never_contains_diff_body_text(self, monkeypatch):
        captured = []
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM("ok", captured=captured))

        await csp.summarize(Config(), _changeset_with_diff())

        _, content = captured[0]
        assert _DIFF_BODY_MARKER not in content
        assert "@@" not in content
        assert "---" not in content


class TestSummarizeDegradesSafely:
    async def test_empty_changeset_makes_no_call(self, monkeypatch):
        called = {"n": 0}

        async def _boom(*a, **k):
            called["n"] += 1
            raise AssertionError("should not be called for an empty changeset")

        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _boom)

        out = await csp.summarize(Config(), cs.Changeset())
        assert out == ""
        assert called["n"] == 0

    async def test_model_exception_yields_empty_string(self, monkeypatch):
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line",
                             _StubLLM(exc=RuntimeError("model unavailable")))

        out = await csp.summarize(Config(), _changeset_with_diff())
        assert out == ""

    async def test_cancellation_propagates_instead_of_being_swallowed(self, monkeypatch):
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line",
                             _StubLLM(exc=asyncio.CancelledError()))

        with pytest.raises(asyncio.CancelledError):
            await csp.summarize(Config(), _changeset_with_diff())


class TestWordCap:
    async def test_an_overlong_reply_is_capped_to_fifteen_words(self, monkeypatch):
        long_reply = " ".join(f"word{i}" for i in range(40))
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM(long_reply))

        out = await csp.summarize(Config(), _changeset_with_diff())
        assert len(out.split()) <= 15

    async def test_a_short_reply_passes_through_unchanged_apart_from_trailing_punctuation(self, monkeypatch):
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM("renamed the widget helper."))

        out = await csp.summarize(Config(), _changeset_with_diff())
        assert out == "renamed the widget helper"


class TestSummarizeAndPersist:
    async def test_it_writes_prose_into_the_already_persisted_a_record(self, tmp_path, monkeypatch):
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM("consolidated the parsers"))

        class _FakeQALogger:
            def _get_a_dir(self):
                return tmp_path

        a_path = tmp_path / "A-1.json"
        a_path.write_text(json.dumps({"turn_id": 7, "changeset": {"tier": "list"}}), encoding="utf-8")

        prose = await csp.summarize_and_persist(Config(), _changeset_with_diff(), _FakeQALogger(), 7)

        assert prose == "consolidated the parsers"
        on_disk = json.loads(a_path.read_text(encoding="utf-8"))
        assert on_disk["changeset"]["prose"] == "consolidated the parsers"

    async def test_no_matching_a_record_does_not_raise(self, tmp_path, monkeypatch):
        import agent.summarizer as sumr
        monkeypatch.setattr(sumr, "_call_llm_one_line", _StubLLM("consolidated the parsers"))

        class _FakeQALogger:
            def _get_a_dir(self):
                return tmp_path  # empty dir, no A file at all

        prose = await csp.summarize_and_persist(Config(), _changeset_with_diff(), _FakeQALogger(), 7)
        assert prose == "consolidated the parsers"


class TestChangesetDataclass:
    def test_prose_defaults_to_empty_string(self):
        assert cs.Changeset().prose == ""

    def test_prose_round_trips_through_json(self):
        c = _changeset_with_diff()
        c.prose = "reworked the changeset diffstat plumbing"
        back = cs.from_json(cs.to_json(c))
        assert back.prose == "reworked the changeset diffstat plumbing"

    def test_missing_prose_key_loads_as_empty_string(self):
        assert cs.from_json({"files": []}).prose == ""


class TestAgentWiring:
    @pytest.fixture()
    def agent_in(self):
        cfg = Config()
        a = object.__new__(Agent)
        a.config = cfg
        return a

    def test_default_mode_is_off(self, agent_in):
        assert agent_in._changeset_prose_mode() == "off"

    async def test_off_mode_never_calls_the_summarizer(self, agent_in, monkeypatch):
        called = {"n": 0}

        async def _boom(*a, **k):
            called["n"] += 1
            raise AssertionError("off must not call the summarizer")

        monkeypatch.setattr(csp, "summarize", _boom)

        mode = agent_in._changeset_prose_mode()
        assert mode == "off"
        # Mirrors the guard in Agent.run_turn: _apply_changeset_prose is only
        # ever invoked when mode == "always".
        assert called["n"] == 0

    async def test_always_mode_awaits_and_sets_prose(self, agent_in, monkeypatch):
        agent_in.config.ui.changeset.prose_summary = "always"

        async def _fake_summarize(config, cs_arg):
            return "tightened the retry loop"

        monkeypatch.setattr(csp, "summarize", _fake_summarize)

        c = _changeset_with_diff()
        await agent_in._apply_changeset_prose(c)

        assert c.prose == "tightened the retry loop"

    async def test_apply_changeset_prose_never_raises(self, agent_in, monkeypatch):
        async def _boom(*a, **k):
            raise RuntimeError("summarizer exploded")

        monkeypatch.setattr(csp, "summarize", _boom)

        c = _changeset_with_diff()
        await agent_in._apply_changeset_prose(c)  # must not raise
        assert c.prose == ""

    def test_mode_reads_defensively_when_section_missing(self, agent_in):
        agent_in.config.ui.changeset = None
        assert agent_in._changeset_prose_mode() == "off"
