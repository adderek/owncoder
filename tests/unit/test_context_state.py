"""Context-budget awareness: the anti-thrash loop guard.

Without it: a large read pushes the turn over the compaction threshold,
compaction elides the read, the model re-reads the same file, repeat.
"""
from __future__ import annotations

import pytest

from agent.core import context_state as cs


@pytest.fixture(autouse=True)
def _reset():
    cs.reset()
    yield
    cs.reset()


class TestSnapshot:
    def test_headroom_and_fraction(self):
        cs.publish(used=19800, budget=24576, window=32768)
        snap = cs.current()
        assert snap.headroom == 24576 - 19800
        assert 0.60 < snap.fraction < 0.61

    def test_headroom_never_negative(self):
        cs.publish(used=30000, budget=24576, window=32768)
        assert cs.current().headroom == 0

    def test_would_survive(self):
        cs.publish(used=20000, budget=24576, window=32768)
        assert cs.current().would_survive(4000)
        assert not cs.current().would_survive(9000)

    def test_compactions_survive_republish(self):
        cs.publish(used=1, budget=2, window=3)
        cs.note_compaction()
        cs.publish(used=10, budget=20, window=30)
        assert cs.current().compactions == 1

    def test_note_compaction_without_publish(self):
        cs.note_compaction()
        assert cs.current().compactions == 1


class TestBudgetLine:
    def test_states_usage_and_headroom(self):
        cs.publish(used=19800, budget=24576, window=32768)
        line = cs.format_budget_line(cs.current())
        assert "60% used" in line
        assert "19.8k/32.8k" in line
        assert "headroom" in line

    def test_mentions_prior_compactions(self):
        cs.publish(used=19800, budget=24576, window=32768)
        cs.note_compaction()
        line = cs.format_budget_line(cs.current())
        assert "1×" in line
        assert "find_symbol" in line


class TestBudgetNotice:
    def _injected(self, kind, text):
        return {"role": "user", "content": text, "_injected_kind": kind}

    def test_silent_below_threshold(self):
        from agent.core.turn import _apply_budget_notice
        cs.publish(used=1000, budget=24576, window=32768)
        assert _apply_budget_notice([{"role": "user", "content": "hi"}], None, self._injected) == [
            {"role": "user", "content": "hi"}]

    def test_appends_notice_above_threshold(self):
        from agent.core.turn import _apply_budget_notice, _BUDGET_NOTICE_KIND
        cs.publish(used=25000, budget=24576, window=32768)
        out = _apply_budget_notice([{"role": "user", "content": "hi"}], None, self._injected)
        assert len(out) == 2
        assert out[-1]["_injected_kind"] == _BUDGET_NOTICE_KIND

    def test_only_one_notice_survives(self):
        from agent.core.turn import _apply_budget_notice, _BUDGET_NOTICE_KIND
        cs.publish(used=25000, budget=24576, window=32768)
        msgs = [{"role": "user", "content": "hi"}]
        for _ in range(3):
            msgs = _apply_budget_notice(msgs, None, self._injected)
        assert sum(1 for m in msgs if m.get("_injected_kind") == _BUDGET_NOTICE_KIND) == 1

    def test_stale_notice_dropped_when_usage_falls(self):
        from agent.core.turn import _apply_budget_notice, _BUDGET_NOTICE_KIND
        cs.publish(used=25000, budget=24576, window=32768)
        msgs = _apply_budget_notice([{"role": "user", "content": "hi"}], None, self._injected)
        cs.publish(used=2000, budget=24576, window=32768)
        msgs = _apply_budget_notice(msgs, None, self._injected)
        assert not any(m.get("_injected_kind") == _BUDGET_NOTICE_KIND for m in msgs)


class TestReadPreflight:
    def test_oversized_read_serves_outline_not_content(self, tmp_path, monkeypatch):
        from agent.config import Config
        from agent.tools.files import setup as files_setup, read_file

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        files_setup(cfg)
        body = "def alpha():\n" + ("    x = 1  # padding padding padding\n" * 4000)
        (tmp_path / "big.py").write_text(body)

        cs.publish(used=20000, budget=20500, window=32768)  # ~500 tokens of headroom
        r = read_file("big.py")
        assert r["metadata"]["budget_limited"] is True
        assert r["metadata"]["estimated_tokens"] > r["metadata"]["headroom_tokens"]
        assert "would be summarised away" in r["content"]
        assert "1:def alpha():" in r["content"], "first window still served"

    def test_read_fits_when_headroom_is_ample(self, tmp_path):
        from agent.config import Config
        from agent.tools.files import setup as files_setup, read_file

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        files_setup(cfg)
        (tmp_path / "small.py").write_text("def alpha():\n    return 1\n")
        cs.publish(used=1000, budget=24576, window=32768)
        r = read_file("small.py")
        assert not r["metadata"].get("budget_limited")
        assert "return 1" in r["content"]

    def test_ranged_read_is_never_budget_blocked(self, tmp_path):
        from agent.config import Config
        from agent.tools.files import setup as files_setup, read_file

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        files_setup(cfg)
        (tmp_path / "big.py").write_text("x = 1\n" * 5000)
        cs.publish(used=20000, budget=20100, window=32768)
        r = read_file("big.py", start_line=1, end_line=3)
        assert not r["metadata"].get("budget_limited")


class TestRereadAfterCompactionHint:
    def test_hint_fires_once_after_compaction(self):
        from agent.core.tool_hints import reset_tool_hints, tool_hints

        reset_tool_hints()
        cs.publish(used=1000, budget=24576, window=32768)
        assert tool_hints("read_file", {"path": "a.py"}, {"content": ""}) == []
        cs.note_compaction()
        h = tool_hints("read_file", {"path": "a.py"}, {"content": ""})
        assert h and "elided by compaction" in h[0]
        later = tool_hints("read_file", {"path": "a.py"}, {"content": ""})
        assert not any("elided by compaction" in h for h in later)
        reset_tool_hints()

    def test_no_hint_for_first_read_of_a_file(self):
        from agent.core.tool_hints import reset_tool_hints, tool_hints

        reset_tool_hints()
        cs.note_compaction()
        assert tool_hints("read_file", {"path": "fresh.py"}, {"content": ""}) == []
        reset_tool_hints()


def test_notice_never_splits_tool_calls_from_results():
    """A user message between an assistant tool_calls turn and its tool results
    is an invalid exchange — the notice waits for the next iteration."""
    from agent.core.turn import _apply_budget_notice, _BUDGET_NOTICE_KIND

    cs.publish(used=25000, budget=24576, window=32768)
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
    ]
    out = _apply_budget_notice(msgs, None, lambda k, t: {"role": "user", "content": t, "_injected_kind": k})
    assert not any(m.get("_injected_kind") == _BUDGET_NOTICE_KIND for m in out)
    assert out == msgs
