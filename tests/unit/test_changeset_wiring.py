"""The round wiring that turns an edit journal into a per-round changeset.

core/changeset.py can build a changeset from any journal window; these cover the
part that decides *which* window a round gets, keeps the result on the agent,
and hands it to the UI callback and the QA log. Everything here goes through
the agent's own helpers rather than a live turn, because a turn needs a model.
"""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.core.agent import Agent
from agent.tools.files import setup as files_setup, write_file, _undo_stack
import agent.core.changeset as cs
import agent.core.checkpoint as ckpt


@pytest.fixture()
def agent_in(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    _undo_stack.clear()
    ckpt.reset()
    a = object.__new__(Agent)          # bypass __init__: it builds an LLM client
    a.config = cfg
    a._session_id = None
    a.last_changeset = None
    return a, tmp_path


class TestCollectChangeset:
    def test_it_measures_from_the_window_it_was_given(self, agent_in):
        a, tmp = agent_in
        write_file("before.py", "earlier round\n")
        since = cs.open_window()
        write_file("during.py", "this round\n")
        c = a._collect_changeset(1, since)
        assert [f.path for f in c.files] == ["during.py"]
        assert c.turn_id == 1

    def test_it_honours_the_configured_limits(self, agent_in):
        a, tmp = agent_in
        a.config.ui.changeset.inline_max_files = 0     # nothing may open inline
        since = cs.open_window()
        write_file("a.py", "x\n")
        assert a._collect_changeset(1, since).tier == "list"

    def test_disabling_the_feature_yields_an_empty_changeset(self, agent_in):
        a, tmp = agent_in
        a.config.ui.changeset.enabled = False
        since = cs.open_window()
        write_file("a.py", "x\n")
        c = a._collect_changeset(1, since)
        assert not c and c.turn_id == 1

    def test_a_collection_failure_never_breaks_the_round(self, agent_in, monkeypatch):
        a, tmp = agent_in
        monkeypatch.setattr(cs, "collect", lambda *args, **kw: 1 / 0)
        since = cs.open_window()
        write_file("a.py", "x\n")
        c = a._collect_changeset(4, since)
        assert not c and c.turn_id == 4

    def test_a_round_that_changed_nothing_is_falsy(self, agent_in):
        a, _ = agent_in
        assert not a._collect_changeset(1, cs.open_window())


class TestSpillDir:
    def test_without_a_session_diffs_stay_in_memory(self, agent_in):
        a, _ = agent_in
        assert a._changeset_spill_dir(1) is None

    def test_with_a_session_it_is_per_turn(self, agent_in, monkeypatch):
        a, tmp = agent_in
        a._session_id = "sess-1"
        monkeypatch.setattr("agent.memory.session.get_session_full_dir",
                            lambda sid: tmp / "sessions" / sid)
        d = a._changeset_spill_dir(7)
        assert d.parts[-3:] == ("sess-1", "changesets", "7")


class TestPathTrackingIsShared:
    def test_the_agent_uses_the_one_definition(self):
        """core/agent.py used to inline this parse; three copies had to agree."""
        import agent.core.agent as mod
        assert mod.paths_from_tool_call is cs.paths_from_tool_call
