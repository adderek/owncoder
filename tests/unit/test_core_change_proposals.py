"""The agent's only path to the core rules: propose, never write.

The core exists to be the part of the prompt the agent's learning cannot edit.
That guarantee is worth nothing if there is *also* a tool that edits it, so
these tests pin the shape of the surface as much as its behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace as N

import pytest

from agent import ideas as ideas_mod
from agent.tools import core_rules_tools


@pytest.fixture()
def project(tmp_path, monkeypatch):
    (tmp_path / ".agent").mkdir()
    config = N(tools=N(working_dir=str(tmp_path), agent_dir=".agent"))
    ideas_mod.configure(str(tmp_path), ".agent")
    core_rules_tools.setup(config)
    yield config
    ideas_mod.configure(str(tmp_path), ".agent")


class TestProposal:
    def test_a_proposal_lands_in_the_backlog(self, project):
        result = core_rules_tools.propose_core_change(
            title="Require a second read before deleting",
            rationale="Deleted a file this session that the user had just written.",
            proposed_text="- Re-read a file immediately before deleting it.",
        )
        assert result["proposed"] is True
        ideas = ideas_mod.list_ideas()
        assert len(ideas) == 1
        assert ideas[0]["type"] == "core_change"
        assert ideas[0]["status"] == "raw"
        assert ideas[0]["source"] == "agent"

    def test_the_rationale_and_proposed_text_are_kept(self, project):
        core_rules_tools.propose_core_change(
            title="t", rationale="because X happened",
            proposed_text="- new rule", replaces="- old rule")
        body = ideas_mod.list_ideas()[0]["body"]
        assert "because X happened" in body
        assert "- new rule" in body
        assert "- old rule" in body

    def test_it_says_out_loud_that_nothing_changed(self, project):
        """The model must not report a filed proposal as an applied rule."""
        result = core_rules_tools.propose_core_change(title="t", rationale="r")
        assert "unchanged" in result["note"]

    def test_the_body_says_who_applies_it(self, project):
        core_rules_tools.propose_core_change(title="t", rationale="r")
        assert "human" in ideas_mod.list_ideas()[0]["body"]

    def test_priority_is_clamped(self, project):
        core_rules_tools.propose_core_change(title="t", rationale="r", priority=99)
        assert ideas_mod.list_ideas()[0]["priority"] == 5

    def test_a_missing_store_is_reported_not_swallowed(self, monkeypatch):
        monkeypatch.setattr(core_rules_tools, "_ideas_store", lambda: None)
        result = core_rules_tools.propose_core_change(title="t", rationale="r")
        assert "error" in result and result.get("proposed") is not True

    def test_core_change_is_a_real_idea_type(self):
        """A type the store rejects would be silently rewritten to "idea" and
        the proposal would vanish into the general backlog."""
        from agent.ideas.store import IDEA_TYPES

        assert "core_change" in IDEA_TYPES

    def test_the_proposal_does_not_touch_the_core_file(self, project):
        from agent.core import core_rules

        before = core_rules.CORE_PATH.read_bytes()
        core_rules_tools.propose_core_change(title="t", rationale="r",
                                             proposed_text="- sneaky rule")
        assert core_rules.CORE_PATH.read_bytes() == before
        assert not core_rules.project_core_path(project).exists()


class TestNoWriteSurface:
    def test_no_registered_tool_offers_to_edit_the_core(self):
        """The guarantee is structural: there must be no write path at all."""
        import inspect

        source = inspect.getsource(core_rules_tools)
        assert "write_text" not in source
        assert "open(" not in source

    def test_only_read_and_propose_are_registered(self):
        from agent.tools import get_tool

        assert get_tool("propose_core_change") is not None
        assert get_tool("core_rules_history") is not None
        assert get_tool("edit_core_rules") is None
        assert get_tool("set_core_rules") is None


class TestHistoryTool:
    def test_it_reports_the_current_digest_and_entries(self, project):
        from agent.core import core_rules

        core_rules.record_state(project)
        result = core_rules_tools.core_rules_history()
        assert result["current_digest"] == core_rules.digest(project)
        assert result["count"] == 1
        assert result["entries"][0]["files"]

    def test_an_empty_history_explains_itself(self, project):
        result = core_rules_tools.core_rules_history()
        assert result["count"] == 0
        assert "has not changed" in result["note"]

    def test_the_limit_is_at_least_one(self, project):
        from agent.core import core_rules

        core_rules.record_state(project)
        assert core_rules_tools.core_rules_history(limit=0)["count"] == 1
