"""The record of how the tool surface changed, and why.

A tool schema edit changes the agent's behaviour and leaves no trace in any
transcript. Without this, "it used to handle that correctly" has nowhere to be
checked — which is the whole reason the ledger stores a reason and not just a
diff.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from agent.core import tool_ledger


@pytest.fixture()
def project(tmp_path):
    (tmp_path / ".agent").mkdir()
    return N(tools=N(working_dir=str(tmp_path), agent_dir=".agent"))


_UNSET = object()


def _schema(name, description="does a thing", properties=None, required=_UNSET):
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object",
                       "properties": properties or {"path": {"type": "string"}},
                       "required": ["path"] if required is _UNSET else required},
    }}


@pytest.fixture(autouse=True)
def _no_git_lookup(monkeypatch):
    """Most tests are about the diff, not about git; the reason is its own test."""
    monkeypatch.setattr(tool_ledger, "_reason", lambda name: {"reason": "test"})


class TestBaseline:
    def test_the_first_run_records_every_tool_as_added(self, project):
        entries = tool_ledger.record_changes(project, [_schema("read_file"), _schema("edit_file")])
        assert {e["tool"] for e in entries} == {"read_file", "edit_file"}
        assert all(e["change"] == "added" for e in entries)

    def test_an_unchanged_surface_records_nothing_the_second_time(self, project):
        schemas = [_schema("read_file")]
        tool_ledger.record_changes(project, schemas)
        assert tool_ledger.record_changes(project, schemas) == []
        assert len(tool_ledger.history(project)) == 1

    def test_no_schemas_is_not_an_error(self, project):
        assert tool_ledger.record_changes(project, []) == []


class TestChanges:
    def test_a_new_tool_is_recorded_as_added(self, project):
        tool_ledger.record_changes(project, [_schema("read_file")])
        entries = tool_ledger.record_changes(project, [_schema("read_file"), _schema("grep_code")])
        assert [(e["change"], e["tool"]) for e in entries] == [("added", "grep_code")]

    def test_a_dropped_tool_is_recorded_as_removed(self, project):
        tool_ledger.record_changes(project, [_schema("read_file"), _schema("old_tool")])
        entries = tool_ledger.record_changes(project, [_schema("read_file")])
        assert [(e["change"], e["tool"]) for e in entries] == [("removed", "old_tool")]

    def test_a_removed_tool_coming_back_is_an_addition_again(self, project):
        tool_ledger.record_changes(project, [_schema("t")])
        tool_ledger.record_changes(project, [])
        entries = tool_ledger.record_changes(project, [_schema("t")])
        assert entries[0]["change"] == "added"

    def test_an_edited_description_is_recorded_as_a_description_change(self, project):
        """Descriptions steer the model; editing one changes behaviour as much
        as editing code."""
        tool_ledger.record_changes(project, [_schema("t", description="old")])
        entries = tool_ledger.record_changes(project, [_schema("t", description="new")])
        assert entries[0]["change"] == "changed"
        assert entries[0]["changed"] == ["description"]

    def test_a_new_parameter_is_recorded(self, project):
        tool_ledger.record_changes(project, [_schema("t")])
        entries = tool_ledger.record_changes(project, [_schema(
            "t", properties={"path": {"type": "string"}, "limit": {"type": "integer"}})])
        assert entries[0]["changed"] == ["parameters"]

    def test_a_newly_required_parameter_is_called_out_separately(self, project):
        """Tightening `required` breaks calls that used to work, so it deserves
        its own label rather than hiding inside 'parameters'."""
        tool_ledger.record_changes(project, [_schema("t", required=[])])
        entries = tool_ledger.record_changes(project, [_schema("t", required=["path"])])
        assert set(entries[0]["changed"]) == {"parameters", "required"}

    def test_key_order_alone_is_not_a_change(self, project):
        """Serialisation order must not manufacture history."""
        tool_ledger.record_changes(project, [_schema("t")])
        reordered = {"type": "function", "function": {
            "parameters": {"required": ["path"], "type": "object",
                           "properties": {"path": {"type": "string"}}},
            "description": "does a thing", "name": "t"}}
        assert tool_ledger.record_changes(project, [reordered]) == []


class TestReason:
    @pytest.fixture(autouse=True)
    def _real_reason(self, monkeypatch):
        """This class is about the reason itself, so the stub is off here."""
        monkeypatch.undo()

    def test_the_reason_is_the_commit_that_touched_the_implementing_file(self, tmp_path,
                                                                         monkeypatch):
        repo = tmp_path / "repo"
        repo.mkdir()
        env = {"GIT_AUTHOR_NAME": "Ada", "GIT_AUTHOR_EMAIL": "a@e",
               "GIT_COMMITTER_NAME": "Ada", "GIT_COMMITTER_EMAIL": "a@e",
               "PATH": "/usr/bin:/bin"}
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
        impl = repo / "impl.py"
        impl.write_text("def t():\n    return 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "impl.py"], cwd=repo, check=True, env=env)
        subprocess.run(["git", "commit", "-qm", "narrow the path argument"],
                       cwd=repo, check=True, env=env)

        monkeypatch.setattr(tool_ledger, "_source_file", lambda name: impl)
        found = tool_ledger._reason("t")
        assert found["reason"] == "narrow the path argument"
        assert found["file"] == str(impl)

    def test_an_unlocatable_tool_says_so_rather_than_guessing(self, monkeypatch):
        monkeypatch.setattr(tool_ledger, "_source_file", lambda name: None)
        assert "not found" in tool_ledger._reason("t")["reason"]

    def test_a_real_registered_tool_resolves_to_its_own_file(self):
        """The registry holds live functions, so attribution is exact — no
        grepping for where a tool might be defined."""
        from agent.config import Config
        from agent.tools import load_all_tools

        load_all_tools(config=Config())
        path = tool_ledger._source_file("read_file")
        assert path is not None and path.name.endswith(".py")
        assert "tools" in path.parts


class TestHistory:
    def test_it_can_be_filtered_to_one_tool(self, project):
        tool_ledger.record_changes(project, [_schema("a"), _schema("b")])
        assert [e["tool"] for e in tool_ledger.history(project, tool="a")] == ["a"]

    def test_replay_bookkeeping_stays_out_of_the_public_view(self, project):
        tool_ledger.record_changes(project, [_schema("a")])
        entry = tool_ledger.public_history(project)[0]
        assert not [k for k in entry if k.startswith("_")]
        assert entry["tool"] == "a" and entry["change"] == "added"

    def test_a_corrupt_line_does_not_lose_the_rest(self, project):
        tool_ledger.record_changes(project, [_schema("a")])
        with tool_ledger.history_path(project).open("a", encoding="utf-8") as f:
            f.write("{ truncated\n")
        assert len(tool_ledger.history(project)) == 1

    def test_a_truncated_ledger_degrades_to_missing_history_not_a_wrong_diff(self, project):
        """State is replayed from the entries, so losing entries loses history —
        it must not silently invent a change that never happened."""
        tool_ledger.record_changes(project, [_schema("a")])
        tool_ledger.history_path(project).write_text("", encoding="utf-8")
        entries = tool_ledger.record_changes(project, [_schema("a")])
        assert [e["change"] for e in entries] == ["added"]     # re-baselined, not "changed"

    def test_no_ledger_is_an_empty_list(self, project):
        assert tool_ledger.history(project) == []

    def test_an_unwritable_ledger_does_not_break_startup(self, project, monkeypatch):
        def _boom(*a, **kw):
            raise OSError("read-only")

        monkeypatch.setattr(Path, "open", _boom)
        assert tool_ledger.record_changes(project, [_schema("a")]) == []


class TestQueryTool:
    def test_the_tool_reports_recorded_changes(self, project):
        from agent.tools import core_rules_tools

        core_rules_tools.setup(project)
        tool_ledger.record_changes(project, [_schema("read_file")])
        result = core_rules_tools.tool_change_history(tool="read_file")
        assert result["count"] == 1
        assert result["entries"][0]["change"] == "added"

    def test_an_unknown_tool_says_nothing_is_recorded(self, project):
        from agent.tools import core_rules_tools

        core_rules_tools.setup(project)
        result = core_rules_tools.tool_change_history(tool="nope")
        assert result["count"] == 0
        assert "No recorded change" in result["note"]
