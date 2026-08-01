"""The session rollup (core/changeset.py) — one changeset for a whole session.

`merge_changesets` used to live in ui/readline_loop.py, where only the readline
UI could reach it; the merge tests moved here with it when the Textual and
browser UIs started rendering the same rollup. All three read this code, so
these are the semantics that must hold for every one of them.

`session_rollup` adds the other half: the rounds come from the QA log, not from
whatever the UI happens to be holding in memory, so a *resumed* session rolls
up the whole session rather than only the rounds since it was reloaded.
"""
from __future__ import annotations

from agent.core.changeset import (
    Changeset,
    FileChange,
    changeset_enabled,
    merge_changesets,
    rollup_json,
    rollup_line,
    session_changesets,
    session_rollup,
    session_rollup_enabled,
)


class _Section:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _UI:
    def __init__(self, changeset):
        self.changeset = changeset


class _Config:
    def __init__(self, changeset=None):
        self.ui = _UI(changeset if changeset is not None else _Section(enabled=True, session_rollup=True))


def _fc(path, added=0, removed=0, status="modified", diff=None, **kw):
    return FileChange(path=path, added=added, removed=removed, status=status, diff=diff, **kw)


class TestMergeChangesets:
    def test_a_path_touched_twice_is_counted_once_with_summed_churn(self):
        round1 = Changeset(files=[_fc("a.py", added=3, removed=1), _fc("b.py", added=2)])
        round2 = Changeset(files=[_fc("a.py", added=1, removed=2), _fc("c.py", removed=4)])
        merged = merge_changesets([round1, round2])
        paths = [f.path for f in merged.files]
        assert paths.count("a.py") == 1
        a = next(f for f in merged.files if f.path == "a.py")
        assert (a.added, a.removed) == (4, 3)
        assert set(paths) == {"a.py", "b.py", "c.py"}

    def test_empty_list_merges_to_an_empty_changeset(self):
        merged = merge_changesets([])
        assert merged.file_count == 0

    def test_a_file_created_then_edited_is_still_added_for_the_session(self):
        """Status is measured from the session start, not from the last round."""
        round1 = Changeset(files=[_fc("a.py", added=3, status="added")])
        round2 = Changeset(files=[_fc("a.py", added=1, status="modified")])
        assert merge_changesets([round1, round2]).files[0].status == "added"

    def test_a_file_deleted_in_a_later_round_reads_as_deleted(self):
        round1 = Changeset(files=[_fc("a.py", added=3, status="added")])
        round2 = Changeset(files=[_fc("a.py", removed=3, status="deleted")])
        assert merge_changesets([round1, round2]).files[0].status == "deleted"

    def test_foreign_actors_accumulate_across_rounds(self):
        round1 = Changeset(files=[_fc("a.py", added=1, foreign_edit=True, foreign_actors=["x"])])
        round2 = Changeset(files=[_fc("a.py", added=1, foreign_edit=True, foreign_actors=["y"])])
        merged = merge_changesets([round1, round2])
        a = merged.files[0]
        assert a.foreign_edit
        assert set(a.foreign_actors) == {"x", "y"}

    def test_diff_text_is_not_merged(self):
        """The rollup is a file list; a diff belongs to one round."""
        round1 = Changeset(files=[_fc("a.py", added=1, diff="--- a\n+++ b\n+one\n")])
        round2 = Changeset(files=[_fc("a.py", added=1, diff="--- a\n+++ b\n+two\n")])
        assert merge_changesets([round1, round2]).files[0].diff is None


class TestFeatureToggles:
    def test_session_rollup_toggle(self):
        assert session_rollup_enabled(_Config(_Section(session_rollup=True)))
        assert not session_rollup_enabled(_Config(_Section(session_rollup=False)))

    def test_missing_ui_section_defaults_to_enabled(self):
        class _Bare:
            pass
        assert changeset_enabled(_Bare())
        assert session_rollup_enabled(_Bare())


def _a_record(turn_id: int, files: list[dict]) -> dict:
    """An A-record as the QA log stores it, with a modern ``changeset`` key."""
    return {
        "turn_id": turn_id,
        "changeset": {"turn_id": turn_id, "tier": "list", "files": files},
    }


class TestSessionChangesets:
    def test_rounds_are_rebuilt_from_the_qa_log(self, monkeypatch):
        history = [
            (1, {}, _a_record(1, [{"path": "a.py", "added": 3}])),
            (2, {}, _a_record(2, [{"path": "b.py", "added": 1}])),
        ]
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: history)
        rounds = session_changesets("s1")
        assert [cs.turn_id for cs in rounds] == [1, 2]
        assert [f.path for cs in rounds for f in cs.files] == ["a.py", "b.py"]

    def test_a_legacy_record_still_contributes(self, monkeypatch):
        """Sessions written before the feature carry only modified_files."""
        history = [(1, {}, {"turn_id": 1, "modified_files": ["old.py"]})]
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: history)
        assert [f.path for f in session_changesets("s1")[0].files] == ["old.py"]

    def test_an_unreadable_log_is_no_rollup_rather_than_a_crash(self, monkeypatch):
        def _boom(sid):
            raise OSError("no such session")
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", _boom)
        assert session_changesets("s1") == []


class TestSessionRollup:
    def test_a_resumed_session_rolls_up_rounds_it_never_saw_live(self, monkeypatch):
        """The point of reading the log: rounds from before the reload count."""
        history = [(1, {}, _a_record(1, [{"path": "a.py", "added": 3}]))]
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: history)
        live = Changeset(turn_id=2, files=[_fc("b.py", added=1)])
        rollup = session_rollup("s1", [live])
        assert sorted(f.path for f in rollup.files) == ["a.py", "b.py"]

    def test_a_round_in_both_the_log_and_memory_is_counted_once(self, monkeypatch):
        """The finished round is written to the log at turn end, so the live
        list and the log overlap by exactly one round most of the time."""
        history = [(1, {}, _a_record(1, [{"path": "a.py", "added": 3}]))]
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: history)
        same_round = Changeset(turn_id=1, files=[_fc("a.py", added=3)])
        rollup = session_rollup("s1", [same_round])
        assert rollup.file_count == 1
        assert rollup.files[0].added == 3

    def test_an_untagged_live_round_is_kept(self, monkeypatch):
        """turn_id 0 cannot be matched against the log, so it is not dropped."""
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: [])
        rollup = session_rollup("s1", [Changeset(files=[_fc("a.py", added=1)])])
        assert [f.path for f in rollup.files] == ["a.py"]

    def test_no_session_id_falls_back_to_the_live_rounds(self, monkeypatch):
        def _unexpected(sid):
            raise AssertionError("must not read the log without a session id")
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", _unexpected)
        rollup = session_rollup("", [Changeset(turn_id=1, files=[_fc("a.py", added=1)])])
        assert [f.path for f in rollup.files] == ["a.py"]


class TestRollupWording:
    def test_every_ui_says_the_same_thing(self):
        cs = Changeset(files=[_fc("a.py", added=4, removed=1), _fc("b.py", added=2)])
        assert rollup_line(cs) == "session: 2 files changed, +6 -1"

    def test_an_empty_rollup_has_no_line(self):
        assert rollup_line(Changeset()) == ""
        assert rollup_line(None) == ""
        assert rollup_json(Changeset()) is None

    def test_the_browser_gets_the_line_and_its_numbers(self):
        cs = Changeset(files=[_fc("a.py", added=4, removed=1)])
        assert rollup_json(cs) == {
            "line": "session: 1 file changed, +4 -1",
            "files": 1, "added": 4, "removed": 1,
        }
