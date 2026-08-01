"""Unit tests for the readline UI's changeset display (ui/readline_loop.py).

`core/changeset.py` builds the Changeset; everything under test here is the
plain-scrollback presentation layer bolted onto it: the colored end-of-round
block, the numbered `/changes` file list, drilling into a single file's diff,
and the session rollup that merges every round's changeset by path. If any of
these regress, the readline UI goes back to being silent about what a round
changed.
"""
from __future__ import annotations

from agent.config.models import ThemeConfig
from agent.core.changeset import Changeset, FileChange
from agent.ui.readline_loop import (
    changeset_enabled,
    changeset_spill_dir,
    format_changeset_block,
    format_changeset_file_diff,
    format_changeset_file_list,
    merge_changesets,
    session_rollup_enabled,
)

_theme = ThemeConfig()


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


class TestFormatChangesetBlock:
    def test_empty_changeset_renders_nothing(self):
        assert format_changeset_block(Changeset(), _theme) == []
        assert format_changeset_block(None, _theme) == []

    def test_count_tier_is_just_the_headline(self):
        cs = Changeset(files=[_fc("a.py", added=5, removed=1)], tier="count")
        lines = format_changeset_block(cs, _theme)
        assert len(lines) == 1
        assert "1 file changed" in lines[0]
        assert "+5" in lines[0] and "-1" in lines[0]

    def test_list_tier_has_a_row_per_file_but_no_diff_text(self):
        cs = Changeset(
            files=[_fc("a.py", added=1, diff="+x\n"), _fc("b.py", removed=1, diff="-y\n")],
            tier="list",
        )
        lines = format_changeset_block(cs, _theme)
        joined = "\n".join(lines)
        assert "a.py" in joined and "b.py" in joined
        assert "+x" not in joined and "-y" not in joined

    def test_inline_tier_includes_the_diff_text(self):
        cs = Changeset(files=[_fc("a.py", added=1, diff="+hello\n")], tier="inline")
        joined = "\n".join(format_changeset_block(cs, _theme))
        assert "+hello" in joined

    def test_foreign_edit_note_uses_warning_color(self):
        f = _fc("a.py", added=1, diff="+x\n", foreign_edit=True, foreign_actors=["other-agent"])
        cs = Changeset(files=[f], tier="inline")
        lines = format_changeset_block(cs, _theme)
        note_lines = [l for l in lines if "also edited by other-agent" in l]
        assert len(note_lines) == 1
        assert f"[{_theme.warning}]" in note_lines[0]

    def test_feature_disabled_means_nothing_to_show(self):
        assert not changeset_enabled(_Config(_Section(enabled=False)))
        assert changeset_enabled(_Config(_Section(enabled=True)))


class TestChangesFileList:
    def test_lists_are_numbered_from_one(self):
        cs = Changeset(files=[_fc("a.py", added=1), _fc("b.py", removed=2)], tier="list")
        lines = format_changeset_file_list(cs, _theme)
        assert any("1. " in l and "a.py" in l for l in lines)
        assert any("2. " in l and "b.py" in l for l in lines)

    def test_empty_changeset_gives_no_rows(self):
        assert format_changeset_file_list(Changeset(), _theme) == []
        assert format_changeset_file_list(None, _theme) == []


class TestChangesFileDiff:
    def test_valid_index_returns_the_diff(self):
        cs = Changeset(files=[_fc("a.py", added=1, diff="+hello\n")], tier="inline")
        ok, lines = format_changeset_file_diff(cs, 1)
        assert ok
        assert any("+hello" in l for l in lines)

    def test_out_of_range_index_is_a_helpful_message(self):
        cs = Changeset(files=[_fc("a.py", added=1, diff="+hello\n")], tier="inline")
        ok, lines = format_changeset_file_diff(cs, 5)
        assert not ok
        assert "1-1" in lines[0] or "1" in lines[0]

    def test_zero_and_negative_index_are_out_of_range(self):
        cs = Changeset(files=[_fc("a.py", added=1, diff="+hello\n")], tier="inline")
        ok, _ = format_changeset_file_diff(cs, 0)
        assert not ok
        ok, _ = format_changeset_file_diff(cs, -1)
        assert not ok

    def test_no_changeset_yet_is_a_helpful_message(self):
        ok, lines = format_changeset_file_diff(None, 1)
        assert not ok
        assert lines

    def test_binary_file_says_so_instead_of_diffing(self):
        cs = Changeset(files=[_fc("img.png", binary=True)], tier="inline")
        ok, lines = format_changeset_file_diff(cs, 1)
        assert ok
        assert any("binary" in l for l in lines)


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


class TestFeatureToggles:
    def test_session_rollup_toggle(self):
        assert session_rollup_enabled(_Config(_Section(session_rollup=True)))
        assert not session_rollup_enabled(_Config(_Section(session_rollup=False)))

    def test_missing_ui_section_defaults_to_enabled(self):
        class _Bare:
            pass
        assert changeset_enabled(_Bare())
        assert session_rollup_enabled(_Bare())


class TestChangesetSpillDir:
    def test_no_session_or_no_changeset_gives_no_dir(self):
        assert changeset_spill_dir(None, Changeset()) is None
        assert changeset_spill_dir(object(), None) is None
