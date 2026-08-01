"""Unit tests for core.changeset — the per-round file-change summary.

The changeset is derived from the edit journal rather than from ``git diff``,
so these tests care most about the two things that choice buys: exact per-round
attribution, and noticing when somebody who is not this agent wrote the file.
"""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.tools.files import setup as files_setup, write_file, _undo_stack
import agent.core.changeset as cs
import agent.core.checkpoint as ckpt


@pytest.fixture()
def workdir(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    _undo_stack.clear()
    ckpt.reset()
    return tmp_path, cfg


def _collect(tmp, since, **kw):
    return cs.collect(since, working_dir=tmp, **kw)


class TestPathsFromToolCall:
    def test_write_and_patch_carry_a_path(self):
        assert cs.paths_from_tool_call("write_file", '{"path": "a.py"}') == ["a.py"]
        assert cs.paths_from_tool_call("patch_file", {"path": "b.py"}) == ["b.py"]

    def test_edit_file_carries_chunks(self):
        args = '{"chunks": [{"path": "a.py"}, {"path": "b.py"}, {"path": "a.py"}]}'
        assert cs.paths_from_tool_call("edit_file", args) == ["a.py", "b.py"]

    def test_edit_file_flat_form(self):
        """edit_file also accepts path+anchor+replacement without chunks."""
        args = '{"path": "a.py", "anchor": "x", "replacement": "y"}'
        assert cs.paths_from_tool_call("edit_file", args) == ["a.py"]

    def test_non_mutating_tool_names_nothing(self):
        assert cs.paths_from_tool_call("read_file", '{"path": "a.py"}') == []

    def test_unparseable_arguments_are_skipped(self):
        assert cs.paths_from_tool_call("write_file", "{not json") == []
        assert cs.paths_from_tool_call("write_file", "[]") == []
        assert cs.paths_from_tool_call("write_file", None) == []


class TestCollect:
    def test_a_created_file_reads_as_added(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("new.py", "a\nb\n")
        c = _collect(tmp, since)
        assert c.file_count == 1
        f = c.files[0]
        assert (f.path, f.status, f.added, f.removed) == ("new.py", "added", 2, 0)
        assert "+a" in f.diff and "+b" in f.diff

    def test_a_modified_file_counts_only_the_delta(self, workdir):
        tmp, _ = workdir
        (tmp / "a.py").write_text("keep\nold\n")
        since = cs.open_window()
        write_file("a.py", "keep\nnew\nextra\n")
        f = _collect(tmp, since).files[0]
        assert (f.status, f.added, f.removed) == ("modified", 2, 1)

    def test_edits_before_the_window_are_not_counted(self, workdir):
        tmp, _ = workdir
        write_file("a.py", "round one\n")
        since = cs.open_window()          # window opens *after* the first edit
        write_file("b.py", "round two\n")
        c = _collect(tmp, since)
        assert [f.path for f in c.files] == ["b.py"]

    def test_repeated_edits_use_the_earliest_pre_image(self, workdir):
        """Three writes in one round are one net change, measured from the start."""
        tmp, _ = workdir
        (tmp / "a.py").write_text("v0\n")
        since = cs.open_window()
        write_file("a.py", "v1\n")
        write_file("a.py", "v2\n")
        write_file("a.py", "v3\n")
        c = _collect(tmp, since)
        assert c.file_count == 1
        f = c.files[0]
        assert (f.added, f.removed) == (1, 1)
        assert "-v0" in f.diff and "+v3" in f.diff
        assert "v1" not in f.diff and "v2" not in f.diff

    def test_a_file_deleted_after_the_edit_is_reported_as_deleted(self, workdir):
        tmp, _ = workdir
        (tmp / "gone.py").write_text("one\ntwo\n")
        since = cs.open_window()
        write_file("gone.py", "one\ntwo\nthree\n")
        (tmp / "gone.py").unlink()
        f = _collect(tmp, since).files[0]
        assert f.status == "deleted"
        # Measured against the state at window start, which had two lines.
        assert f.removed == 2
        assert f.foreign_edit          # we did not delete it through a tool

    def test_a_file_created_then_deleted_in_one_round_nets_to_nothing(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("gone.py", "one\ntwo\n")
        (tmp / "gone.py").unlink()
        f = _collect(tmp, since).files[0]
        # It did not exist when the round opened and does not exist now, so
        # there is nothing to count — but somebody removed it behind our back.
        assert (f.removed, f.added) == (0, 0)
        assert f.foreign_edit

    def test_a_binary_file_is_flagged_and_not_diffed(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("b.bin", "placeholder\n")
        (tmp / "b.bin").write_bytes(b"\x00\x01\x02")
        f = _collect(tmp, since).files[0]
        assert f.binary and f.diff is None

    def test_an_empty_window_is_falsy(self, workdir):
        tmp, _ = workdir
        c = _collect(tmp, cs.open_window())
        assert not c and c.file_count == 0 and cs.render_text(c) == []

    def test_files_are_ordered_by_churn(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("small.py", "x\n")
        write_file("big.py", "".join(f"line{i}\n" for i in range(20)))
        assert [f.path for f in _collect(tmp, since).files] == ["big.py", "small.py"]

    def test_headline_counts_everything(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "1\n2\n")
        write_file("b.py", "3\n")
        assert _collect(tmp, since).headline() == "2 files changed, +3 -0"

    def test_headline_is_worded_for_one_file(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "1\n")
        assert _collect(tmp, since).headline().startswith("1 file changed")


class TestForeignEdits:
    def test_an_edit_behind_our_back_is_flagged_not_mis_diffed(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "mine\n")
        (tmp / "a.py").write_text("mine\nsomeone else\n")   # not through a tool
        f = _collect(tmp, since).files[0]
        assert f.foreign_edit
        assert f.foreign_actors == []                       # unattributed
        assert "outside any agent" in f.note()

    def test_an_untouched_file_is_not_flagged(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "mine\n")
        f = _collect(tmp, since).files[0]
        assert not f.foreign_edit and f.note() == ""

    def test_another_actor_is_named(self, workdir):
        """Two agents in one tree interleave into the same journal."""
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "mine\n")
        # Simulate a concurrent agent recording an edit to the same file.
        ckpt._journal.append({
            "seq": ckpt.current_seq() + 1, "path": "a.py", "before": "mine\n",
            "ts": 0.0, "actor": "owncoder-4711",
            "before_sha": None, "after_sha": None,
        })
        f = _collect(tmp, since).files[0]
        assert f.foreign_edit
        assert f.foreign_actors == ["owncoder-4711"]
        assert "owncoder-4711" in f.note()

    def test_another_actors_edits_are_not_claimed_as_ours(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        ckpt._journal.append({
            "seq": ckpt.current_seq() + 1, "path": "theirs.py", "before": None,
            "ts": 0.0, "actor": "owncoder-4711",
            "before_sha": None, "after_sha": None,
        })
        assert _collect(tmp, since).file_count == 0

    def test_an_unknown_post_state_is_not_read_as_unchanged(self, workdir):
        """A missing after_sha means "we cannot tell", so no false all-clear."""
        tmp, _ = workdir
        (tmp / "a.py").write_text("v0\n")
        since = cs.open_window()
        write_file("a.py", "v1\n")
        ckpt._journal[-1]["after_sha"] = None
        f = _collect(tmp, since).files[0]
        assert not f.foreign_edit      # unknown, so no claim either way
        assert f.note() == ""


class TestTiers:
    @pytest.mark.parametrize("files,churn,expected", [
        (1, 1, "inline"),
        (3, 80, "inline"),          # both limits exactly at the boundary
        (4, 10, "list"),            # too many files
        (2, 5000, "list"),          # few files, far too many lines
        (50, 99999, "list"),
        (51, 1, "count"),
    ])
    def test_boundaries(self, files, churn, expected):
        assert cs.pick_tier(files, churn, cs.Limits()) == expected

    def test_a_small_change_opens_inline(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "one\n")
        assert _collect(tmp, since).tier == "inline"

    def test_a_wide_change_opens_as_a_list(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        for i in range(5):
            write_file(f"f{i}.py", "x\n")
        assert _collect(tmp, since).tier == "list"


class TestDiffCaps:
    def test_an_oversized_diff_is_dropped_not_kept(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("big.py", "".join(f"line {i}\n" for i in range(500)))
        c = _collect(tmp, since, limits=cs.Limits(max_diff_bytes=200))
        f = c.files[0]
        assert f.truncated and f.diff is None and c.truncated
        assert f.added == 500          # counts survive even when the text does not

    def test_an_oversized_diff_spills_and_reloads(self, workdir, tmp_path):
        tmp, _ = workdir
        spill = tmp_path / "spill"
        since = cs.open_window()
        write_file("big.py", "".join(f"line {i}\n" for i in range(500)))
        c = _collect(tmp, since, limits=cs.Limits(max_diff_bytes=200), spill_dir=spill)
        f = c.files[0]
        assert f.diff_ref and (spill / f.diff_ref).is_file()
        assert "line 499" in cs.load_diff(f, spill)

    def test_the_round_budget_stops_capture(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        for i in range(3):
            write_file(f"f{i}.py", "".join(f"line {j}\n" for j in range(100)))
        c = _collect(tmp, since, limits=cs.Limits(max_total_bytes=300))
        assert c.truncated
        assert sum(1 for f in c.files if f.diff is None) >= 2


class TestRender:
    def test_count_tier_is_one_line(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "x\n")
        c = _collect(tmp, since)
        assert cs.render_text(c, tier="count") == ["1 file changed, +1 -0"]

    def test_list_tier_is_one_row_per_file(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "x\n")
        write_file("b.py", "y\n")
        lines = cs.render_text(_collect(tmp, since), tier="list")
        assert len(lines) == 3
        assert lines[1].strip().startswith("+ a.py")

    def test_inline_tier_includes_the_diff(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "hello\n")
        lines = cs.render_text(_collect(tmp, since), tier="inline")
        assert any("+hello" in l for l in lines)

    def test_a_foreign_edit_is_called_out(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "mine\n")
        (tmp / "a.py").write_text("theirs\n")
        lines = cs.render_text(_collect(tmp, since), tier="list")
        assert any("outside any agent" in l for l in lines)


class TestSerialisation:
    def test_round_trip(self, workdir):
        tmp, _ = workdir
        since = cs.open_window()
        write_file("a.py", "x\n")
        c = _collect(tmp, since)
        back = cs.from_json(cs.to_json(c))
        assert back.headline() == c.headline()
        assert back.tier == c.tier
        assert back.files[0].diff == c.files[0].diff

    def test_unknown_fields_from_a_newer_agent_are_dropped(self):
        c = cs.from_json({"files": [{"path": "a.py", "added": 1, "invented": 9}]})
        assert c.files[0].path == "a.py" and c.files[0].added == 1

    def test_a_junk_payload_loads_as_empty(self):
        assert cs.from_json({}).file_count == 0
        assert cs.from_json({"files": [None, "x"]}).file_count == 0


class TestLimitsFromConfig:
    def test_no_section_gives_defaults(self):
        assert cs.limits_from_config(Config()) == cs.Limits()

    def test_values_are_read_and_junk_ignored(self):
        class _Section:
            inline_max_files = 9
            inline_max_lines = "nonsense"

        class _Cfg:
            class ui:
                changeset = _Section()

        lim = cs.limits_from_config(_Cfg())
        assert lim.inline_max_files == 9
        assert lim.inline_max_lines == cs.Limits().inline_max_lines
