"""expect_rev compare-and-swap on the mutating file tools (docs/TODO-2.md step 1)."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.core import checkpoint, revisions
from agent.tools.edit_file import edit_file
from agent.tools.files import (
    _undo_stack, patch_file, read_file, setup as files_setup, write_file,
)


def _cfg(tmp_path, mode="warn", handles=True) -> Config:
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    cfg.tools.revisions.mode = mode
    cfg.tools.revisions.handles = handles
    return cfg


@pytest.fixture
def warn(tmp_path):
    files_setup(_cfg(tmp_path, "warn"))
    _undo_stack.clear()
    yield tmp_path
    revisions.reset()


@pytest.fixture
def require(tmp_path):
    files_setup(_cfg(tmp_path, "require"))
    _undo_stack.clear()
    yield tmp_path
    revisions.reset()


@pytest.fixture
def off(tmp_path):
    files_setup(_cfg(tmp_path, "off"))
    _undo_stack.clear()
    yield tmp_path
    revisions.reset()


def _rev(path="f.py"):
    return read_file(path)["rev"]


class TestReadReturnsRev:
    def test_read_carries_rev_and_handle(self, warn):
        write_file("f.py", "x = 1\n")
        r = read_file("f.py")
        assert r["rev"] == "sha256:" + checkpoint.sha_text("x = 1\n")
        assert r["rev_handle"].startswith("r")

    def test_windowed_read_reports_whole_file_rev(self, warn):
        body = "".join(f"line {i}\n" for i in range(600))
        write_file("big.txt", body)
        whole = read_file("big.txt")["rev"]
        assert read_file("big.txt", start_line=5, end_line=9)["rev"] == whole
        assert whole == "sha256:" + checkpoint.sha_text(body)

    def test_no_rev_when_off(self, off):
        write_file("f.py", "x = 1\n")
        assert "rev" not in read_file("f.py")

    def test_no_handle_when_handles_disabled(self, tmp_path):
        files_setup(_cfg(tmp_path, "warn", handles=False))
        try:
            write_file("f.py", "x = 1\n")
            r = read_file("f.py")
            assert "rev" in r and "rev_handle" not in r
        finally:
            revisions.reset()

    def test_same_content_reuses_handle(self, warn):
        write_file("f.py", "x = 1\n")
        assert read_file("f.py")["rev_handle"] == read_file("f.py")["rev_handle"]


class TestWriteFilePin:
    def test_matching_rev_applies(self, warn):
        write_file("f.py", "x = 1\n")
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev=_rev())
        assert (warn / "f.py").read_text() == "x = 2\n"

    def test_short_prefix_accepted(self, warn):
        write_file("f.py", "x = 1\n")
        prefix = _rev().split(":")[1][:8]
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev=prefix)

    def test_handle_accepted(self, warn):
        write_file("f.py", "x = 1\n")
        handle = read_file("f.py")["rev_handle"]
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev=handle)

    def test_stale_rev_refuses_and_leaves_file(self, warn):
        write_file("f.py", "x = 1\n")
        stale = _rev()
        (warn / "f.py").write_text("someone else\n")   # the user, or another agent
        res = write_file("f.py", "x = 2\n", expect_rev=stale)
        assert res["kind"] == "rev_mismatch"
        assert (warn / "f.py").read_text() == "someone else\n"

    def test_rejection_names_both_revisions(self, warn):
        write_file("f.py", "x = 1\n")
        stale = _rev().split(":")[1]
        (warn / "f.py").write_text("moved\n")
        res = write_file("f.py", "x = 2\n", expect_rev=stale)
        current = checkpoint.sha_text("moved\n")
        assert stale[:8] in res["error"] and current[:8] in res["error"]
        assert res["current_rev"] == "sha256:" + current

    def test_rejection_names_the_actor(self, warn, monkeypatch):
        write_file("f.py", "x = 1\n")
        stale = _rev()
        monkeypatch.setattr(checkpoint, "_actor_id", "owncoder-4711")
        write_file("f.py", "theirs\n", expect_rev="any")
        res = write_file("f.py", "mine\n", expect_rev=stale)
        assert res["current_actor"] == "owncoder-4711"
        assert "owncoder-4711" in res["error"]

    def test_rejection_carries_a_usable_retry_rev(self, warn):
        write_file("f.py", "x = 1\n")
        stale = _rev()
        (warn / "f.py").write_text("moved\n")
        res = write_file("f.py", "x = 2\n", expect_rev=stale)
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev=res["current_rev"])

    def test_new_asserts_absence(self, warn):
        assert "ok" in write_file("fresh.py", "x = 1\n", expect_rev="new")
        res = write_file("fresh.py", "x = 2\n", expect_rev="new")
        assert res["kind"] == "rev_mismatch"
        assert (warn / "fresh.py").read_text() == "x = 1\n"

    def test_rev_on_missing_file_refuses(self, warn):
        res = write_file("gone.py", "x\n", expect_rev="a" * 64)
        assert res["kind"] == "rev_mismatch"
        assert not (warn / "gone.py").exists()

    def test_any_opts_out(self, warn):
        write_file("f.py", "x = 1\n")
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev="any")

    def test_no_pin_is_allowed_in_warn(self, warn):
        write_file("f.py", "x = 1\n")
        assert "ok" in write_file("f.py", "x = 2\n")


class TestMalformedPins:
    def test_unparseable_refuses(self, warn):
        write_file("f.py", "x = 1\n")
        assert write_file("f.py", "x = 2\n", expect_rev="latest")["kind"] == "rev_mismatch"

    def test_too_short_prefix_refuses(self, warn):
        write_file("f.py", "x = 1\n")
        prefix = _rev().split(":")[1][:4]
        res = write_file("f.py", "x = 2\n", expect_rev=prefix)
        assert res["kind"] == "rev_mismatch"
        assert "7" in res["error"]

    def test_unknown_handle_refuses_rather_than_falling_back(self, warn):
        write_file("f.py", "x = 1\n")
        res = write_file("f.py", "x = 2\n", expect_rev="r999")
        assert res["kind"] == "rev_mismatch"
        assert (warn / "f.py").read_text() == "x = 1\n"

    def test_handle_from_another_file_refuses(self, warn):
        write_file("a.py", "same\n")
        write_file("b.py", "same\n")
        handle = read_file("a.py")["rev_handle"]
        res = write_file("b.py", "changed\n", expect_rev=handle)
        assert res["kind"] == "rev_mismatch"
        assert "a.py" in res["error"]

    def test_reserved_prefixes_refuse_explicitly(self, warn):
        write_file("f.py", "x = 1\n")
        res = write_file("f.py", "x = 2\n", expect_rev="tree:abc123def")
        assert res["kind"] == "rev_mismatch"
        assert "not implemented" in res["error"]

    def test_handles_do_not_survive_a_new_session(self, warn, tmp_path):
        write_file("f.py", "x = 1\n")
        handle = read_file("f.py")["rev_handle"]
        files_setup(_cfg(tmp_path, "warn"))     # restart
        assert write_file("f.py", "x = 2\n", expect_rev=handle)["kind"] == "rev_mismatch"


class TestModes:
    def test_off_ignores_even_a_wrong_pin(self, off):
        write_file("f.py", "x = 1\n")
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev="deadbeefdeadbeef")

    def test_require_refuses_an_unpinned_write(self, require):
        write_file("f.py", "x = 1\n", expect_rev="new")
        res = write_file("f.py", "x = 2\n")
        assert res["kind"] == "rev_mismatch"
        assert (require / "f.py").read_text() == "x = 1\n"

    def test_require_rejection_quotes_the_rev_to_retry_with(self, require):
        write_file("f.py", "x = 1\n", expect_rev="new")
        res = write_file("f.py", "x = 2\n")
        assert "ok" in write_file("f.py", "x = 2\n", expect_rev=res["current_rev"])

    def test_require_demands_new_for_a_fresh_file(self, require):
        res = write_file("fresh.py", "x\n")
        assert res["kind"] == "rev_mismatch"
        assert '"new"' in res["error"]

    def test_wrong_pin_refuses_in_warn_too(self, warn):
        write_file("f.py", "x = 1\n")
        assert write_file("f.py", "x = 2\n", expect_rev="deadbeefdeadbeef")["kind"] == "rev_mismatch"


class TestEditFilePin:
    def test_matching_rev_applies(self, warn):
        write_file("f.py", "x = 1\n")
        res = edit_file(path="f.py", anchor="x = 1", replacement="x = 2",
                        expect_rev=_rev())
        assert res.get("ok")
        assert (warn / "f.py").read_text() == "x = 2\n"

    def test_stale_rev_refuses(self, warn):
        write_file("f.py", "x = 1\n")
        stale = _rev()
        (warn / "f.py").write_text("x = 1\n# theirs\n")
        res = edit_file(path="f.py", anchor="x = 1", replacement="x = 2",
                        expect_rev=stale)
        assert res["error"] == "atomic_rollback"
        assert res["errors"][0]["kind"] == "rev_mismatch"
        assert (warn / "f.py").read_text() == "x = 1\n# theirs\n"

    def test_top_level_rev_covers_every_chunk(self, warn):
        write_file("f.py", "a = 1\nb = 2\n")
        res = edit_file(chunks=[
            {"path": "f.py", "anchor": "a = 1", "replacement": "a = 9"},
            {"path": "f.py", "anchor": "b = 2", "replacement": "b = 9"},
        ], expect_rev=_rev())
        assert res.get("ok")
        assert (warn / "f.py").read_text() == "a = 9\nb = 9\n"

    def test_per_chunk_rev_wins_over_top_level(self, warn):
        write_file("f.py", "a = 1\n")
        res = edit_file(
            chunks=[{"path": "f.py", "anchor": "a = 1", "replacement": "a = 9",
                     "expect_rev": "deadbeefdeadbeef"}],
            expect_rev=_rev(),
        )
        assert res["errors"][0]["kind"] == "rev_mismatch"

    def test_one_stale_chunk_blocks_the_whole_call(self, warn):
        write_file("a.py", "a = 1\n")
        write_file("b.py", "b = 2\n")
        res = edit_file(chunks=[
            {"path": "a.py", "anchor": "a = 1", "replacement": "a = 9",
             "expect_rev": read_file("a.py")["rev"]},
            {"path": "b.py", "anchor": "b = 2", "replacement": "b = 9",
             "expect_rev": "deadbeefdeadbeef"},
        ])
        assert res["error"] == "atomic_rollback"
        assert (warn / "a.py").read_text() == "a = 1\n"

    def test_require_refuses_unpinned_chunk(self, require):
        write_file("f.py", "x = 1\n", expect_rev="new")
        res = edit_file(path="f.py", anchor="x = 1", replacement="x = 2")
        assert res["errors"][0]["kind"] == "rev_mismatch"
        assert (require / "f.py").read_text() == "x = 1\n"


class TestPatchFilePin:
    DIFF = (
        "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-one\n+two\n"
    )

    def test_matching_rev_applies(self, warn):
        write_file("f.txt", "one\n")
        res = patch_file("f.txt", self.DIFF, expect_rev=read_file("f.txt")["rev"])
        assert "ok" in res, res
        assert (warn / "f.txt").read_text() == "two\n"

    def test_stale_rev_refuses(self, warn):
        write_file("f.txt", "one\n")
        stale = read_file("f.txt")["rev"]
        (warn / "f.txt").write_text("one\ntheirs\n")
        res = patch_file("f.txt", self.DIFF, expect_rev=stale)
        assert res["kind"] == "rev_mismatch"
        assert (warn / "f.txt").read_text() == "one\ntheirs\n"


class TestTwoAgentsCannotLoseWork:
    def test_second_agent_is_refused_not_silently_overwritten(self, warn, monkeypatch):
        """Both agents read rev A; the slower one must not clobber the faster."""
        write_file("shared.txt", "base\n")
        rev_a = _rev("shared.txt")

        monkeypatch.setattr(checkpoint, "_actor_id", "owncoder-1")
        assert "ok" in write_file("shared.txt", "from agent one\n", expect_rev=rev_a)

        monkeypatch.setattr(checkpoint, "_actor_id", "owncoder-2")
        res = write_file("shared.txt", "from agent two\n", expect_rev=rev_a)
        assert res["kind"] == "rev_mismatch"
        assert res["current_actor"] == "owncoder-1"
        assert (warn / "shared.txt").read_text() == "from agent one\n"


class TestJournal:
    def test_opt_out_is_recorded(self, warn):
        write_file("f.py", "x = 1\n", expect_rev="new")
        write_file("f.py", "x = 2\n", expect_rev="any")
        assert [e["pinned"] for e in checkpoint.journal_entries()] == ["pinned", "any"]

    def test_nothing_recorded_when_off(self, off):
        write_file("f.py", "x = 1\n")
        assert checkpoint.journal_entries()[-1]["pinned"] is None
