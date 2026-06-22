"""Unit tests for agent/tools/edit_file.py — anchored-chunk edit tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent.config import Config
from agent.tools import load_all_tools, _registry, _schemas
from agent.tools.edit_file import edit_file, _register_edit_file
from agent.tools.files import _undo_stack, setup as files_setup, write_file
from agent.tools.rules import load_rules, get_rules, Rules, EditConfig, RulesConfig


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _setup(tmp_path):
    """Isolated working dir, rules, and undo stack for each test."""
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    _undo_stack.clear()

    # Load rules so get_rules() returns sane defaults pointing at tmp_path
    load_rules(str(tmp_path))
    _register_edit_file()
    yield
    _undo_stack.clear()


@pytest.fixture
def work(tmp_path):
    return tmp_path


# ── Happy path ─────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_single_chunk(self, work):
        (work / "f.py").write_text("alpha\nbeta\ngamma\n")
        r = edit_file([{"path": "f.py", "anchor": "beta", "replacement": "BETA"}])
        assert r.get("ok") is True
        assert (work / "f.py").read_text() == "alpha\nBETA\ngamma\n"

    def test_multi_chunk_same_file(self, work):
        (work / "f.py").write_text("a\nb\nc\nd\n")
        r = edit_file(
            [
                {"path": "f.py", "anchor": "a\n", "replacement": "AAA\n"},
                {"path": "f.py", "anchor": "d\n", "replacement": "DDD\n"},
            ]
        )
        assert r.get("ok") is True
        text = (work / "f.py").read_text()
        assert "AAA" in text and "DDD" in text

    def test_multi_chunk_multi_file(self, work):
        (work / "x.py").write_text("foo\n")
        (work / "y.py").write_text("bar\n")
        r = edit_file(
            [
                {"path": "x.py", "anchor": "foo", "replacement": "FOO"},
                {"path": "y.py", "anchor": "bar", "replacement": "BAR"},
            ]
        )
        assert r.get("ok") is True
        assert "FOO" in (work / "x.py").read_text()
        assert "BAR" in (work / "y.py").read_text()

    def test_delete_anchor_empty_replacement(self, work):
        (work / "f.py").write_text("keep\ndelete me\nalso keep\n")
        r = edit_file([{"path": "f.py", "anchor": "delete me\n", "replacement": ""}])
        assert r.get("ok") is True
        text = (work / "f.py").read_text()
        assert "delete me" not in text
        assert "keep" in text


# ── Anchor not found ───────────────────────────────────────────────────────


class TestAnchorNotFound:
    def test_anchor_not_found_returns_error(self, work):
        (work / "f.py").write_text("hello world\n")
        r = edit_file([{"path": "f.py", "anchor": "not_in_file", "replacement": "x"}])
        assert "error" in r
        assert any(e["kind"] == "anchor_not_found" for e in r["errors"])

    def test_no_write_on_anchor_not_found(self, work):
        original = "hello world\n"
        (work / "f.py").write_text(original)
        edit_file([{"path": "f.py", "anchor": "not_in_file", "replacement": "x"}])
        assert (work / "f.py").read_text() == original


# ── Anchor ambiguous ───────────────────────────────────────────────────────


class TestAnchorAmbiguous:
    def test_ambiguous_without_range_hint(self, work):
        (work / "f.py").write_text("foo\nfoo\n")
        r = edit_file([{"path": "f.py", "anchor": "foo", "replacement": "bar"}])
        assert "error" in r
        errors = r["errors"]
        assert any(e["kind"] == "anchor_ambiguous" for e in errors)

    def test_ambiguous_includes_candidates(self, work):
        (work / "f.py").write_text("foo\nfoo\n")
        r = edit_file([{"path": "f.py", "anchor": "foo", "replacement": "bar"}])
        err = next(e for e in r["errors"] if e["kind"] == "anchor_ambiguous")
        assert "candidates" in err
        assert len(err["candidates"]) >= 2

    def test_ambiguous_no_write(self, work):
        original = "foo\nfoo\n"
        (work / "f.py").write_text(original)
        edit_file([{"path": "f.py", "anchor": "foo", "replacement": "bar"}])
        assert (work / "f.py").read_text() == original

    def test_range_hint_disambiguates(self, work):
        (work / "f.py").write_text("foo\nother\nfoo\n")
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "foo",
                    "replacement": "FIRST",
                    "range_hint": [1, 1],
                }
            ]
        )
        assert r.get("ok") is True
        text = (work / "f.py").read_text()
        assert text.startswith("FIRST")
        # second 'foo' untouched
        assert text.count("foo") == 1


# ── anchor_sha256 mismatch ─────────────────────────────────────────────────


class TestAnchorShaMismatch:
    def test_wrong_sha_returns_error(self, work):
        (work / "f.py").write_text("correct anchor\n")
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "correct anchor",
                    "replacement": "new",
                    "anchor_sha256": "deadbeef" * 8,
                }
            ]
        )
        assert "error" in r
        assert any(e["kind"] == "anchor_sha_mismatch" for e in r["errors"])

    def test_wrong_sha_no_write(self, work):
        original = "correct anchor\n"
        (work / "f.py").write_text(original)
        edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "correct anchor",
                    "replacement": "new",
                    "anchor_sha256": "deadbeef" * 8,
                }
            ]
        )
        assert (work / "f.py").read_text() == original

    def test_correct_sha_succeeds(self, work):
        anchor = "correct anchor"
        sha = hashlib.sha256(anchor.encode()).hexdigest()
        (work / "f.py").write_text(anchor + "\n")
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": anchor,
                    "replacement": "new",
                    "anchor_sha256": sha,
                }
            ]
        )
        assert r.get("ok") is True


# ── expect_removed / expect_added outside tolerance ───────────────────────


class TestDeltaTolerance:
    def test_expect_removed_mismatch(self, work):
        (work / "f.py").write_text("line1\nline2\n")
        # anchor is 1 line but we claim 5
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "line1",
                    "replacement": "x",
                    "expect_removed": 5,
                }
            ]
        )
        assert "error" in r
        assert any(e["kind"] == "delta_exceeds_tolerance" for e in r["errors"])

    def test_expect_added_mismatch(self, work):
        (work / "f.py").write_text("line1\nline2\n")
        # replacement is 1 line but we claim 10
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "line1",
                    "replacement": "x",
                    "expect_added": 10,
                }
            ]
        )
        assert "error" in r
        assert any(e["kind"] == "delta_exceeds_tolerance" for e in r["errors"])

    def test_within_tolerance_succeeds(self, work):
        (work / "f.py").write_text("line1\nline2\n")
        # anchor=1 line, expect_removed=2 → within default tolerance of 2
        r = edit_file(
            [
                {
                    "path": "f.py",
                    "anchor": "line1",
                    "replacement": "x",
                    "expect_removed": 2,
                }
            ]
        )
        assert r.get("ok") is True


# ── max_chunk_lines exceeded ───────────────────────────────────────────────


class TestChunkTooLarge:
    def test_chunk_too_large_returns_error(self, work):
        big_lines = "\n".join(f"line{i}" for i in range(300))
        (work / "f.py").write_text(big_lines + "\n")
        # anchor is 300 lines, default max_chunk_lines=200
        r = edit_file([{"path": "f.py", "anchor": big_lines, "replacement": "small"}])
        assert "error" in r
        assert any(e["kind"] == "chunk_too_large" for e in r["errors"])

    def test_chunk_too_large_no_write(self, work):
        big_lines = "\n".join(f"line{i}" for i in range(300))
        original = big_lines + "\n"
        (work / "f.py").write_text(original)
        edit_file([{"path": "f.py", "anchor": big_lines, "replacement": "small"}])
        assert (work / "f.py").read_text() == original


# ── max_file_fraction exceeded (prevents whole-file replace) ──────────────


class TestFractionExceeded:
    def test_fraction_exceeded_prevents_whole_file_replace(self, work, tmp_path):
        # 40 lines — fraction check kicks in at >=20 lines
        lines = "\n".join(f"line{i}" for i in range(40))
        (work / "f.py").write_text(lines + "\n")
        # anchor spans all 40 lines → >50% → rejected
        r = edit_file([{"path": "f.py", "anchor": lines, "replacement": "tiny"}])
        assert "error" in r
        assert any(e["kind"] == "fraction_exceeded" for e in r["errors"])

    def test_small_anchor_fraction_ok(self, work):
        lines = "\n".join(f"line{i}" for i in range(40))
        (work / "f.py").write_text(lines + "\n")
        r = edit_file([{"path": "f.py", "anchor": "line0", "replacement": "LINE0"}])
        assert r.get("ok") is True


# ── Atomic rollback ────────────────────────────────────────────────────────


class TestAtomicRollback:
    def test_second_chunk_fails_first_not_written(self, work):
        (work / "f.py").write_text("good anchor\nbad section\n")
        original = (work / "f.py").read_text()
        r = edit_file(
            [
                {"path": "f.py", "anchor": "good anchor", "replacement": "CHANGED"},
                {"path": "f.py", "anchor": "NOT IN FILE", "replacement": "x"},
            ]
        )
        assert "error" in r
        # atomic rollback: nothing written
        assert (work / "f.py").read_text() == original

    def test_rollback_across_two_files(self, work):
        (work / "a.py").write_text("alpha\n")
        (work / "b.py").write_text("beta\n")
        orig_a = (work / "a.py").read_text()
        orig_b = (work / "b.py").read_text()
        r = edit_file(
            [
                {"path": "a.py", "anchor": "alpha", "replacement": "ALPHA"},
                {"path": "b.py", "anchor": "NOT THERE", "replacement": "x"},
            ]
        )
        assert "error" in r
        assert (work / "a.py").read_text() == orig_a
        assert (work / "b.py").read_text() == orig_b


# ── match="model" path ────────────────────────────────────────────────────


class TestMatchModel:
    def test_match_model_exposes_knob(self, work, tmp_path):
        """When edit.match='model' the schema should include a 'match' property."""
        from agent.tools.rules import _rules, RulesConfig, Rules, EditConfig
        import agent.tools.rules as rules_mod

        ec = EditConfig(match="model")
        rc = RulesConfig()
        rc.edit = ec
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(Rules(config=rc))
        try:
            from agent.tools.edit_file import _build_schema

            schema = _build_schema()
            assert "match_mode" in schema["parameters"]["properties"]
        finally:
            rules_mod.set_rules(old_rules)

    def test_match_loose_finds_drifted_anchor(self, work, tmp_path):
        """When match='loose' a whitespace-drifted anchor still matches."""
        from agent.tools.rules import _rules, RulesConfig, Rules, EditConfig
        import agent.tools.rules as rules_mod

        (work / "f.py").write_text("foo  bar\n")
        ec = EditConfig(match="loose")
        rc = RulesConfig()
        rc.edit = ec
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(Rules(config=rc))
        try:
            r = edit_file([{"path": "f.py", "anchor": "foo bar", "replacement": "baz"}])
            assert r.get("ok") is True
        finally:
            rules_mod.set_rules(old_rules)


# ── on_chunk_fail="skip" ──────────────────────────────────────────────────


class TestOnChunkFailSkip:
    def test_skip_applies_good_chunks_reports_failed(self, work, tmp_path):
        import agent.tools.rules as rules_mod
        from agent.tools.rules import RulesConfig, Rules, EditConfig

        (work / "f.py").write_text("good\nbad section here\n")
        ec = EditConfig(on_chunk_fail="skip")
        rc = RulesConfig()
        rc.edit = ec
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(Rules(config=rc))
        try:
            r = edit_file(
                [
                    {"path": "f.py", "anchor": "good", "replacement": "GOOD"},
                    {"path": "f.py", "anchor": "NOT IN FILE", "replacement": "x"},
                ]
            )
        finally:
            rules_mod.set_rules(old_rules)
        # Good chunk applied
        assert "GOOD" in (work / "f.py").read_text()
        # Failed chunk reported
        assert "skipped" in r or r.get("outcome") == "skip_partial"

    def test_wide_chunk_overlapping_nonadjacent_detected(self, work):
        # A wide chunk A spans lines 1-5. B (line 3) overlaps A but ends before
        # C (line 5) starts; C also overlaps A. The old adjacent-pair check only
        # compared C against B (no overlap) and missed C-vs-A, applying A and C
        # with overlapping ranges. Both B and C must be flagged as overlaps.
        import agent.tools.rules as rules_mod
        from agent.tools.rules import RulesConfig, Rules, EditConfig

        (work / "f.py").write_text("AAAAA\nmid1\nBBBBB\nmid2\nCCCCC\ntail\n")
        ec = EditConfig(on_chunk_fail="skip")
        rc = RulesConfig()
        rc.edit = ec
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(Rules(config=rc))
        try:
            r = edit_file([
                {"path": "f.py", "anchor": "AAAAA\nmid1\nBBBBB\nmid2\nCCCCC", "replacement": "ZZZ"},
                {"path": "f.py", "anchor": "BBBBB", "replacement": "Q"},
                {"path": "f.py", "anchor": "CCCCC", "replacement": "W"},
            ])
        finally:
            rules_mod.set_rules(old_rules)

        skipped = r.get("skipped", [])
        overlap_idx = {s["chunk_index"] for s in skipped if s.get("kind") == "chunks_overlap"}
        assert overlap_idx == {1, 2}, r
        # Only the wide chunk applied; file is not corrupted by double application.
        assert (work / "f.py").read_text() == "ZZZ\ntail\n"

    def test_skip_mode_exposed_in_schema_when_configured(self, work):
        import agent.tools.rules as rules_mod
        from agent.tools.rules import RulesConfig, Rules, EditConfig
        from agent.tools.edit_file import _build_schema

        ec = EditConfig(on_chunk_fail="model")
        rc = RulesConfig()
        rc.edit = ec
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(Rules(config=rc))
        try:
            schema = _build_schema()
            assert "on_chunk_fail" in schema["parameters"]["properties"]
        finally:
            rules_mod.set_rules(old_rules)


# ── Readonly / ignored paths ───────────────────────────────────────────────


class TestReadonlyIgnored:
    def test_readonly_path_rejected(self, work, tmp_path):
        import agent.tools.rules as rules_mod
        from agent.tools.rules import RulesConfig, Rules, ReadonlyMatcher

        (work / "locked.py").write_text("secret\n")
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(
            Rules(
                readonly=ReadonlyMatcher(["locked.py"]),
                config=old_rules.config,
            )
        )
        try:
            r = edit_file(
                [{"path": "locked.py", "anchor": "secret", "replacement": "x"}]
            )
        finally:
            rules_mod.set_rules(old_rules)
        assert "error" in r
        assert any(e["kind"] == "readonly" for e in r["errors"])

    def test_ignored_path_rejected(self, work):
        import agent.tools.rules as rules_mod
        from agent.tools.rules import RulesConfig, Rules, PathMatcher

        (work / "hidden.py").write_text("content\n")
        old_rules = rules_mod.get_rules()
        rules_mod.set_rules(
            Rules(
                ignore=PathMatcher(["hidden.py"]),
                config=old_rules.config,
            )
        )
        try:
            r = edit_file(
                [{"path": "hidden.py", "anchor": "content", "replacement": "x"}]
            )
        finally:
            rules_mod.set_rules(old_rules)
        assert "error" in r

    def test_file_not_found(self, work):
        r = edit_file(
            [{"path": "does_not_exist.py", "anchor": "x", "replacement": "y"}]
        )
        assert "error" in r
        assert any(e["kind"] == "file_not_found" for e in r["errors"])


# ── Auto-unescape double-encoded anchors ──────────────────────────────────


class TestAutoUnescape:
    def test_double_encoded_newline_matches(self, work):
        (work / "f.py").write_text('def foo():\n    return "bar"\n')
        # Model emitted \\n (literal backslash-n) instead of real newline
        r = edit_file([{"path": "f.py", "anchor": 'def foo():\\n    return "bar"', "replacement": 'def foo():\\n    return "baz"'}])
        assert r.get("ok") is True
        assert 'return "baz"' in (work / "f.py").read_text()

    def test_double_encoded_sets_auto_unescaped_flag(self, work):
        (work / "f.py").write_text('def foo():\n    return "bar"\n')
        r = edit_file([{"path": "f.py", "anchor": 'def foo():\\n    return "bar"', "replacement": 'def foo():\\n    return "baz"'}])
        assert r.get("ok") is True
        assert r["applied"][0].get("auto_unescaped") is True

    def test_no_false_positive_when_anchor_has_real_newline(self, work):
        (work / "f.py").write_text("line1\nline2\n")
        # Anchor with real newline — no double-encoding trigger
        r = edit_file([{"path": "f.py", "anchor": "line1\nline2", "replacement": "LINE1\nLINE2"}])
        assert r.get("ok") is True
        assert r["applied"][0].get("auto_unescaped") is None

    def test_literal_backslash_n_in_file_not_clobbered(self, work):
        # File actually contains literal \\n (e.g. raw string); anchor also has \\n.
        # _looks_double_escaped = False because anchor has real \\n chars but not the
        # trigger pattern (anchor has literal backslash-n but no \n — wait, that IS the trigger).
        # This test checks: if file has literal \\n AND anchor has literal \\n, exact match wins.
        (work / "f.py").write_text("line1\\nline2\n")  # actual \\ in file
        r = edit_file([{"path": "f.py", "anchor": "line1\\nline2", "replacement": "REPLACED"}])
        assert r.get("ok") is True
        # No auto-unescape needed — exact match found
        assert r["applied"][0].get("auto_unescaped") is None


# ── Audit log record ──────────────────────────────────────────────────────


class TestAuditLog:
    def test_audit_record_appended_on_success(self, work):
        (work / "f.py").write_text("hello\n")
        edit_file([{"path": "f.py", "anchor": "hello", "replacement": "world"}])
        log = work / ".agent" / "edit_stats.jsonl"
        assert log.exists(), "edit_stats.jsonl should be written"
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(r["tool"] == "edit_file" for r in records)

    def test_audit_record_appended_on_error(self, work):
        (work / "f.py").write_text("hello\n")
        edit_file([{"path": "f.py", "anchor": "NOT_HERE", "replacement": "x"}])
        log = work / ".agent" / "edit_stats.jsonl"
        assert log.exists()
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert any(
            r.get("outcome") in ("atomic_rollback", "skip_partial")
            or r["tool"] == "edit_file"
            for r in records
        )


class TestAdjustReplacementIndent:
    """Continuation-line indent fix-up uses the anchor's line-local indentation,
    not the whole-file prefix (which used to inject the file's leading newlines)."""

    def test_top_of_file_blank_lines_no_newline_injection(self):
        from agent.tools.edit_file.validator import _adjust_replacement_indent
        orig = "\n\n    class Foo:\n        pass\n"
        s = orig.index("class Foo:")
        out, adjusted = _adjust_replacement_indent(
            "class Foo:\n    def bar(): pass", s, orig, "class Foo:")
        assert adjusted is True
        assert out == "class Foo:\n        def bar(): pass"
        assert "\n\n" not in out

    def test_mid_file_indented_anchor_adjusts(self):
        from agent.tools.edit_file.validator import _adjust_replacement_indent
        orig = "x = 1\ny = 2\n    class Foo:\n        pass\n"
        s = orig.index("class Foo:")
        out, adjusted = _adjust_replacement_indent(
            "class Foo:\n    def bar(): pass", s, orig, "class Foo:")
        assert adjusted is True
        assert out == "class Foo:\n        def bar(): pass"

    def test_column_zero_anchor_not_adjusted(self):
        from agent.tools.edit_file.validator import _adjust_replacement_indent
        orig = "class Foo:\n    pass\n"
        out, adjusted = _adjust_replacement_indent(
            "class Foo:\n    def bar(): pass", 0, orig, "class Foo:")
        assert adjusted is False
        assert out == "class Foo:\n    def bar(): pass"
