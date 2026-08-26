"""Unit tests for agent/tools/files.py — pure filesystem operations, no LLM."""
from __future__ import annotations

import pytest
from agent.config import Config
from agent.tools import load_all_tools
from agent.tools.files import read_file, write_file, patch_file, replace_text, list_files, setup as files_setup, _undo_stack
from agent.tools.files.undo import undo_file


@pytest.fixture(autouse=True)
def _setup_tools(tmp_path):
    """Configure tools to use tmp_path as working dir."""
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = str(tmp_path / ".agent")
    files_setup(cfg)
    _undo_stack.clear()
    yield


@pytest.fixture
def work(tmp_path):
    return tmp_path


class TestWriteThenRead:
    def test_basic_roundtrip(self, work):
        result = write_file("hello.txt", "world\n")
        assert "ok" in result
        r = read_file("hello.txt")
        assert "world" in r["content"]

    def test_creates_parent_dirs(self, work):
        result = write_file("sub/dir/file.txt", "nested\n")
        assert "ok" in result
        r = read_file("sub/dir/file.txt")
        assert "nested" in r["content"]

    def test_overwrite_existing(self, work):
        write_file("f.txt", "v1")
        write_file("f.txt", "v2")
        r = read_file("f.txt")
        assert "v2" in r["content"]


class TestReadFile:
    def test_missing_file(self, work):
        r = read_file("nonexistent.txt")
        assert "error" in r

    def test_line_range(self, work):
        content = "\n".join(f"line {i}" for i in range(1, 11))
        write_file("lines.txt", content)
        r = read_file("lines.txt", start_line=3, end_line=5)
        assert "line 3" in r["content"]
        assert "line 5" in r["content"]
        assert "line 6" not in r["content"]

    def test_large_file_hint(self, work):
        content = "\n".join(f"line {i}" for i in range(1, 600))
        write_file("big.txt", content)
        r = read_file("big.txt")
        assert "lines" in r.get("content", "") and "showing lines" in r.get("content", "")


class TestPathEscape:
    def test_absolute_path_outside_working_dir(self, work):
        with pytest.raises(ValueError, match="escapes working directory"):
            read_file("/etc/passwd")

    def test_relative_escape(self, work):
        with pytest.raises(ValueError, match="escapes working directory"):
            read_file("../../etc/passwd")


class TestReplaceText:
    def test_basic_replace(self, work):
        write_file("f.py", "foo = 1\nbar = 2\n")
        r = replace_text("f.py", "foo = 1", "foo = 42")
        assert "ok" in r
        content = read_file("f.py")["content"]
        assert "42" in content

    def test_search_not_found(self, work):
        write_file("f.py", "hello\n")
        r = replace_text("f.py", "not_here", "replacement")
        assert "error" in r

    def test_missing_file(self, work):
        r = replace_text("missing.py", "x", "y")
        assert "error" in r


class TestListFiles:
    def test_basic_list(self, work):
        write_file("a.py", "x")
        write_file("b.py", "y")
        r = list_files(".")
        assert r["count"] >= 2
        paths = [f["path"] for f in r["files"]]
        assert "a.py" in paths
        assert "b.py" in paths

    def test_not_a_directory(self, work):
        write_file("f.txt", "x")
        r = list_files("f.txt")
        assert "error" in r

    def test_cap_returns_summary_not_paths(self, work):
        for i in range(20):
            write_file(f"src/mod_{i}.py", "x")
            write_file(f"docs/page_{i}.md", "x")
        r = list_files(".", max_results=5)
        assert r.get("truncated") is True
        assert r["total"] >= 40
        assert "files" not in r  # path list is suppressed on overflow
        dirs = {d["dir"]: d["count"] for d in r["by_top_dir"]}
        assert dirs.get("src", 0) == 20
        assert dirs.get("docs", 0) == 20
        assert "hint" in r

    def test_cap_not_triggered_when_under(self, work):
        write_file("a.py", "x")
        write_file("b.py", "y")
        r = list_files(".", max_results=10)
        assert r.get("truncated") is not True
        assert "files" in r


class TestUndoFile:
    def test_undo_after_write(self, work):
        write_file("undo_me.txt", "original\n")
        write_file("undo_me.txt", "changed\n")
        r = undo_file("undo_me.txt")
        assert r == {"ok": "undo_me.txt"}
        content = (work / "undo_me.txt").read_text()
        assert content == "original\n"

    def test_undo_clears_snapshot(self, work):
        write_file("x.txt", "v1\n")
        write_file("x.txt", "v2\n")
        undo_file("x.txt")
        # second undo has no snapshot
        r = undo_file("x.txt")
        assert "error" in r
        assert "No undo snapshot" in r["error"]

    def test_undo_no_snapshot(self):
        r = undo_file("nonexistent.txt")
        assert "error" in r

    def test_undo_after_edit_file(self, work):
        from agent.tools.edit_file.core import edit_file
        write_file("target.py", "def foo():\n    pass\n")
        edit_file(chunks=[{"path": "target.py", "anchor": "    pass", "replacement": "    return 1"}])
        r = undo_file("target.py")
        assert r == {"ok": "target.py"}
        content = (work / "target.py").read_text()
        assert "pass" in content
        assert "return 1" not in content

    def test_failed_undo_keeps_snapshot(self, work, monkeypatch):
        # If the revert write fails, the snapshot must survive so the undo can be
        # retried (the old code popped before writing and lost it on failure).
        from agent.tools.files import paths as _paths
        write_file("keep.txt", "v1\n")
        write_file("keep.txt", "v2\n")

        def _boom(self, *a, **k):
            raise OSError("disk full")

        monkeypatch.setattr("pathlib.Path.write_text", _boom)
        r = undo_file("keep.txt")
        assert "error" in r
        # Snapshot retained for a retry.
        assert "keep.txt" in _paths._undo_stack
        monkeypatch.undo()
        r2 = undo_file("keep.txt")
        assert r2 == {"ok": "keep.txt"}
        assert (work / "keep.txt").read_text() == "v1\n"


class TestWriteFileGuards:
    def test_dry_run_no_write(self, work):
        from agent.tools.rules import get_rules
        get_rules().config.dry_run = True
        try:
            r = write_file("dryfile.txt", "data\n")
            assert r.get("dry_run") is True
            assert not (work / "dryfile.txt").exists()
        finally:
            get_rules().config.dry_run = False

    def test_size_limit_rejected(self, work):
        from agent.tools.rules import get_rules
        original_limit = get_rules().config.max_write_size
        get_rules().config.max_write_size = 10
        try:
            r = write_file("big.txt", "x" * 100)
            assert "error" in r
        finally:
            get_rules().config.max_write_size = original_limit


class TestTruncatedReadOutline:
    """A truncated read must hand back a map of the rest of the file, so the
    model jumps to the landmark instead of paging (session 20260826T202043_fed4:
    three reads — 1-200, 300-400, 400-450 — to find "// --- DOMKI ---")."""

    def _big_js(self) -> str:
        parts = ["// header\n" * 10]
        parts.append("// === LIGHTING ===\n")
        parts.append("function createLight() {\n  return 1;\n}\n")
        parts.append("const filler = 0;\n" * 500)
        parts.append("// --- DOMKI ---\n")
        parts.append("function createHouse(x, z) {\n  return x;\n}\n")
        return "".join(parts)

    def test_outline_lists_landmarks_beyond_the_window(self, work):
        (work / "scene.js").write_text(self._big_js())
        r = read_file("scene.js")
        content = r["content"]
        assert "outline of the whole file" in content
        assert "section DOMKI" in content
        assert "func createHouse" in content
        names = {e["name"] for e in r["metadata"]["outline"]}
        assert {"LIGHTING", "DOMKI", "createLight", "createHouse"} <= names
        # The landmark is past the served window — the line number is the point.
        domki = next(e for e in r["metadata"]["outline"] if e["name"] == "DOMKI")
        assert domki["line"] > 200

    def test_ranged_read_has_no_outline(self, work):
        (work / "scene.js").write_text(self._big_js())
        r = read_file("scene.js", start_line=1, end_line=20)
        assert "outline of the whole file" not in r["content"]

    def test_small_file_read_unchanged(self, work):
        (work / "small.py").write_text("def a():\n    return 1\n")
        r = read_file("small.py")
        assert "outline of the whole file" not in r["content"]
        assert "def a" in r["content"]


class TestOutline:
    def test_python_defs_and_classes(self):
        from agent.tools.files.outline import outline
        entries = outline("class Foo:\n    def bar(self):\n        pass\n")
        assert [(e["kind"], e["name"]) for e in entries] == [("class", "Foo"), ("def", "bar")]

    def test_js_arrow_and_function(self):
        from agent.tools.files.outline import outline
        src = "export const load = async () => {}\nfunction plain(a) {}\n"
        assert [e["name"] for e in outline(src)] == ["load", "plain"]

    def test_section_banners(self):
        from agent.tools.files.outline import outline
        src = "// === OSWIETLENIE ===\n# --- DOMKI ---\n"
        assert [(e["kind"], e["name"]) for e in outline(src)] == [
            ("section", "OSWIETLENIE"), ("section", "DOMKI")]

    def test_truncation_marker(self):
        from agent.tools.files.outline import outline
        src = "".join(f"def f{i}():\n    pass\n" for i in range(30))
        entries = outline(src, max_entries=5)
        assert len(entries) == 6
        assert entries[-1]["kind"] == "..."
        assert "25 more" in entries[-1]["name"]


class TestIndexCoverageFormatting:
    def test_root_files_are_not_reported_as_directories(self):
        from agent.cli.chat import _get_index_coverage, _format_index_coverage

        class _Store:
            def list_paths(self):
                return ["/w/collect.py", "/w/AGENTS.md", "/w/agent/core/turn.py"]

        coverage = _get_index_coverage(_Store(), "/w")
        assert coverage == {".": 2, "agent": 1}
        text = _format_index_coverage(coverage)
        assert "collect.py/" not in text
        assert "(repository root): 2 file(s)" in text
        assert "agent/: 1 file(s)" in text
        assert "3 indexed file(s)" in text

    def test_long_directory_list_is_capped(self):
        from agent.cli.chat import _format_index_coverage, _COVERAGE_MAX_DIRS

        coverage = {f"d{i}": 1 for i in range(_COVERAGE_MAX_DIRS + 5)}
        text = _format_index_coverage(coverage)
        assert text.count("file(s)") <= _COVERAGE_MAX_DIRS + 2
        assert "5 more directories" in text


class TestOutlineOnlyWall:
    """With tools.outline_only_after_reads set, the Nth unbounded read of one
    file stops serving content and hands back the map instead."""

    @pytest.fixture(autouse=True)
    def _wall(self, tmp_path):
        from agent.core.tool_hints import reset_tool_hints, tool_hints
        from agent.tools.files import setup as files_setup

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        cfg.tools.outline_only_after_reads = 2
        files_setup(cfg)
        reset_tool_hints()
        # read_file itself does not count; the hint layer does, as in production.
        self._count = lambda path: tool_hints("read_file", {"path": path}, {"content": ""})
        yield
        reset_tool_hints()

    def test_wall_serves_outline_only(self, work):
        (work / "m.py").write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n")
        for _ in range(2):
            r = read_file("m.py")
            assert "def alpha" in r["content"]
            self._count("m.py")
        r = read_file("m.py")
        assert r["metadata"]["outline_only"] is True
        assert "outline only" in r["content"]
        assert "find_symbol" in r["content"]
        assert "1: def alpha" in r["content"]
        assert "return 1" not in r["content"]

    def test_ranged_read_still_works_behind_the_wall(self, work):
        (work / "m.py").write_text("def alpha():\n    return 1\n")
        for _ in range(3):
            self._count("m.py")
        r = read_file("m.py", start_line=1, end_line=2)
        assert "return 1" in r["content"]
        assert not r["metadata"].get("outline_only")

    def test_other_files_unaffected(self, work):
        (work / "m.py").write_text("def alpha():\n    return 1\n")
        (work / "n.py").write_text("def gamma():\n    return 3\n")
        for _ in range(3):
            self._count("m.py")
        r = read_file("n.py")
        assert "return 3" in r["content"]

    def test_disabled_by_default(self, work):
        from agent.config import Config as _C
        cfg = _C()
        assert cfg.tools.outline_only_after_reads == 0
