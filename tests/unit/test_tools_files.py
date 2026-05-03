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
        assert "warning" in r


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
