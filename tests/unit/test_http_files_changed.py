"""A turn says which files it changed.

The diff viewer, /api/diff and the per-turn file list all existed — reachable
only from the condensed Q/A view, so a live turn never said what it wrote.
"""
import inspect
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
BLOCK = APP_JS[APP_JS.index("const MUTATING_TOOLS = ["):APP_JS.index("function toolCall(")]


class TestWhichToolsCount:
    def test_it_watches_the_same_tools_the_agent_tracks(self):
        """agent/core/agent.py keeps the authoritative list for the QA log."""
        import agent.core.agent as core

        src = inspect.getsource(core)
        i = src.index('if name in ("write_file", "patch_file", "edit_file")')
        for tool in ("write_file", "patch_file", "edit_file"):
            assert "'%s'" % tool in BLOCK, tool
        assert i > 0

    def test_it_reads_the_same_argument_shapes(self):
        """edit_file carries chunks; the other two carry a path."""
        assert "parsed.chunks || []" in BLOCK
        assert "parsed.path ? [parsed.path] : []" in BLOCK

    def test_unparseable_arguments_are_skipped(self):
        i = BLOCK.index("function toolPaths(")
        assert "catch (e) { return []; }" in BLOCK[i:i + 400]


class TestStrip:
    def test_a_file_is_listed_once(self):
        i = BLOCK.index("function noteFiles(")
        assert "turn.files.indexOf(p) < 0" in BLOCK[i:i + 300]

    def test_it_is_mounted_when_the_turn_ends(self):
        i = APP_JS.index("function endTurn()")
        assert "if (t.files.length) mount(filesStrip(t.files));" in APP_JS[i:i + 1400]

    def test_names_reuse_the_existing_diff_viewer(self):
        i = BLOCK.index("function filesStrip(")
        assert "toggleDiff(wrap, b.dataset.file)" in BLOCK[i:i + 900]

    def test_the_count_is_worded_for_one_file(self):
        i = BLOCK.index("function filesStrip(")
        assert "' file changed' : ' files changed'" in BLOCK[i:i + 500]

    def test_the_file_names_are_escaped(self):
        """A path is model-supplied text going into innerHTML."""
        i = BLOCK.index("function filesStrip(")
        body = BLOCK[i:i + 900]
        assert "esc(f)" in body

    def test_it_is_styled(self):
        assert ".files-changed {" in APP_CSS
        assert ".files-changed .fc-file.open {" in APP_CSS


class TestBothPaths:
    def test_live_and_replayed_turns_both_collect(self):
        for fn, end in (("function toolCall(", "function toolResult("),
                        ("function replayToolCall(", "function replayToolResult(")):
            body = APP_JS[APP_JS.index(fn):APP_JS.index(end)]
            assert "noteFiles(name, argsFull);" in body, fn

    def test_the_turn_carries_the_list(self):
        i = APP_JS.index("function beginTurn()")
        assert "files: []" in APP_JS[i:i + 800]
