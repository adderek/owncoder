"""Attachments can start the way they usually do: a paste or a drop.

/api/upload already existed; only the 📎 picker reached it.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
BLOCK = APP_JS[APP_JS.index("async function uploadFiles("):
               APP_JS.index("// ── Message history")]


class TestPaste:
    def test_pasted_files_are_uploaded(self):
        assert "input.addEventListener('paste'" in BLOCK
        assert "clipboardData" in BLOCK and "uploadFiles(" in BLOCK

    def test_a_text_paste_is_left_alone(self):
        """preventDefault only after we know there are files."""
        i = BLOCK.index("input.addEventListener('paste'")
        body = BLOCK[i:i + 500]
        assert body.index("if (!files.length) return;") < body.index("preventDefault()")

    def test_a_nameless_blob_gets_a_readable_name(self):
        """The path is what the agent will be told to open."""
        i = BLOCK.index("function pastedName(")
        body = BLOCK[i:i + 400]
        assert "'paste-'" in body and "file.type.split('/')[1]" in body


class TestDrop:
    def test_the_chat_column_accepts_a_drop(self):
        for ev in ("dragover", "dragleave", "drop"):
            assert "centerEl.addEventListener('%s'" % ev in BLOCK, ev

    def test_a_backlog_reorder_drag_is_not_a_file_drop(self):
        """The backlog rows run their own HTML5 drag; it carries no files."""
        i = BLOCK.index("function draggingFiles(")
        assert "indexOf('Files')" in BLOCK[i:i + 300]
        for ev in ("dragover", "drop"):
            j = BLOCK.index("centerEl.addEventListener('%s'" % ev)
            assert "if (!draggingFiles(e)) return;" in BLOCK[j:j + 200], ev

    def test_the_target_is_visible_while_dragging(self):
        assert "#center.dropping::after" in APP_CSS
        assert 'content: "drop to attach"' in APP_CSS

    def test_the_overlay_does_not_eat_the_drop(self):
        i = APP_CSS.index("#center.dropping::after")
        assert "pointer-events: none" in APP_CSS[i:i + 500]


class TestSharedPath:
    def test_all_three_routes_use_one_upload_loop(self):
        assert BLOCK.count("uploadFiles(") >= 3     # picker, paste, drop

    def test_one_bad_file_does_not_abort_the_batch(self):
        i = BLOCK.index("async function uploadFiles(")
        body = BLOCK[i:i + 400]
        assert "try {" in body and "catch (err)" in body
