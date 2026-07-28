"""The message box behaves like a prompt: ↑ recalls, drafts survive a reload."""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HIST = APP_JS[APP_JS.index("const HIST_KEY"):APP_JS.index("input.addEventListener('keydown'")]


class TestHistory:
    def test_it_is_persisted_and_capped(self):
        assert "localStorage.setItem(HIST_KEY" in HIST
        assert "const HIST_MAX = 100" in HIST
        assert "history.slice(-HIST_MAX)" in HIST

    def test_repeating_the_same_line_does_not_fill_the_ring(self):
        assert "text === history[history.length - 1]" in HIST

    def test_corrupt_storage_does_not_break_the_prompt(self):
        assert "if (!Array.isArray(history)) history = [];" in HIST

    def test_browsing_stashes_the_unsent_draft(self):
        assert "histDraft = input.value;" in HIST
        assert "histApply(histDraft);" in HIST

    def test_sending_records_the_line(self):
        i = APP_JS.index("async function send()")
        assert "histPush(text);" in APP_JS[i:i + 400]


class TestArrowsStillEditText:
    def test_browsing_only_starts_from_the_edge_of_the_text(self):
        """↑ inside a multi-line message must move the caret, not the history."""
        i = APP_JS.index("input.addEventListener('keydown'")
        body = APP_JS[i:APP_JS.index("input.addEventListener('input'")]
        assert "onFirstLine()" in body and "onLastLine()" in body
        assert "histMove(-1)" in body and "histMove(1)" in body

    def test_typing_leaves_the_history(self):
        i = APP_JS.index("input.addEventListener('input'")
        assert "histIdx = -1;" in APP_JS[i:i + 300]


class TestDraft:
    def test_it_is_restored_on_load(self):
        assert "localStorage.getItem(DRAFT_KEY)" in APP_JS

    def test_it_is_cleared_once_sent(self):
        i = APP_JS.index("async function send()")
        body = APP_JS[i:APP_JS.index("input.addEventListener('keydown'")]
        assert body.count("saveDraft();") >= 2      # /clear branch and the send path

    def test_a_failed_send_keeps_it(self):
        i = APP_JS.index("send failed (server unreachable?)")
        assert "saveDraft();" in APP_JS[i - 200:i]

    def test_an_empty_box_stores_nothing(self):
        i = HIST.index("function saveDraft()")
        assert "removeItem(DRAFT_KEY)" in HIST[i:i + 300]
