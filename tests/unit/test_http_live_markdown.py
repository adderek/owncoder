"""A streaming answer is readable while it streams.

Tokens landed as raw text and only became markdown when the turn ended, so a
long code answer was unformatted for exactly as long as it took to produce.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
BLOCK = APP_JS[APP_JS.index("let streamRaw = '';"):APP_JS.index("function endStream()")]


class TestRendering:
    def test_tokens_go_to_a_raw_buffer(self):
        """The rendered HTML cannot be appended to, and the fold keeps the raw."""
        i = APP_JS.index("if (ev.type === 'token')")
        body = APP_JS[i:i + 700]
        assert "streamRaw += ev.text;" in body
        assert "streamEl.textContent += ev.text;" not in body

    def test_an_unfinished_fence_is_closed_for_the_render_only(self):
        i = BLOCK.index("function balancedMd(")
        body = BLOCK[i:BLOCK.index("function renderStream()")]
        assert "fences % 2" in body
        # the buffer itself is never modified
        assert "streamRaw =" not in body

    def test_renders_are_throttled(self):
        """One render per burst of tokens, not one per token."""
        assert "const STREAM_RENDER_MS = 120;" in BLOCK
        i = BLOCK.index("function scheduleStreamRender()")
        assert "if (streamTimer) return;" in BLOCK[i:i + 200]

    def test_a_huge_answer_falls_back_to_text(self):
        assert "const STREAM_MD_LIMIT = 400000;" in BLOCK
        i = BLOCK.index("function renderStream()")
        assert "streamEl.textContent = streamRaw;" in BLOCK[i:i + 300]


class TestCleanup:
    def test_ending_a_stream_stops_the_timer(self):
        i = APP_JS.index("function endStream()")
        assert "stopStreamRender();" in APP_JS[i:i + 200]

    def test_ending_a_stream_renders_the_raw_markdown(self):
        i = APP_JS.index("function endStream()")
        assert "streamRaw || streamEl.textContent" in APP_JS[i:i + 500]

    def test_the_buffer_does_not_leak_into_the_next_stream(self):
        for fn in ("function endStream()", "function dropStream()"):
            i = APP_JS.index(fn)
            assert "streamRaw = '';" in APP_JS[i:i + 500], fn

    def test_a_dropped_stream_stops_the_timer_too(self):
        i = APP_JS.index("function dropStream()")
        assert "stopStreamRender();" in APP_JS[i:i + 200]


class TestStyle:
    def test_the_bubble_no_longer_forces_preformatted_text(self):
        """Rendered block elements bring their own spacing."""
        assert ".assistant.streaming > .md { white-space: normal; }" in APP_CSS
        i = APP_CSS.index(".assistant.streaming {")
        assert "white-space: pre-wrap" not in APP_CSS[i:i + 120]
