"""The context warning offers the remedy.

The token bar turned red past 75% and stopped there; the fix, /compact, was a
command you had to already know about. It was the only warning in the UI with
no next step attached.
"""
from pathlib import Path

from agent.ui.http_loop import _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")


class TestVisibility:
    def test_the_button_exists_and_starts_hidden(self):
        assert 'id="compact"' in _PAGE
        i = _PAGE.index('id="compact"')
        assert 'style="display:none"' in _PAGE[i:i + 200]

    def test_it_appears_exactly_when_the_bar_goes_hot(self):
        i = APP_JS.index("fill.className = pct > 75 ? 'hot' : '';")
        body = APP_JS[i:i + 400]
        assert "cb.style.display = pct > 75 ? '' : 'none';" in body

    def test_the_tooltip_says_how_full_it_is(self):
        i = APP_JS.index("fill.className = pct > 75 ? 'hot' : '';")
        assert "Math.round(pct) + '% full" in APP_JS[i:i + 500]

    def test_it_is_coloured_like_the_warning_it_answers(self):
        i = APP_CSS.index("#compact {")
        assert "var(--warn)" in APP_CSS[i:i + 120]


class TestAction:
    def test_it_sends_the_same_command_a_person_would_type(self):
        """One code path server-side, so the two cannot drift."""
        i = APP_JS.index("getElementById('compact').addEventListener")
        body = APP_JS[i:i + 900]
        assert "'/api/chat'" in body and "text: '/compact'" in body

    def test_it_refuses_while_a_turn_is_running(self):
        """Compacting rewrites the history the running turn is reading."""
        i = APP_JS.index("getElementById('compact').addEventListener")
        body = APP_JS[i:i + 900]
        assert "if (busyFlag) {" in body
        assert body.index("if (busyFlag)") < body.index("'/api/chat'")

    def test_the_refusal_explains_itself(self):
        assert "compact waits for the running turn" in APP_JS

    def test_the_button_cannot_be_double_fired(self):
        i = APP_JS.index("getElementById('compact').addEventListener")
        body = APP_JS[i:i + 900]
        assert "btn.disabled = true;" in body

    def test_it_recovers_even_when_the_post_fails(self):
        i = APP_JS.index("getElementById('compact').addEventListener")
        body = APP_JS[i:i + 900]
        assert "} finally {" in body and "btn.disabled = false;" in body
