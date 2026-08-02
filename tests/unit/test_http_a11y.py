"""Header controls are reachable without a mouse.

The chips were <span>s carrying click handlers: not tabbable, no Enter/Space,
no focus ring. Anything clickable in the header is a real button now.
"""
import re
from pathlib import Path

import pytest

from agent.ui.http_loop import _PAGE

APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")

HEADER = _PAGE[_PAGE.index('<div id="header">'):_PAGE.index('<div id="main">')]

CLICKABLE = ["model", "session", "workdir", "privchip", "layout", "condchip", "iostats",
             "bgchip", "tokenwrap"]


class TestKeyboardReach:
    @pytest.mark.parametrize("el_id", CLICKABLE)
    def test_every_clickable_header_element_is_a_button(self, el_id):
        assert "getElementById('%s')" % el_id in APP_JS or \
            "getElementById(\"%s\")" % el_id in APP_JS, "no handler?"
        m = re.search(r'<(\w+)[^>]*\bid="%s"' % el_id, HEADER)
        assert m, el_id
        assert m.group(1) == "button", "%s is a <%s>" % (el_id, m.group(1))

    @pytest.mark.parametrize("el_id", CLICKABLE)
    def test_the_buttons_do_not_submit_anything(self, el_id):
        """Default type=submit; harmless without a form today, a trap later."""
        m = re.search(r'<button[^>]*\bid="%s"[^>]*>' % el_id, HEADER)
        assert 'type="button"' in m.group(0), el_id

    def test_an_icon_only_button_says_what_it_does(self):
        for m in re.finditer(r'<button[^>]*class="icon"[^>]*>', HEADER):
            assert "aria-label=" in m.group(0), m.group(0)

    def test_the_token_bar_has_a_name(self):
        """Its content is two bars and a number: nothing to read out."""
        m = re.search(r'<button[^>]*\bid="tokenwrap"[^>]*>', HEADER)
        assert "aria-label=" in m.group(0)


class TestFocusIsVisible:
    def test_there_is_a_ring_for_keyboard_users(self):
        assert ":focus-visible { outline: 2px solid var(--accent)" in APP_CSS

    def test_chips_reset_the_button_font(self):
        """A <button> inherits neither font nor line-height."""
        chip = APP_CSS[APP_CSS.index(".chip {"):APP_CSS.index(".chip.btn {")]
        assert "font-family: inherit" in chip


class TestLiveState:
    def test_the_status_text_is_announced(self):
        assert 'id="status" role="status" aria-live="polite"' in HEADER
