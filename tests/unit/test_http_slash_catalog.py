"""The message box completes slash commands it can actually run.

The placeholder promised "/ for commands" long before anything completed
them. The catalogue is the terminal UI's own table minus what only a
terminal can do, so the two never drift apart.
"""
import re
from pathlib import Path

import pytest

from agent.ui.http_loop import _TERMINAL_ONLY, _slash_catalog

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
PALETTE = APP_JS[APP_JS.index("let slashCmds = null;"):
                 APP_JS.index("// ── Message history")]


def _handler_source() -> str:
    src = HTTP_LOOP.read_text(encoding="utf-8")
    i = src.index("async def _handle_slash")
    return src[i:src.index("\nasync def ", i + 10)]


class TestCatalog:
    def test_it_is_not_empty(self):
        assert len(_slash_catalog()) > 20

    def test_every_offered_command_is_handled(self):
        """Advertising a command the browser drops on the floor is worse than
        not advertising it."""
        handled = set(re.findall(r'"(/[a-z_?!-]+)"', _handler_source()))
        handled.add("/clear")          # handled in the browser, never sent
        missing = [c["name"] for c in _slash_catalog() if c["name"] not in handled]
        assert missing == []

    @pytest.mark.parametrize("name", sorted(_TERMINAL_ONLY))
    def test_terminal_only_commands_are_withheld(self, name):
        assert name not in [c["name"] for c in _slash_catalog()]

    def test_entries_carry_what_the_ui_needs(self):
        for c in _slash_catalog():
            assert set(c) == {"name", "aliases", "desc", "arg"}, c
            assert c["name"].startswith("/") and c["desc"]
            assert isinstance(c["arg"], bool)

    def test_it_is_one_table_with_the_terminal_ui(self):
        """No second hand-maintained list to fall out of date."""
        src = HTTP_LOOP.read_text(encoding="utf-8")
        i = src.index("def _slash_catalog")
        assert "from agent.ui.slash import _SLASH_COMMANDS" in src[i:i + 900]


class TestEndpoint:
    def test_the_route_exists(self):
        src = HTTP_LOOP.read_text(encoding="utf-8")
        assert 'elif self.path == "/api/slash":' in src
        assert '"commands": _slash_catalog()' in src


class TestPalette:
    def test_it_only_shows_while_the_word_is_unfinished(self):
        i = PALETTE.index("function slashQuery()")
        body = PALETTE[i:i + 300]
        assert "v.startsWith('/')" in body and "/[\\s\\n]/.test(v)" in body

    def test_prefix_matches_come_first(self):
        i = PALETTE.index("function slashRank(")
        body = PALETTE[i:PALETTE.index("function slashBox()")]
        assert "n.startsWith(ql)" in body
        assert "return pre.concat(sub)" in body

    def test_the_catalogue_is_fetched_once(self):
        i = PALETTE.index("async function loadSlashCmds()")
        assert "if (slashCmds) return slashCmds;" in PALETTE[i:i + 200]

    def test_the_palette_owns_its_keys(self):
        """Otherwise ↑ would walk the history and Esc would shut the drawers."""
        i = APP_JS.index("input.addEventListener('keydown'")
        body = APP_JS[i:APP_JS.index("input.addEventListener('input'", i)]
        assert body.index("slashKey(e)") < body.index("histMove(-1)")
        assert "e.stopPropagation();" in body

    def test_a_command_taking_an_argument_completes_with_a_space(self):
        i = PALETTE.index("function slashApply(")
        assert "cmd.name + (cmd.arg ? ' ' : '')" in PALETTE[i:i + 300]

    def test_nothing_is_sent_by_surprise(self):
        """Enter completes the highlighted command unless you typed it out."""
        i = PALETTE.index("if (e.key === 'Enter'")
        assert "sel.name !== input.value.trim()" in PALETTE[i:i + 400]

    def test_it_picks_on_mousedown(self):
        """click lands after blur, and blur has already closed the list."""
        assert "addEventListener('mousedown'" in PALETTE
        # blur closes every completion popup, this one included
        i = APP_JS.index("input.addEventListener('blur'")
        assert "slashClose();" in APP_JS[i:i + 120]
