"""The readline prompt must bracket its colour escapes.

GNU readline measures the prompt to place the cursor and to redraw the line.
An unbracketed escape is counted as printable width, so the prompt miscounts
itself and a redraw puts the raw "[38;2;56;142;60m>" on screen instead of a
green ">". The markers below are readline's RL_PROMPT_START_IGNORE /
RL_PROMPT_END_IGNORE.
"""
from __future__ import annotations

from agent.ui.colors import _hex_to_ansi, readline_prompt


class TestReadlinePrompt:
    def test_the_escape_is_bracketed_and_the_text_is_not(self):
        out = readline_prompt("\033[38;2;56;142;60m", ">")
        assert out == "\001\033[38;2;56;142;60m\002>\001\033[0m\002"

    def test_every_non_printing_run_is_wrapped(self):
        """Both the colour and the reset are invisible and must be ignored."""
        out = readline_prompt(_hex_to_ansi("#388E3C"), ">")
        assert out.count("\001") == out.count("\002") == 2

    def test_the_visible_width_is_one_character(self):
        out = readline_prompt(_hex_to_ansi("#388E3C"), ">")
        visible = []
        skipping = False
        for ch in out:
            if ch == "\001":
                skipping = True
            elif ch == "\002":
                skipping = False
            elif not skipping:
                visible.append(ch)
        assert "".join(visible) == ">"

    def test_a_themeless_colour_adds_no_markers(self):
        """_hex_to_ansi returns "" for a named or short colour; bracketing
        nothing would leave stray markers in the prompt."""
        assert readline_prompt("", ">") == ">"
        assert readline_prompt(_hex_to_ansi("green"), ">") == ">"


class TestTheLoopUsesIt:
    def test_the_prompt_is_built_through_the_helper(self):
        import inspect
        from agent.ui import readline_loop

        src = inspect.getsource(readline_loop)
        assert 'readline_prompt(_hex_to_ansi(t.prompt), ">")' in src
        assert "input(prompt_esc)" in src
        assert '\\033[0m ").strip()' not in src
