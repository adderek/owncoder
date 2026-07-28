"""The countdown prompts can be answered from the keyboard.

A permission prompt DENYs when it expires and the loop guard stops the turn,
which makes them the most time-critical question this UI asks — and they
could only be answered by finding a small button with the mouse.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
ARM = APP_JS[APP_JS.index("function armPromptKeys("):
             APP_JS.index("function loopGuardPrompt(")]


class TestBothPromptsAreArmed:
    def test_the_permission_prompt(self):
        i = APP_JS.index("function permissionPrompt(")
        assert "armPromptKeys(d);" in APP_JS[i:APP_JS.index("function resolvePermission(")]

    def test_the_loop_guard(self):
        i = APP_JS.index("function loopGuardPrompt(")
        assert "armPromptKeys(d);" in APP_JS[i:APP_JS.index("function resolveLoopGuard(")]


class TestHints:
    def test_choices_are_numbered_on_screen(self):
        assert "'<span class=\"key\">' + (i + 1) + '</span>'" in ARM
        assert ".loopguard .key {" in APP_CSS

    def test_the_number_is_in_the_tooltip_too(self):
        assert "'press ' + (i + 1)" in ARM

    def test_the_first_choice_takes_focus(self):
        """So Enter and Tab work without touching the mouse."""
        assert "btns[0].focus();" in ARM

    def test_focus_is_not_stolen_from_the_message_box(self):
        assert "document.activeElement !== input" in ARM


class TestGuards:
    def test_digits_typed_into_a_field_are_not_answers(self):
        i = APP_JS.index("const box = permEl || lgEl;")
        body = APP_JS[i:i + 600]
        assert "el.tagName === 'TEXTAREA' || el.tagName === 'INPUT'" in body

    def test_modified_digits_are_left_to_the_browser(self):
        i = APP_JS.index("const box = permEl || lgEl;")
        assert "e.ctrlKey || e.metaKey || e.altKey" in APP_JS[i - 120:i + 120]

    def test_nothing_happens_without_a_pending_prompt(self):
        i = APP_JS.index("const box = permEl || lgEl;")
        assert "if (!box" in APP_JS[i:i + 120]

    def test_a_digit_beyond_the_choices_does_nothing(self):
        i = APP_JS.index("const box = permEl || lgEl;")
        body = APP_JS[i:i + 600]
        assert "if (!btn) return;" in body
