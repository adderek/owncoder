"""Your own messages can be edited and sent again.

Rows carried a copy button and nothing else, so changing one word of a long
prompt meant retyping or copy-pasting it.
"""
from pathlib import Path

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")
ROW = APP_JS[APP_JS.index("function row(cls, html, text)"):
             APP_JS.index("function copyText(")]


class TestButton:
    def test_only_user_messages_get_it(self):
        """Re-sending the agent's own words is not a thing anyone wants."""
        assert "if (cls.indexOf('user') > 0) {" in ROW
        i = ROW.index("if (cls.indexOf('user') > 0) {")
        assert "'copy reuse'" in ROW[i:i + 400]

    def test_it_does_not_sit_on_top_of_copy(self):
        assert ".copy.reuse { right: 34px; }" in APP_CSS


class TestBehaviour:
    def test_it_loads_the_box_instead_of_sending(self):
        """A re-run costs a whole turn; it should take a deliberate Enter."""
        i = APP_JS.index("function reuseMessage(")
        body = APP_JS[i:i + 500]
        assert "input.value" in body
        assert "send()" not in body

    def test_it_keeps_whatever_was_already_typed(self):
        i = APP_JS.index("function reuseMessage(")
        body = APP_JS[i:i + 500]
        assert "cur && cur !== text" in body

    def test_the_click_is_handled_before_plain_copy(self):
        """.reuse also carries .copy for its styling."""
        i = APP_JS.index("log.addEventListener('click'")
        body = APP_JS[i:APP_JS.index("function reuseMessage(")]
        assert body.index("closest('.reuse')") < body.index("closest('.copy')")
