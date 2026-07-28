"""An answer can be asked for again.

Retry appeared only when a turn *failed*. A poor answer could not be re-run:
✎ re-sent the question, but the poor answer stayed in the history and shaped
whatever came next.
"""
from pathlib import Path

from agent.ui.http_loop import _HttpUI

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Server:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    def get_messages(self):
        return self.msgs

    def set_messages(self, msgs, session_id=""):
        self.msgs = list(msgs)


def _ui(msgs):
    ui = _HttpUI.__new__(_HttpUI)
    ui.session = None
    ui.server = _Server(msgs)
    return ui


class TestDrop:
    def test_it_removes_the_answer_and_everything_it_produced(self):
        """Left in place, the model reads its own answer and agrees with it."""
        ui = _ui([
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "a2"},
        ])
        out = ui.drop_last_exchange()
        assert out["ok"] and out["text"] == "q2" and out["dropped"] == 4
        assert [m["role"] for m in ui.server.msgs] == ["user", "assistant"]

    def test_the_earlier_history_is_untouched(self):
        ui = _ui([{"role": "user", "content": "q1"},
                  {"role": "assistant", "content": "a1"},
                  {"role": "user", "content": "q2"},
                  {"role": "assistant", "content": "a2"}])
        ui.drop_last_exchange()
        assert ui.server.msgs == [{"role": "user", "content": "q1"},
                                  {"role": "assistant", "content": "a1"}]

    def test_an_empty_history_is_refused(self):
        out = _ui([]).drop_last_exchange()
        assert out["ok"] is False

    def test_a_history_with_no_user_turn_is_refused(self):
        out = _ui([{"role": "assistant", "content": "hello"}]).drop_last_exchange()
        assert out["ok"] is False and "nothing" in out["msg"]

    def test_a_textless_last_message_is_refused(self):
        """Nothing to re-send, and dropping the history anyway would lose it."""
        ui = _ui([{"role": "user", "content": ""}])
        assert ui.drop_last_exchange()["ok"] is False
        assert len(ui.server.msgs) == 1

    def test_the_route_refuses_mid_turn(self):
        i = HTTP_LOOP.index('elif self.path == "/api/regenerate":')
        block = HTTP_LOOP[i:i + 300]
        assert "if ui.busy:" in block
        assert block.index("ui.busy") < block.index("drop_last_exchange")


class TestButton:
    def test_only_answers_carry_it(self):
        i = APP_JS.index("function row(cls, html, text)")
        body = APP_JS[i:APP_JS.index("function copyText(")]
        assert "cls.indexOf('assistant') > 0" in body
        assert "'copy regen'" in body

    def test_only_the_last_answer_can_be_rerun(self):
        """Re-running an older one leaves later turns answering a dead question."""
        i = APP_JS.index("const gb = e.target.closest('.regen');")
        body = APP_JS[i:i + 600]
        assert "rows[rows.length - 1]" in body
        assert "only the last answer can be regenerated" in body

    def test_it_waits_for_a_running_turn(self):
        i = APP_JS.index("async function regenerate(")
        assert "if (busyFlag)" in APP_JS[i:i + 200]

    def test_it_drops_before_it_asks(self):
        i = APP_JS.index("async function regenerate(")
        body = APP_JS[i:i + 900]
        assert body.index("'/api/regenerate'") < body.index("'/api/chat'")

    def test_the_view_is_rebuilt_from_the_server(self):
        """The dropped exchange has to leave the log as well as the history."""
        i = APP_JS.index("async function regenerate(")
        assert "await resyncView();" in APP_JS[i:i + 900]

    def test_a_server_refusal_stops_it(self):
        i = APP_JS.index("async function regenerate(")
        body = APP_JS[i:i + 900]
        assert "if (!r.ok)" in body and body.index("if (!r.ok)") < body.index("'/api/chat'")
