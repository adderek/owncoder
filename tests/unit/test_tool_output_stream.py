"""Tool results reach the UI, not just the fact that a tool ran.

The turn engine had the result text in hand and told the UIs only the name
and a boolean, so "⚙ read_file ✓" was all a reader ever got.
"""
import inspect
import json
from pathlib import Path

import pytest

from agent.ui.http_loop import _result_preview

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
UI = Path(__file__).resolve().parents[2] / "ui"


class TestCallbackPlumbing:
    @pytest.mark.parametrize("mod,fn", [
        ("agent.core.turn", "run_turn"),
        ("agent.core.agent", None),
        ("agent.ui_server.local", None),
        ("agent.ui_server.remote_bridge", None),
        ("agent.ui_server.protocol", None),
    ])
    def test_every_layer_accepts_it(self, mod, fn):
        import importlib
        m = importlib.import_module(mod)
        src = inspect.getsource(m)
        assert "on_tool_record" in src, mod

    def test_it_is_optional_everywhere(self):
        """Existing UIs pass none of it and must keep working."""
        import agent.core.turn as turn
        sig = inspect.signature(turn.run_turn)
        assert sig.parameters["on_tool_record"].default is None

    def test_a_failing_callback_cannot_break_a_turn(self):
        src = inspect.getsource(__import__("agent.core.turn", fromlist=["x"]))
        i = src.index("if on_tool_record is not None:")
        assert "except Exception:" in src[i:i + 700]

    def test_the_record_matches_what_the_side_log_keeps(self):
        """One shape, so a reader of either sees the same fields."""
        src = inspect.getsource(__import__("agent.core.turn", fromlist=["x"]))
        i = src.index("if on_tool_record is not None:")
        block = src[i:i + 700]
        for key in ("turn", "tool_call_id", "tool", "arguments", "result",
                    "ok", "duration_ms"):
            assert '"%s"' % key in block, key


class TestParallelMode:
    """The IPC boundary carries it too, or it would silently vanish there."""

    def test_the_record_survives_the_wire(self):
        from agent.ipc.messages import ToolRecordEvent, event_from_wire

        rec = {"turn": 3, "tool_call_id": "c1", "tool": "bash",
               "arguments": {"cmd": "ls"}, "result": "a\nb", "ok": True,
               "duration_ms": 12.5}
        back = event_from_wire(ToolRecordEvent(rec).to_wire())
        assert isinstance(back, ToolRecordEvent)
        assert back.record == rec

    def test_the_controller_forwards_it(self):
        src = inspect.getsource(__import__("agent.ipc.controller", fromlist=["x"]))
        assert "_safe_call(on_tool_record, event.record)" in src

    def test_the_worker_sends_it(self):
        src = inspect.getsource(__import__("agent.ipc.agent_worker", fromlist=["x"]))
        assert "on_tool_record=_on_tool_record" in src


class TestPreview:
    def test_it_unwraps_the_common_envelopes(self):
        assert _result_preview(json.dumps({"output": "hello"})) == "hello"
        assert _result_preview(json.dumps({"stdout": "out"})) == "out"
        assert _result_preview(json.dumps({"error": "nope"})) == "nope"

    def test_an_unrecognised_envelope_is_shown_as_json(self):
        assert _result_preview(json.dumps({"rows": 3})) == '{\n  "rows": 3\n}'

    def test_plain_text_passes_through(self):
        assert _result_preview("just text") == "just text"

    def test_it_is_capped(self):
        """Tool output is unbounded and goes to every connected client."""
        out = _result_preview("x" * 9000)
        assert len(out) == 4001 and out.endswith("…")

    def test_nothing_is_not_a_crash(self):
        assert _result_preview(None) == ""


class TestHttpEvent:
    def test_the_event_is_published(self):
        src = (UI / "http_loop.py").read_text(encoding="utf-8")
        i = src.index("on_tool_record=lambda rec:")
        block = src[i:i + 400]
        assert '"type": "tool_io"' in block
        assert "_result_preview(rec.get(\"result\"))" in block

    def test_the_browser_attaches_it_to_the_right_fold(self):
        i = APP_JS.index("ev.type === 'tool_io'")
        assert "toolOutput(ev.name" in APP_JS[i:i + 120]

    def test_folds_awaiting_their_text_are_bounded(self):
        """A backend that never sends it must not leak elements."""
        i = APP_JS.index("function toolResult(")
        assert "if (q.length > 20) q.shift();" in APP_JS[i:i + 600]

    def test_output_is_inserted_as_text_not_html(self):
        i = APP_JS.index("function toolOutput(")
        body = APP_JS[i:APP_JS.index("\n}", i)]
        assert "body.textContent = text;" in body
        assert "innerHTML" not in body
