"""Chat restore + fold/unfold (ui/view_mixin.py _restore_chat_history/_rerender_chat)."""
from __future__ import annotations

import asyncio

from textual.app import App

from agent.config.models import ThemeConfig
from agent.ui.textual_widgets import build_widget_classes
from agent.ui.view_mixin import ViewMixin

_theme = ThemeConfig()
_w = build_widget_classes(_theme)


class _FakeServer:
    def get_ui_config(self, session_id=""):
        return {"chat_restore_expand_last": 2}


class _App(ViewMixin, App):
    _t = _theme
    _wt = _w
    _server = _FakeServer()
    _session = None
    _agent_running = False

    def compose(self):
        yield _w.ConversationView(id="chat-log", markup=True, highlight=False)


def _history(n=5):
    msgs, qa = [], []
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer **{i}**",
                     "tool_calls": [{"function": {"name": "read_file"}}]})
        qa.append((i, {"turn_id": i, "content": f"question {i}", "summary_q": f"Q{i}"},
                   {"turn_id": i, "content": f"answer **{i}**", "summary_a": f"A{i}",
                    "tool_calls": ["read_file"],
                    "modified_files": [{"path": f"f{i}.py", "added": 3, "removed": 1}],
                    "model_calls": [{"role": "main", "model": "m", "tier": "local", "t": 0.1}],
                    "duration": 5.0}))
    return msgs, qa


def test_restore_folds_old_turns_and_registers_click_targets():
    async def go():
        app = _App()
        async with app.run_test(size=(100, 40)) as pilot:
            msgs, qa = _history()
            app._restore_chat_history(msgs, resume_marker=True, qa_entries=qa)
            await pilot.pause()
            log = app.query_one("#chat-log")
            assert app._chat_folded == {0, 1, 2}
            # two expanded turns → two distinct round-stats payloads + file targets
            assert len({id(v) for v in app._chat_model_lines.values()}) == 2
            assert app._chat_file_lines
            assert len(app._chat_line_to_ordinal) == len(log.lines)
            assert len(app._chat_user_lines) == 5
    asyncio.run(go())


def test_toggle_fold_rerenders_and_click_toggles():
    async def go():
        app = _App()
        async with app.run_test(size=(100, 40)) as pilot:
            msgs, qa = _history()
            app._restore_chat_history(msgs, qa_entries=qa)
            await pilot.pause()
            log = app.query_one("#chat-log")
            n_folded = len(log.lines)
            app._toggle_chat_fold(0)
            await pilot.pause()
            assert 0 not in app._chat_folded
            assert len(log.lines) > n_folded
            app._toggle_chat_fold(0)
            await pilot.pause()
            assert len(log.lines) == n_folded
            # blocked while agent runs
            app._agent_running = True
            app._toggle_chat_fold(1)
            assert 1 in app._chat_folded
            app._agent_running = False
            # mouse click on first line (folded turn 0) unfolds
            await pilot.click("#chat-log", offset=(5, 0))
            await pilot.pause()
            assert 0 not in app._chat_folded
    asyncio.run(go())
