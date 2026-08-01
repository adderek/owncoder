"""Auto-fold of a round's action journal (ui/view_mixin.py _auto_fold_turn and
friends, hooked from _append_qa_turn and terminal.py._begin_chat).

A round's journal is live feedback while the agent works and clutter once
it's done, so the *current* round must stay expanded while a *finished* one
folds to its one-line header — but only when the user hasn't already made
their own call on that turn. A regression here means either the live round
folds while still running (losing the only feedback available), a completed
round never folds and the log turns to noise, or a manual fold/unfold gets
silently reverted the next time a round starts — the "pin" contract broken.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from textual.app import App

from agent.config.models import ThemeConfig
from agent.ui.textual_widgets import build_widget_classes
from agent.ui.view_mixin import ViewMixin

_theme = ThemeConfig()
_w = build_widget_classes(_theme)


def _server(fold_journal="on_next_round", missing_config=False):
    class _Srv:
        def get_ui_config(self, session_id: str = "") -> dict:
            return {"chat_restore_expand_last": 2}

    srv = _Srv()
    if missing_config:
        srv._agent = None
    else:
        srv._agent = SimpleNamespace(
            config=SimpleNamespace(
                ui=SimpleNamespace(changeset=SimpleNamespace(fold_journal=fold_journal))
            )
        )
    return srv


class _App(ViewMixin, App):
    _t = _theme
    _wt = _w
    _session = None
    _agent_running = False

    def compose(self):
        yield _w.ConversationView(id="chat-log", markup=True, highlight=False)


def _turn(i: int) -> tuple:
    q_d = {"turn_id": i, "content": f"question {i}", "summary_q": f"Q{i}"}
    a_d = {
        "turn_id": i,
        "content": f"answer {i}",
        "summary_a": f"A{i}",
        "tool_calls": ["read_file"],
        "modified_files": [{"path": f"f{i}.py", "added": 1, "removed": 0}],
        "model_calls": [{"role": "main", "model": "m", "tier": "local", "t": 0.1}],
        "duration": 1.0,
    }
    return q_d, a_d


def _run(coro) -> None:
    asyncio.run(coro())


class TestCurrentRoundStaysExpanded:
    def test_a_just_finished_round_is_not_folded_before_the_next_one_starts(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()
                assert getattr(app, "_chat_folded", set()) == set()
        _run(go)


class TestNewRoundFoldsThePrevious:
    def test_starting_round_two_folds_round_one(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()  # round 0 ends: stays open
                app._auto_fold_before_new_round()  # round 1 begins
                assert app._chat_folded == {0}

                app._chat_qa_data.append(_turn(1))
                app._auto_fold_finished_round()  # round 1 ends: stays open
                assert 1 not in app._chat_folded
        _run(go)


class TestManualPinWins:
    def test_a_hand_unfolded_turn_stays_unfolded_across_two_further_rounds(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_before_new_round()  # round 1 begins: folds turn 0
                assert 0 in app._chat_folded

                app._toggle_chat_fold(0)  # user unfolds it by hand
                assert 0 not in app._chat_folded
                assert 0 in app._chat_fold_pinned

                # Two more auto-fold passes over the same ordinal must never
                # override the pin.
                app._auto_fold_turn(0)
                app._auto_fold_turn(0)
                assert 0 not in app._chat_folded
        _run(go)

    def test_a_hand_folded_turn_stays_folded(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._toggle_chat_fold(0)  # user folds the still-current round
                assert 0 in app._chat_folded
                assert 0 in app._chat_fold_pinned

                app._auto_fold_before_new_round()  # round 1 begins
                app._auto_fold_turn(0)
                assert 0 in app._chat_folded
        _run(go)


class TestFoldJournalModes:
    def test_on_next_round_waits_for_the_next_round_to_start(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()
                assert getattr(app, "_chat_folded", set()) == set()
                app._auto_fold_before_new_round()
                assert app._chat_folded == {0}
        _run(go)

    def test_immediately_folds_as_soon_as_the_round_ends(self):
        async def go():
            app = _App()
            app._server = _server("immediately")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()
                assert app._chat_folded == {0}
        _run(go)

    def test_never_never_auto_folds(self):
        async def go():
            app = _App()
            app._server = _server("never")
            async with app.run_test():
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()
                app._auto_fold_before_new_round()
                assert getattr(app, "_chat_folded", set()) == set()
        _run(go)


class TestMissingConfigDefaultsSanely:
    def test_a_server_with_no_config_falls_back_to_on_next_round(self):
        async def go():
            app = _App()
            app._server = _server(missing_config=True)
            async with app.run_test():
                assert app._fold_journal_mode() == "on_next_round"
                app._chat_qa_data = [_turn(0)]
                app._auto_fold_finished_round()
                assert getattr(app, "_chat_folded", set()) == set()
                app._auto_fold_before_new_round()
                assert app._chat_folded == {0}
        _run(go)


class TestResumeDoesNotPin:
    def test_folds_from_session_restore_are_not_treated_as_user_pins(self):
        async def go():
            app = _App()
            app._server = _server("on_next_round")
            async with app.run_test():
                msgs, qa = [], []
                for i in range(4):
                    msgs.append({"role": "user", "content": f"question {i}"})
                    msgs.append({
                        "role": "assistant", "content": f"answer {i}",
                        "tool_calls": [{"function": {"name": "read_file"}}],
                    })
                    qa.append((i, *_turn(i)))
                app._restore_chat_history(msgs, qa_entries=qa)
                assert app._chat_folded  # expand_last=2 folds the older ones
                assert app._chat_fold_pinned == set()
        _run(go)
