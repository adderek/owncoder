"""The tab tells you when the agent needs you.

A turn runs for minutes and the permission / loop-guard prompts DENY or stop
on timeout, so someone who tabbed away could silently lose a tool call. The
title, the favicon and (opt-in) a desktop notification carry that state.
"""
from pathlib import Path

from agent.ui.http_loop import _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")


class TestPage:
    def test_the_page_declares_an_icon_to_repaint(self):
        """There was no <link rel=icon> at all: every load 404'd on /favicon.ico."""
        assert '<link rel="icon" id="favicon"' in _PAGE

    def test_the_notification_toggle_is_in_the_header(self):
        assert 'id="notifytoggle"' in _PAGE
        assert 'aria-label="Toggle desktop notifications"' in _PAGE


class TestSignals:
    def test_a_pending_prompt_flags_the_tab(self):
        for ev, marker in (("permission", "setAttention('wait'"),
                           ("loopguard", "setAttention('wait'"),
                           ("ask", "setAttention('wait'")):
            i = APP_JS.index("ev.type === '" + ev + "'")
            assert marker in APP_JS[i:i + 400], ev

    def test_answering_a_prompt_clears_the_flag(self):
        for ev in ("permission_done", "loopguard_done"):
            i = APP_JS.index("ev.type === '" + ev + "'")
            assert "clearAttention();" in APP_JS[i:i + 200], ev

    def test_only_a_turn_that_ran_counts_as_finished(self):
        """The server also reports idle on connect; that is not news."""
        i = APP_JS.index("ev.type === 'state'")
        seg = APP_JS[i:APP_JS.index("ev.type === 'switched'")]
        assert "const wasBusy = busyFlag;" in seg
        assert "else if (wasBusy) setAttention('done'" in seg

    def test_a_pending_question_outranks_finished(self):
        i = APP_JS.index("ev.type === 'state'")
        seg = APP_JS[i:APP_JS.index("ev.type === 'switched'")]
        assert seg.index("askbox") < seg.index("setAttention('done'")

    def test_typing_clears_the_flag(self):
        i = APP_JS.index("async function send()")
        assert "clearAttention();" in APP_JS[i:i + 300]


class TestNotificationsAreOptIn:
    def test_permission_is_never_requested_on_load(self):
        """Requesting unprompted is the behaviour every site is disliked for."""
        i = APP_JS.index("Notification.requestPermission")
        seg = APP_JS[:i]
        # the only request sits inside the toggle's click handler
        assert seg.rindex("getElementById('notifytoggle').addEventListener") > \
            seg.rindex("function setAttention(")

    def test_both_the_browser_grant_and_the_toggle_are_required(self):
        i = APP_JS.index("function notifyEnabled()")
        body = APP_JS[i:APP_JS.index("function notify(")]
        assert "Notification.permission === 'granted'" in body
        assert "localStorage.getItem('oc-notify') === '1'" in body

    def test_nothing_is_sent_while_the_user_is_watching(self):
        i = APP_JS.index("function setAttention(")
        body = APP_JS[i:APP_JS.index("function clearAttention(")]
        assert "away()" in body and "notify(" in body

    def test_away_covers_the_unfocused_window(self):
        """A visible window behind an editor is still not being watched."""
        i = APP_JS.index("function away()")
        body = APP_JS[i:i + 200]
        assert "document.hidden" in body and "hasFocus" in body
