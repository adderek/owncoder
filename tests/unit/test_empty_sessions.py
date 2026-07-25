"""Sessions nobody spoke in.

Starting a UI and closing it used to leave a session.json holding only the
system prompt. Those files then sat at the top of "recent sessions" — reported
as "1 msgs", showing nothing when opened, because the one message was the
prompt. Three fixes, one per layer: do not write them, do not count the prompt
as a message, and let existing ones be deleted.
"""
from __future__ import annotations

import json

import pytest

from agent.memory import session as sess


@pytest.fixture(autouse=True)
def project(tmp_path):
    sess.configure(str(tmp_path), ".agent")
    yield tmp_path
    sess.configure(".", ".agent")


SYSTEM = [{"role": "system", "content": "You are a coding assistant."}]
CONVERSATION = SYSTEM + [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "hello"}]


class TestCounting:
    def test_only_conversation_counts(self):
        assert sess.content_count(CONVERSATION) == 2

    def test_a_system_only_transcript_counts_zero(self):
        assert sess.content_count(SYSTEM) == 0
        assert sess.has_content(SYSTEM) is False

    def test_the_preamble_placeholder_is_not_a_message(self):
        """This is the literal '1 msgs': saved sessions store the system block
        as a placeholder, and counting it made empty look non-empty."""
        assert sess.content_count([sess._PREAMBLE_PLACEHOLDER]) == 0

    def test_tool_results_count(self):
        """A turn that only ran tools is still work that happened."""
        assert sess.content_count(SYSTEM + [{"role": "tool", "content": "{}"}]) == 1

    def test_nothing_at_all_is_safe(self):
        assert sess.content_count([]) == 0
        assert sess.content_count(None) == 0


class TestNotWritten:
    def test_a_session_with_no_conversation_is_not_saved(self, project):
        session = sess.new_session()
        sess.save_session(session, SYSTEM)
        assert list(project.rglob("session.json")) == []

    def test_the_first_real_message_creates_the_file(self, project):
        session = sess.new_session()
        sess.save_session(session, SYSTEM)
        sess.save_session(session, CONVERSATION)
        assert len(list(project.rglob("session.json"))) == 1

    def test_an_existing_session_is_never_unwritten(self, project):
        """Once a file exists it holds something; saving is not deletion."""
        session = sess.new_session()
        sess.save_session(session, CONVERSATION)
        sess.save_session(session, SYSTEM)      # e.g. a teardown after a reset
        assert len(list(project.rglob("session.json"))) == 1

    def test_incognito_is_still_never_written(self, project):
        session = sess.new_session(mode="incognito")
        sess.save_session(session, CONVERSATION)
        assert list(project.rglob("session.json")) == []


class TestListing:
    def _write_empty(self, project, sid_suffix="empty"):
        session = sess.new_session(short_name=sid_suffix)
        path = sess.get_session_full_dir(session.id) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "id": session.id, "name": sid_suffix, "created_at": 1.0, "updated_at": 1.0,
            "messages": [dict(sess._PREAMBLE_PLACEHOLDER)],
        }), encoding="utf-8")
        return session.id

    def test_empty_sessions_are_left_out_by_default(self, project):
        self._write_empty(project)
        sess.save_session(sess.new_session(), CONVERSATION)
        listed = sess.list_sessions()
        assert len(listed) == 1
        assert listed[0]["message_count"] == 2

    def test_they_can_still_be_listed_on_request(self, project):
        self._write_empty(project)
        assert len(sess.list_sessions(include_empty=True)) == 1

    def test_an_empty_session_is_still_loadable_by_id(self, project):
        """Hidden from a list is not deleted; the file is untouched."""
        sid = self._write_empty(project)
        loaded, messages = sess.load_session(sid)
        assert loaded is not None
        assert sess.content_count(messages) == 0

    def test_the_count_reported_is_the_conversation(self, project):
        sess.save_session(sess.new_session(), CONVERSATION)
        assert sess.list_sessions()[0]["message_count"] == 2


class TestPrune:
    class _Console:
        def __init__(self):
            self.lines = []

        def print(self, *args, **kwargs):
            self.lines.append(" ".join(str(a) for a in args))

    def _empty_on_disk(self, project):
        session = sess.new_session()
        path = sess.get_session_full_dir(session.id) / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": session.id, "created_at": 1.0,
                                    "updated_at": 1.0, "messages": []}), encoding="utf-8")
        return session.id

    def test_a_dry_run_deletes_nothing(self, project):
        from agent.cli.sessions import _prune_empty

        sid = self._empty_on_disk(project)
        console = self._Console()
        _prune_empty(console, confirmed=False)
        assert sess.get_session_full_dir(sid).is_dir()
        assert any("-y" in line for line in console.lines)

    def test_confirmed_removes_the_directory(self, project):
        from agent.cli.sessions import _prune_empty

        sid = self._empty_on_disk(project)
        _prune_empty(self._Console(), confirmed=True)
        assert not sess.get_session_full_dir(sid).exists()

    def test_a_session_with_content_is_never_touched(self, project):
        """The one thing this must not do is delete someone's work."""
        from agent.cli.sessions import _prune_empty

        session = sess.new_session()
        sess.save_session(session, CONVERSATION)
        self._empty_on_disk(project)
        _prune_empty(self._Console(), confirmed=True)
        assert sess.get_session_full_dir(session.id).is_dir()
        assert sess.load_session(session.id)[0] is not None

    def test_nothing_to_do_says_so(self, project):
        from agent.cli.sessions import _prune_empty

        console = self._Console()
        _prune_empty(console, confirmed=True)
        assert any("No content-free" in line for line in console.lines)


def test_the_flags_are_registered():
    from agent.cli.main import build_parser

    args = build_parser().parse_args(["sessions", "--prune-empty", "-y"])
    assert args.prune_empty is True and args.yes is True
    assert build_parser().parse_args(["sessions", "--include-empty"]).include_empty is True
