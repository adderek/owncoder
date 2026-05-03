"""Unit tests for agent/memory/session.py — session persistence."""
from __future__ import annotations

import pytest
from agent.memory.session import (
    new_session,
    save_session,
    load_session,
    list_sessions,
    configure,
    _sanitize_short_name,
)


@pytest.fixture(autouse=True)
def _setup_session_dir(tmp_path):
    configure(str(tmp_path), ".agent")
    yield


class TestNewSession:
    def test_creates_session_with_id(self):
        s = new_session()
        assert s.id
        assert "T" in s.id  # ISO-8601 format

    def test_short_name_sanitized(self):
        s = new_session(short_name="hello world!@#$")
        assert s.short_name == "helloworld"

    def test_with_tags(self):
        s = new_session(tags=["debug", "test"])
        assert s.tags == ["debug", "test"]


class TestSanitizeShortName:
    def test_strips_unsafe_chars(self):
        assert _sanitize_short_name("hello world!") == "helloworld"

    def test_allows_safe_chars(self):
        assert _sanitize_short_name("hello-world_v2.0") == "hello-world_v2.0"

    def test_truncates_long_names(self):
        assert len(_sanitize_short_name("a" * 100)) == 64


class TestSaveLoadRoundTrip:
    def test_basic_roundtrip(self):
        s = new_session(short_name="test")
        messages = [{"role": "user", "content": "hi"}]
        save_session(s, messages)
        loaded, loaded_msgs = load_session(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded_msgs == messages

    def test_load_by_short_name(self):
        s = new_session(short_name="myname")
        save_session(s, [{"role": "user", "content": "hello"}])
        loaded, msgs = load_session("myname")
        assert loaded is not None
        assert loaded.short_name == "myname"

    def test_load_nonexistent(self):
        loaded, msgs = load_session("does-not-exist")
        assert loaded is None
        assert msgs == []

    def test_save_updates_timestamp(self):
        s = new_session()
        original_ts = s.updated_at
        save_session(s, [])
        assert s.updated_at >= original_ts


class TestListSessions:
    def test_empty(self):
        sessions = list_sessions()
        assert isinstance(sessions, list)

    def test_lists_saved_sessions(self):
        s1 = new_session(short_name="s1", name="Session One")
        save_session(s1, [{"role": "user", "content": "a"}])
        s2 = new_session(short_name="s2", name="Session Two")
        save_session(s2, [{"role": "user", "content": "b"}])

        sessions = list_sessions()
        assert len(sessions) >= 2
        names = {s["short_name"] for s in sessions}
        assert "s1" in names
        assert "s2" in names
