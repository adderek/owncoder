"""Tests for session listing order/cap and substring search."""
from __future__ import annotations

import time

from agent.memory import session as sess


def _seed(tmp_path):
    sess.configure(str(tmp_path), ".agent")
    a = sess.new_session(name="Alpha refactor", classification="refactor", tags=["db", "perf"])
    sess.save_session(a, [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
    time.sleep(0.01)
    b = sess.new_session(name="Beta auth bugfix", classification="bugfix", tags=["auth"])
    sess.save_session(b, [{"role": "user", "content": "x"}])
    time.sleep(0.01)
    c = sess.new_session(name="Gamma docs", classification="docs", tags=["readme"])
    sess.save_session(c, [{"role": "user", "content": "x"}])
    return a, b, c


def test_list_oldest_first(tmp_path):
    a, b, c = _seed(tmp_path)
    names = [s["name"] for s in sess.list_sessions(oldest_first=True)]
    assert names == ["Alpha refactor", "Beta auth bugfix", "Gamma docs"]


def test_list_newest_first_default(tmp_path):
    a, b, c = _seed(tmp_path)
    names = [s["name"] for s in sess.list_sessions()]
    assert names == ["Gamma docs", "Beta auth bugfix", "Alpha refactor"]


def test_list_limit(tmp_path):
    _seed(tmp_path)
    assert len(sess.list_sessions(limit=2)) == 2
    assert len(sess.list_sessions(oldest_first=True, limit=1)) == 1


def test_list_includes_classification(tmp_path):
    _seed(tmp_path)
    cls = {s["name"]: s["classification"] for s in sess.list_sessions()}
    assert cls["Alpha refactor"] == "refactor"
    assert cls["Beta auth bugfix"] == "bugfix"


def test_search_by_tag(tmp_path):
    _seed(tmp_path)
    hits = sess.search_sessions("auth")
    assert [h["name"] for h in hits] == ["Beta auth bugfix"]


def test_search_by_classification(tmp_path):
    _seed(tmp_path)
    hits = sess.search_sessions("docs")
    assert [h["name"] for h in hits] == ["Gamma docs"]


def test_search_name_ranks_first(tmp_path):
    _seed(tmp_path)
    hits = sess.search_sessions("alpha")
    assert hits[0]["name"] == "Alpha refactor"


def test_search_empty_returns_recent(tmp_path):
    _seed(tmp_path)
    hits = sess.search_sessions("")
    assert hits[0]["name"] == "Gamma docs"  # newest first


def test_search_no_match(tmp_path):
    _seed(tmp_path)
    assert sess.search_sessions("zzzznomatch") == []
