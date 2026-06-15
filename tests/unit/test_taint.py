"""Tests for cross-file taint reachability (agent.security.taint)."""
from __future__ import annotations

from agent.security import taint
from agent.security.taint import Func


def test_direct_source_to_sink_cross_file():
    funcs = {
        "read_input": Func("read_input", "io.c", 10, calls={"handle"}, is_source=True),
        "handle":     Func("handle", "logic.c", 5, calls={"run_cmd"}),
        "run_cmd":    Func("run_cmd", "exec.c", 3, calls=set(), is_sink=True),
    }
    paths = taint.find_taint_paths(funcs)
    assert len(paths) == 1
    p = paths[0]
    assert p["cross_file"] is True
    assert "read_input" in p["path"] and "run_cmd" in p["path"]
    assert p["hops"] == 2


def test_no_path_when_disconnected():
    funcs = {
        "src": Func("src", "a.c", 1, calls={"nothing"}, is_source=True),
        "sink": Func("sink", "b.c", 1, calls=set(), is_sink=True),
    }
    assert taint.find_taint_paths(funcs) == []


def test_source_that_is_also_sink_is_not_self_path():
    # A function both reading input and doing a sink op should not report a
    # zero-hop path to itself (node != src guard).
    funcs = {"both": Func("both", "a.c", 1, calls=set(), is_source=True, is_sink=True)}
    assert taint.find_taint_paths(funcs) == []


def test_cross_file_sorted_first():
    funcs = {
        "s1": Func("s1", "a.c", 1, calls={"k1"}, is_source=True),
        "k1": Func("k1", "a.c", 2, calls=set(), is_sink=True),       # same-file path
        "s2": Func("s2", "a.c", 3, calls={"k2"}, is_source=True),
        "k2": Func("k2", "b.c", 4, calls=set(), is_sink=True),       # cross-file path
    }
    paths = taint.find_taint_paths(funcs)
    assert len(paths) == 2
    assert paths[0]["cross_file"] is True   # cross-file first


def test_classification_regex():
    assert taint._SOURCE_RE.search("x = getenv(\"PATH\")")
    assert taint._SINK_RE.search("system(cmd)")
    assert taint._SINK_RE.search("memcpy(d, s, n)")
    assert not taint._SINK_RE.search("printf(\"ok\")")
