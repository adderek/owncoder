"""Just-in-time tool-routing hints (agent/core/tool_hints.py)."""
from __future__ import annotations

import pytest

from agent.core.tool_hints import (
    _READ_HINT_THRESHOLD,
    reset_tool_hints,
    tool_hints,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_tool_hints()
    yield
    reset_tool_hints()


def _read(path="scene.js"):
    return tool_hints("read_file", {"path": path}, {"content": "..."})


class TestRepeatedReads:
    def test_hint_after_threshold(self):
        for _ in range(_READ_HINT_THRESHOLD - 1):
            assert _read() == []
        hint = _read()
        assert hint and "grep_code" in hint[0]
        assert "3×" in hint[0]
        assert "outline" in hint[0]

    def test_fires_once_per_path(self):
        for _ in range(_READ_HINT_THRESHOLD):
            _read()
        assert _read() == []

    def test_paths_counted_separately(self):
        for _ in range(_READ_HINT_THRESHOLD):
            _read("a.py")
        for _ in range(_READ_HINT_THRESHOLD - 1):
            assert _read("b.py") == []
        assert _read("b.py")

    def test_reset_clears_state(self):
        for _ in range(_READ_HINT_THRESHOLD):
            _read()
        reset_tool_hints()
        assert _read() == []


class TestSearchRouting:
    def test_identifier_query_suggests_grep(self):
        h = tool_hints("search_code", {"query": "createHouse"}, {"results": [{"x": 1}]})
        assert h and "grep_code" in h[0]

    def test_concept_query_is_left_alone(self):
        h = tool_hints("search_code", {"query": "where is auth handled"}, {"results": [{"x": 1}]})
        assert h == []

    def test_empty_result_warns_against_concluding_absence(self):
        h = tool_hints("search_code", {"query": "where is auth handled"}, {"results": []})
        assert h and "does not mean the code is absent" in h[0]


class TestGrepRouting:
    def test_definition_pattern_suggests_find_symbol(self):
        h = tool_hints("grep_code", {"pattern": "def run_turn"}, {"results": [{"x": 1}]})
        assert h and "find_symbol" in h[0]

    def test_plain_pattern_is_left_alone(self):
        h = tool_hints("grep_code", {"pattern": "TIMEOUT_SECONDS"}, {"results": [{"x": 1}]})
        assert h == []

    def test_empty_result_suggests_narrowing(self):
        h = tool_hints("grep_code", {"pattern": "TIMEOUT_SECONDS"}, {"results": [], "count": 0})
        assert h and "fixed_string" in h[0]


class TestWholeFileRewrite:
    def test_large_rewrite_suggests_edit_file(self):
        h = tool_hints("write_file", {"path": "big.py"}, {"ok": "big.py", "replaced_lines": 900})
        assert h and "edit_file" in h[0]

    def test_new_file_is_left_alone(self):
        assert tool_hints("write_file", {"path": "new.py"}, {"ok": "new.py"}) == []

    def test_small_rewrite_is_left_alone(self):
        h = tool_hints("write_file", {"path": "s.py"}, {"ok": "s.py", "replaced_lines": 12})
        assert h == []


class TestSafety:
    def test_errors_never_hinted(self):
        assert tool_hints("read_file", {"path": "x"}, {"error": "nope"}) == []

    def test_non_dict_result_is_safe(self):
        assert tool_hints("read_file", {"path": "x"}, "oops") == []


def test_hints_are_appended_not_replaced():
    from agent.core.tool_calls import _maybe_add_tool_hints

    result = {"ok": "big.py", "replaced_lines": 900, "_hints": ["[refactor-hint] existing"]}
    out = _maybe_add_tool_hints("write_file", {"path": "big.py"}, result)
    assert len(out["_hints"]) == 2
    assert out["_hints"][0].startswith("[refactor-hint]")
    assert "edit_file" in out["_hints"][1]
    # Original result object untouched.
    assert len(result["_hints"]) == 1
