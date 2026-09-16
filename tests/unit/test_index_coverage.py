"""The "# Index coverage" block (agent/core/index_coverage.py)."""
from __future__ import annotations

from agent.core.index_coverage import (
    NO_INDEX_MESSAGE,
    coverage_map,
    coverage_message,
    format_coverage,
)


class _Store:
    def __init__(self, paths):
        self._paths = paths

    def list_paths(self):
        return self._paths


def test_coverage_message_lists_indexed_areas():
    store = _Store(["/w/collect.py", "/w/agent/core/turn.py", "/w/agent/cli/main.py"])
    msg = coverage_message(store, "/w")
    assert "# Index coverage" in msg
    assert "agent/: 2 file(s)" in msg
    assert "(repository root): 1 file(s)" in msg


def test_no_index_message_names_only_real_tools():
    # It used to recommend shell_exec, which is not a registered tool.
    assert "shell_exec" not in NO_INDEX_MESSAGE
    assert "run_argv" in NO_INDEX_MESSAGE
    assert "grep_code" in NO_INDEX_MESSAGE


def test_empty_store_falls_back_to_no_index_message():
    assert coverage_message(_Store([]), "/w") == NO_INDEX_MESSAGE
    assert coverage_message(None, "/w") == NO_INDEX_MESSAGE


def test_broken_store_does_not_raise():
    class _Bad:
        def list_paths(self):
            raise RuntimeError("db gone")

    assert coverage_map(_Bad(), "/w") == {}
    assert coverage_message(_Bad(), "/w") == NO_INDEX_MESSAGE


def test_paths_outside_working_dir_count_as_root():
    assert coverage_map(_Store(["/elsewhere/x.py"]), "/w") == {".": 1}


def test_format_is_stable_for_empty_input():
    assert format_coverage({}).startswith("# Index coverage")
