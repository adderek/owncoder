"""HTTP UI folded tool row: argument segments (ui/tool_summary.py)."""
import json

from agent.ui import tool_summary as ts


def _texts(summary):
    return [s["t"] for s in summary["segs"]]


def test_run_argv_builtin_purpose_first_then_shell_joined_argv():
    args = json.dumps({"argv": ["ls", "-l"], "purpose": "List files in directory"})
    out = ts.build("run_argv", args)
    assert _texts(out) == ["List files in directory", "ls -l"]
    purpose, argv = out["segs"]
    assert purpose["w"] == argv["w"]  # equal share of the row
    assert purpose["st"] == "main"
    assert "k" not in argv


def test_default_puts_purpose_first_and_labels_rest():
    out = ts.build("grep", {"pattern": "foo", "path": "src", "purpose": "find foo"})
    assert _texts(out) == ["find foo", "foo", "src"]
    assert [s.get("k") for s in out["segs"]] == [None, "pattern", "path"]


def test_min_never_exceeds_text_length():
    out = ts.build("run_argv", {"argv": ["ls"]})
    assert out["segs"][0]["min"] == 2


def test_tool_override_replaces_fields_and_keeps_default_name():
    spec = {"default": {"name": {"hide_below": 80}},
            "edit_file": {"fields": [{"key": "path|file_path", "w": 3, "fmt": "path"},
                                     {"key": "content", "fmt": "hide"}]}}
    out = ts.build("edit_file", {"file_path": "a/b.py", "content": "x\ny"}, spec)
    assert _texts(out) == ["a/b.py"]
    assert out["segs"][0]["fmt"] == "path"
    assert out["hide_name"] == 80


def test_hidden_field_keeps_arg_out_of_wildcard():
    spec = {"default": {"fields": [{"key": "*"}, {"key": "secret", "fmt": "hide"}]}}
    out = ts.build("t", {"a": 1, "secret": "s"}, spec)
    assert _texts(out) == ["1"]


def test_auto_format_collapses_multiline_and_counts_long_text():
    out = ts.build("t", {"a": "one\ntwo", "b": "1\n2\n3\n4"},
                   {"default": {"fields": [{"key": "*"}]}})
    assert _texts(out) == ["one ⏎ two", "4 lines"]


def test_prefix_and_max_chars():
    spec = {"t": {"fields": [{"key": "q", "prefix": "? ", "max_chars": 5}]}}
    assert _texts(ts.build("t", {"q": "abcdefgh"}, spec)) == ["? abcd…"]


def test_unparseable_args_fall_back_to_none():
    assert ts.build("t", "{not json") is None
    assert ts.build("t", "[1, 2]") is None


def test_bad_config_does_not_raise():
    assert ts.build("t", {"a": 1}, {"default": {"fields": [{"key": "a", "w": "x"}]}}) is None


def test_builtin_not_mutated_between_calls():
    before = json.dumps(ts.BUILTIN, sort_keys=True)
    ts.build("run_argv", {"argv": ["x"], "purpose": "p", "cwd": "/"})
    assert json.dumps(ts.BUILTIN, sort_keys=True) == before


def test_summary_uses_configured_provider():
    ts.configure(lambda: {"t": {"fields": [{"key": "z"}]}})
    try:
        assert _texts(ts.summary("t", {"z": "hit", "y": "no"})) == ["hit"]
    finally:
        ts.configure(lambda: None)


def test_ui_config_has_tool_summary_dict():
    from agent.config.models import UIConfig
    assert UIConfig().tool_summary == {}
