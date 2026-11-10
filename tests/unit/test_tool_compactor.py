"""Unit tests for agent.tool_compactor.compact_result."""
from __future__ import annotations

import pytest
from json import dumps as json_dumps

from agent.config import Config
from agent.tool_compactor import compact_result
from agent.tests._test_helpers import make_response, make_client


def _cfg() -> Config:
    c = Config()
    c.tool_compaction.enabled = True
    c.tool_compaction.min_length_to_compact = 10
    c.tool_compaction.base_url = ""  # reuse main client
    return c


async def test_braces_in_result_not_doubled_in_prompt():
    """Result is a format *value*, so its braces must reach the compactor
    verbatim — never doubled to {{ }} (which corrupts JSON results)."""
    cfg = _cfg()
    raw = '{"matches": ["a.py", "b.py"], "count": 2}' + " padding" * 10
    client = make_client(make_response(content="2 matches: a.py, b.py"))

    compacted, info = await compact_result(
        "search_code", {"q": "x"}, "find matches", raw, cfg, client,
    )

    assert not info["skipped"], info
    # Inspect the prompt actually sent to the compactor.
    sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert '{"matches"' in sent       # original braces preserved
    assert '{{' not in sent           # not double-escaped
    assert compacted == "2 matches: a.py, b.py"


async def test_skips_short_results():
    cfg = _cfg()
    cfg.tool_compaction.min_length_to_compact = 500
    client = make_client(make_response(content="should not be called"))
    out, info = await compact_result("search_code", {}, "p", "tiny", cfg, client)
    assert info["skipped"] and info["reason"] == "too_short"
    assert out == "tiny"
    client.chat.completions.create.assert_not_called()


async def test_no_shrink_keeps_original():
    cfg = _cfg()
    raw = "x" * 100
    client = make_client(make_response(content="y" * 200))  # longer than original
    out, info = await compact_result("search_code", {}, "p", raw, cfg, client)
    assert out == raw
    assert info["reason"] == "no_shrink"


async def test_bad_prompt_template_falls_back(tmp_path):
    """A custom prompt with literal braces (e.g. a JSON example) makes
    str.format raise. compact_result must fall back to the raw result rather
    than crash the turn."""
    cfg = _cfg()
    bad_prompt = tmp_path / "compact_prompt.txt"
    # Stray/literal braces a real prompt might contain: a JSON output example.
    bad_prompt.write_text(
        "Compact {result} for {purpose}. Example output: {\"k\": \"v\"}",
        encoding="utf-8",
    )
    cfg.tool_compaction.prompt_path = str(bad_prompt)
    client = make_client(make_response(content="should not be called"))

    raw = '{"matches": ["a.py"]}' + " padding" * 10
    out, info = await compact_result(
        "search_code", {"q": "x"}, "find matches", raw, cfg, client,
    )

    assert out == raw
    assert info["skipped"] is True
    assert info["reason"].startswith("bad_prompt:")
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("tool", ["edit_file", "write_file"])
async def test_edit_results_never_compacted_by_default(tool):
    """The `current` view is the agent's only post-edit copy of the file."""
    cfg = _cfg()
    raw = '{"ok": true, "current": {"cli.py": "82:    z = 1\\n83:        _common(z)"}}' + " " * 50
    client = make_client(make_response(content="Added cache subcommand to cli.py."))
    out, info = await compact_result(tool, {}, "add subcommand", raw, cfg, client)
    assert out == raw
    assert info["reason"] == "skip_tools"
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("key", ["returncode", "exit_code"])
async def test_failed_command_output_not_compacted(key):
    """pytest's file:line diagnosis must reach the agent verbatim."""
    cfg = _cfg()
    raw = json_dumps({"stdout": "cli.py:88 IndentationError: unexpected indent\n" * 5, key: 2})
    client = make_client(make_response(content="1 error during collection"))
    out, info = await compact_result("run_argv", {}, "run tests", raw, cfg, client)
    assert out == raw
    assert info["reason"] == "error"
    client.chat.completions.create.assert_not_called()


async def test_successful_command_still_compacted():
    cfg = _cfg()
    raw = json_dumps({"stdout": "." * 200 + "\n14 passed in 0.02s\n", "returncode": 0})
    client = make_client(make_response(content="14 passed"))
    out, info = await compact_result("run_argv", {}, "run tests", raw, cfg, client)
    assert out == "14 passed"
    assert not info["skipped"]


@pytest.mark.parametrize("verdict", [
    "No 8051-related plugins found in rizin output.",
    "objdump does not support 8051.",
    "not found",
    "```\nNo asm.8051 plugin found.\n```",
    "0 matches",
])
async def test_negative_verdict_returns_raw(verdict):
    """The compactor sees one command's output and cannot tell "absent" from
    "wrong command" — `rizin -L` (IO plugins) came back as "No 8051-related
    plugins found". A short absence claim must not replace the evidence."""
    cfg = _cfg()
    raw = "rw_  ar  Open ar/lib files\nrw_  ihex  Intel HEX\n" * 20
    client = make_client(make_response(content=verdict))
    out, info = await compact_result("run_argv", {}, "does rizin support 8051", raw, cfg, client)
    assert out == raw
    assert info["skipped"] and info["reason"] == "negative_verdict"


@pytest.mark.parametrize("extract", [
    "Found: rizin, sdcc, objdump",
    "3 matches: a.py, b.py, c.py",
    "tests/test_x.py:12 AssertionError: expected 3",
])
async def test_positive_extract_still_compacted(extract):
    cfg = _cfg()
    client = make_client(make_response(content=extract))
    out, info = await compact_result("run_argv", {}, "p", "x" * 3000, cfg, client)
    assert out == extract
    assert not info["skipped"]


async def test_long_extract_mentioning_absence_is_kept():
    """Only short verdicts are suspect; a detailed extract may note a gap."""
    cfg = _cfg()
    extract = "\n".join(f"mod{i}.py:{i} ok" for i in range(40)) + "\nmod99.py not found"
    client = make_client(make_response(content=extract))
    out, info = await compact_result("run_argv", {}, "p", "x" * 5000, cfg, client)
    assert out == extract and not info["skipped"]


def test_default_min_length_leaves_small_outputs_raw():
    assert Config().tool_compaction.min_length_to_compact >= 2000
