"""Unit tests for agent.tool_compactor.compact_result."""
from __future__ import annotations

import pytest

from agent.config import Config
from agent.tool_compactor import compact_result
from agent._test_helpers import make_response, make_client


def _cfg() -> Config:
    c = Config()
    c.tool_compaction.enabled = True
    c.tool_compaction.min_length_to_compact = 10
    c.tool_compaction.base_url = ""  # reuse main client
    return c


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_skips_short_results():
    cfg = _cfg()
    cfg.tool_compaction.min_length_to_compact = 500
    client = make_client(make_response(content="should not be called"))
    out, info = await compact_result("search_code", {}, "p", "tiny", cfg, client)
    assert info["skipped"] and info["reason"] == "too_short"
    assert out == "tiny"
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_no_shrink_keeps_original():
    cfg = _cfg()
    raw = "x" * 100
    client = make_client(make_response(content="y" * 200))  # longer than original
    out, info = await compact_result("search_code", {}, "p", raw, cfg, client)
    assert out == raw
    assert info["reason"] == "no_shrink"
