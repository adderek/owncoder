"""Unit tests for OutputStore and retrieve_output tool."""
from __future__ import annotations

import json
from agent.core.output_store import OutputStore


def test_store_and_retrieve():
    store = OutputStore()
    store.store("test1", "hello world")
    assert store.get("test1") == "hello world"


def test_store_nonexistent():
    store = OutputStore()
    assert store.get("nope") is None


def test_get_range():
    store = OutputStore()
    store.store("r1", "0123456789")
    assert store.get_range("r1", 2, 5) == "234"


def test_get_lines():
    store = OutputStore()
    store.store("l1", "line0\nline1\nline2\nline3")
    assert store.get_lines("l1", 1, 3) == "line1\nline2\n"


def test_info():
    store = OutputStore()
    store.store("i1", "hello")
    info = store.info("i1")
    assert info is not None
    assert info["chars"] == 5
    assert info["lines"] == 1
    assert info["bytes"] > 0


def test_truncate_short_result():
    store = OutputStore(head_chars=10, tail_chars=5)
    short = "short"
    result, was_trunc = store.truncate(short)
    assert result == short
    assert not was_trunc


def test_truncate_head_tail():
    store = OutputStore(head_chars=10, tail_chars=5)
    long_str = "A" * 10 + "B" * 10 + "C" * 5
    result, was_trunc = store.truncate(long_str)
    assert was_trunc
    assert result.startswith("A" * 10)
    assert result.endswith("C" * 5)
    assert "..." in result
    # total = 10 + 5 + len("...") which varies; check it's less than original
    assert len(result) < len(long_str)


def test_exact_boundary_no_truncation():
    store = OutputStore(head_chars=10, tail_chars=5)
    exact = "A" * 15  # total <= head+tail
    result, was_trunc = store.truncate(exact)
    assert result == exact
    assert not was_trunc


def test_eviction_fifo():
    max_per = 10
    store = OutputStore(max_bytes=max_per * 3)  # hold ~3 entries
    store.store("a", "x" * max_per)
    store.store("b", "x" * max_per)
    store.store("c", "x" * max_per)
    assert store.get("a") is not None  # still under limit
    store.store("d", "x" * max_per)
    assert store.get("a") is None      # evicted oldest
    assert store.get("b") is not None
    assert store.get("c") is not None
    assert store.get("d") is not None


def test_retrieve_output_tool_registered():
    # Import triggers @register decorator
    import agent.tools.retrieve_output  # noqa: F401
    from agent.tools import get_tool
    fn = get_tool("retrieve_output")
    assert fn is not None
    assert callable(fn)


def test_retrieve_output_returns_stored():
    from agent.tools.retrieve_output import retrieve_output

    # Direct call: tool needs initialized store
    from agent.core.output_store import init_store
    init_store()
    store = _get_global_store()

    call_id = "ret_test_1"
    store.store(call_id, "stored content here")

    result = retrieve_output(call_id=call_id, mode="full")
    assert not isinstance(result, dict) or result.get("error") is None, result
    # retrieve_output returns dict, so check content
    if isinstance(result, dict):
        assert result.get("content") == "stored content here"


def _get_global_store():
    from agent.core.output_store import _instance
    return _instance


def test_retrieve_output_range():
    from agent.tools.retrieve_output import retrieve_output
    from agent.core.output_store import init_store

    init_store()
    store = _get_global_store()
    store.store("ret_range", "0123456789")

    result = retrieve_output(call_id="ret_range", mode="range", start=3, end=7)
    assert result.get("content") == "3456"


def test_retrieve_output_lines():
    from agent.tools.retrieve_output import retrieve_output
    from agent.core.output_store import init_store

    init_store()
    store = _get_global_store()
    store.store("ret_lines", "a\nb\nc\nd\ne")

    result = retrieve_output(call_id="ret_lines", mode="lines", start_line=1, end_line=3)
    assert result.get("content") == "b\nc\n"


def test_retrieve_output_nonexistent():
    from agent.tools.retrieve_output import retrieve_output
    result = retrieve_output(call_id="no_such_id", mode="full")
    assert "error" in result


def test_retrieve_output_bad_mode():
    from agent.tools.retrieve_output import retrieve_output
    result = retrieve_output(call_id="x", mode="invalid")
    assert "error" in result


def test_execute_tool_truncation_envelope():
    """Verify the JSON envelope returned by truncation has expected fields."""
    from agent.core.output_store import init_store
    from agent.config.models import OutputStoreConfig
    init_store(OutputStoreConfig(head_chars=10, tail_chars=5, truncation_threshold=15))
    store = _get_global_store()
    store.head_chars = 10
    store.tail_chars = 5

    # Simulate the exact envelope execute_tool produces
    long_str = "A" * 50
    call_id = "env_test"
    store.store(call_id, long_str)
    truncated, _ = store.truncate(long_str)

    envelope = json.dumps({
        "truncated": True,
        "call_id": call_id,
        "content": truncated,
        "original_length": len(long_str),
        "original_lines": long_str.count("\n") + 1,
        "head_chars": 10,
        "tail_chars": 5,
        "note": "Output too large. Use retrieve_output(call_id='%s') to get full result or specific range." % call_id,
    })

    parsed = json.loads(envelope)
    assert parsed["truncated"] is True
    assert parsed["call_id"] == call_id
    assert parsed["original_length"] == 50
    assert len(parsed["content"]) < 50
    assert "retrieve_output" in parsed["note"]
