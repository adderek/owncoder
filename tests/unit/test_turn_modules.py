"""Direct tests for the modules split out of core/turn.py in S7.

run_turn's behaviour is already covered end-to-end (test_turn_guards,
test_loop_guard, test_verify_loop, …). These tests pin the extracted pieces
individually, so a future change to one of them fails here with a readable
assertion instead of somewhere deep inside a fake-LLM turn.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.config import Config
from agent.core import turn_batch, turn_errors, turn_guards, turn_setup


class FakeFn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeCall:
    def __init__(self, name, arguments="{}", id="call-1"):
        self.function = FakeFn(name, arguments)
        self.id = id


def _schema(name):
    return {"type": "function", "function": {"name": name, "description": name}}


# --------------------------------------------------------------------------
# turn_setup.select_tools
# --------------------------------------------------------------------------

def test_excluded_tools_are_not_offered():
    cfg = Config()
    tools, refresh, _ = turn_setup.select_tools(
        [_schema("read_file"), _schema("run_argv")], cfg, {"run_argv"},
    )
    assert [t["function"]["name"] for t in tools] == ["read_file"]
    assert refresh is None          # discovery off by default


def test_find_tools_dropped_when_discovery_is_off():
    cfg = Config()
    cfg.tool_discovery.enabled = False
    tools, refresh, _ = turn_setup.select_tools(
        [_schema("read_file"), _schema("find_tools")], cfg,
    )
    assert [t["function"]["name"] for t in tools] == ["read_file"]
    assert refresh is None


def test_signal_tools_dropped_when_turn_signals_disabled():
    from agent.tools.turn_signals import SIGNAL_TOOL_NAMES

    cfg = Config()
    cfg.turn_signals.enabled = False
    signal = sorted(SIGNAL_TOOL_NAMES)[0]
    tools, _, _ = turn_setup.select_tools([_schema("read_file"), _schema(signal)], cfg)
    assert [t["function"]["name"] for t in tools] == ["read_file"]


def test_discovery_returns_a_refresh_that_reflects_activation():
    cfg = Config()
    cfg.tool_discovery.enabled = True
    from agent.core import tool_discovery as td

    catalog = [_schema("read_file"), _schema("find_tools"), _schema("web_search")]
    tools, refresh, _ = turn_setup.select_tools(catalog, cfg)
    assert refresh is not None
    before = {t["function"]["name"] for t in tools}
    td.activate(["web_search"])
    after = {t["function"]["name"] for t in refresh()}
    # Whatever the core set is, activation can only widen it.
    assert before <= after
    assert "web_search" in after
    td.reset_active()


# --------------------------------------------------------------------------
# turn_setup.normalize_api_messages
# --------------------------------------------------------------------------

def test_internal_keys_are_stripped_and_reasoning_surfaced():
    out = turn_setup.normalize_api_messages([
        {"role": "user", "content": "hi", "_nudged": True},
        {"role": "assistant", "content": "yo", "_reasoning_content": "because"},
        {"role": "user", "content": "again"},
    ])
    assert "_nudged" not in out[0]
    assert out[1]["reasoning_content"] == "because"
    assert "_reasoning_content" not in out[1]


def test_all_system_messages_merge_into_one_leading_message():
    out = turn_setup.normalize_api_messages([
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation note"},
        {"role": "user", "content": "bye"},
    ])
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "first\n\nmid-conversation note"
    assert [m["role"] for m in out[1:]] == ["user", "user"]


def test_trailing_assistant_prefill_is_stripped():
    out = turn_setup.normalize_api_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "partial"},
    ])
    assert [m["role"] for m in out] == ["user"]


def test_trailing_assistant_with_tool_calls_is_kept():
    out = turn_setup.normalize_api_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
    ])
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_reasoning_content_backfilled_for_thinking_sessions():
    out = turn_setup.normalize_api_messages([
        {"role": "assistant", "content": "a", "_reasoning_content": "why"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "b", "tool_calls": [{"id": "x"}]},
    ])
    assistants = [m for m in out if m["role"] == "assistant"]
    assert all("reasoning_content" in m for m in assistants)


# --------------------------------------------------------------------------
# turn_batch
# --------------------------------------------------------------------------

def test_arguments_that_are_not_objects_degrade_to_empty_dict():
    calls = [FakeCall("read_file", '{"path": "a"}'), FakeCall("read_file", "[1,2]"),
             FakeCall("read_file", "not json")]
    parsed, purposes = turn_batch.parse_arguments(calls, compaction_on=False)
    assert parsed == [{"path": "a"}, {}, {}]
    assert purposes == ["", "", ""]


def test_purpose_only_read_when_compaction_is_on():
    calls = [FakeCall("read_file", '{"path": "a", "purpose": "find the bug"}')]
    _, off = turn_batch.parse_arguments(calls, compaction_on=False)
    _, on = turn_batch.parse_arguments(calls, compaction_on=True)
    assert off == [""]
    assert on == ["find the bug"]


def test_signature_is_argument_order_independent():
    a = FakeCall("read_file", '{"path": "x", "start_line": 1}')
    b = FakeCall("read_file", '{"start_line": 1, "path": "x"}')
    assert turn_batch.call_signature(a) == turn_batch.call_signature(b)


def test_duplicate_calls_execute_once_and_share_the_result():
    calls = [FakeCall("read_file", '{"path": "a"}', id="1"),
             FakeCall("read_file", '{"path": "a"}', id="2"),
             FakeCall("read_file", '{"path": "b"}', id="3")]
    ran: list[str] = []

    async def execute(tc, config):
        ran.append(tc.id)
        return '{"content": "%s"}' % tc.function.arguments

    parsed, purposes = turn_batch.parse_arguments(calls, False)
    results, raw, durations = asyncio.run(turn_batch.execute_batch(
        calls, parsed, purposes, Config(), None, execute=execute,
    ))
    assert ran == ["1", "3"]                  # the dupe never ran
    assert results[0] == results[1]           # …but it has a result
    assert results[2] != results[0]
    assert raw == results                     # no compaction → identical
    assert set(durations) == {0, 1, 2}


def test_compaction_replaces_results_but_not_raw_results():
    calls = [FakeCall("read_file", '{"path": "a"}')]

    async def execute(tc, config):
        return "the full original output"

    async def fake_compact(tc, parsed_arg, purpose, raw, config, client, side_log, turn_index):
        return "short"

    cfg = Config()
    cfg.tool_compaction.enabled = True
    parsed, purposes = turn_batch.parse_arguments(calls, True)

    original = turn_batch.compact_tool_result
    turn_batch.compact_tool_result = fake_compact
    try:
        results, raw, _ = asyncio.run(turn_batch.execute_batch(
            calls, parsed, purposes, cfg, None, execute=execute, compaction_on=True,
        ))
    finally:
        turn_batch.compact_tool_result = original
    assert results == ["short"]
    assert raw == ["the full original output"]


# --------------------------------------------------------------------------
# turn_guards.observe_tool_calls
# --------------------------------------------------------------------------

def test_observe_reports_nothing_while_calls_differ():
    from agent.core.loop_detector import LoopDetector

    det = LoopDetector(window=10, threshold=3)
    for path in ("a", "b", "c"):
        assert turn_guards.observe_tool_calls(
            det, [FakeCall("read_file", '{"path": "%s"}' % path)]) == []


def test_observe_reports_the_repeated_call_at_the_threshold():
    from agent.core.loop_detector import LoopDetector

    det = LoopDetector(window=10, threshold=3)
    call = FakeCall("read_file", '{"path": "a"}')
    triggered = []
    for _ in range(3):
        triggered = turn_guards.observe_tool_calls(det, [call])
    assert triggered, "the third identical call should trip the detector"
    name, sig, count, args = triggered[0]
    assert name == "read_file"
    assert count >= 3
    assert args == '{"path": "a"}'


# --------------------------------------------------------------------------
# turn_errors
# --------------------------------------------------------------------------

def test_failover_prefers_a_cloud_peer_over_degrading_to_local(monkeypatch):
    from agent.core import model_routing

    calls: list[str] = []
    monkeypatch.setattr(model_routing, "failover_to_peer",
                        lambda c: calls.append("peer") or "peer-client")
    monkeypatch.setattr(model_routing, "failover_to_local",
                        lambda c: calls.append("local") or "local-client")
    monkeypatch.setattr(model_routing, "failover_to_alternative",
                        lambda c: calls.append("alt") or "alt-client")
    assert turn_errors.try_failover(Config()) == "peer-client"
    assert calls == ["peer"]


def test_failover_falls_through_peer_then_local_then_alternative(monkeypatch):
    from agent.core import model_routing

    calls: list[str] = []

    def _none(name):
        def _f(c):
            calls.append(name)
            return None
        return _f

    monkeypatch.setattr(model_routing, "failover_to_peer", _none("peer"))
    monkeypatch.setattr(model_routing, "failover_to_local", _none("local"))
    monkeypatch.setattr(model_routing, "failover_to_alternative", _none("alt"))
    assert turn_errors.try_failover(Config()) is None
    assert calls == ["peer", "local", "alt"]


def test_no_usable_model_error_lists_disabled_entries_to_enable(monkeypatch):
    from agent.core import model_control

    cfg = Config()
    cfg.model_entries = {"gpu-local": object(), "cloud-free": object()}
    monkeypatch.setattr(model_control, "is_disabled",
                        lambda c, name: name == "cloud-free")
    err = turn_errors.no_usable_model_error(cfg, RuntimeError("boom"))
    assert isinstance(err, turn_errors.NoUsableModelError)
    assert err.candidates == ["cloud-free"]
    assert "cloud-free" in str(err)
    assert isinstance(err.cause, RuntimeError)


@pytest.mark.parametrize("outcome", ["success", "failure", "rate_limited"])
def test_recording_an_outcome_never_raises(monkeypatch, outcome):
    """Reliability stats are telemetry — a broken store must not fail a turn."""
    import agent.metrics.model_stats as ms

    def _boom(config):
        raise RuntimeError("stats backend down")

    monkeypatch.setattr(ms, "resolve_entry_name", _boom)
    turn_errors.record_model_outcome(Config(), outcome)      # no exception


def test_marking_a_cooldown_never_raises(monkeypatch):
    from agent.config import model_probe

    def _boom(*a, **kw):
        raise RuntimeError("probe store down")

    monkeypatch.setattr(model_probe, "mark_rate_limited", _boom)
    turn_errors.mark_endpoint_cooldown(Config())
    turn_errors.mark_endpoint_cooldown(Config(), 300.0)
