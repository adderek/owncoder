"""A spawn_agents worker that runs out of time still hands back what it had."""
from __future__ import annotations

import asyncio

import agent.core.turn as turn_mod
from agent.config import Config
from agent.config.models import ModelEntry
from agent.tools.parallel import main as par


def _config(tmp_path):
    c = Config()
    c.tools.working_dir = str(tmp_path)
    c.tools.agent_dir = str(tmp_path / ".agent")
    e = ModelEntry()
    e.base_url, e.model = "http://127.0.0.1:1/v1", "fake"
    c.model_entries = {"w": e}
    return c


def test_partial_of_reports_last_text_and_tools():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "content": "..."},
        {"role": "assistant", "content": "found the bug in a.py"},
    ]
    assert par._partial_of(msgs, 2) == {
        "partial": "found the bug in a.py", "tool_calls_done": ["read_file"]}


def test_partial_of_is_empty_when_nothing_happened():
    seeded = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    assert par._partial_of(seeded, 2) == {}
    assert par._partial_of([], 2) == {}


async def test_timeout_keeps_the_checkpointed_work(tmp_path, monkeypatch):
    async def slow_turn(messages, partial_sink=None, **_kw):
        partial_sink[:] = messages + [{"role": "assistant", "content": "half an answer"}]
        await asyncio.sleep(30)

    monkeypatch.setattr(turn_mod, "run_turn", slow_turn)
    res = await par._run_worker("task", "w", _config(tmp_path), None, 1, set())
    assert res["output"] is None
    assert res["error"] == "timeout after 1s"
    assert res["partial"] == "half an answer"
    assert res["tool_calls_done"] == []


async def test_finished_worker_result_is_unchanged(tmp_path, monkeypatch):
    async def quick_turn(messages, **_kw):
        return "done", messages

    monkeypatch.setattr(turn_mod, "run_turn", quick_turn)
    res = await par._run_worker("task", "w", _config(tmp_path), None, 5, set())
    assert res == {"model": "w", "output": "done", "error": None, "tokens": {}}


# ── output_schema ────────────────────────────────────────────────────────────

from agent.tools.parallel import schema as sch  # noqa: E402

FINDINGS = {
    "type": "object",
    "required": ["file", "severity"],
    "properties": {
        "file": {"type": "string"},
        "severity": {"enum": ["low", "high"]},
        "lines": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    },
}


class TestSchema:
    def test_conforming_value(self):
        assert sch.validate({"file": "a.py", "severity": "high", "lines": [3]}, FINDINGS) == []

    def test_errors_name_the_path(self):
        errs = sch.validate({"severity": "meh", "lines": [1, "x"]}, FINDINGS)
        assert "$: missing required field 'file'" in errs
        assert any(e.startswith("$.severity:") for e in errs)
        assert "$.lines[1]: expected integer, got string" in errs

    def test_bool_is_not_a_number_and_unknown_keywords_are_ignored(self):
        assert sch.validate(True, {"type": "integer"}) != []
        assert sch.validate(1.0, {"type": "integer"}) == []
        assert sch.validate("x", {"type": "string", "pattern": "^y$", "format": "uri"}) == []

    def test_additional_properties_false(self):
        s = {"type": "object", "properties": {"a": {}}, "additionalProperties": False}
        assert sch.validate({"a": 1, "b": 2}, s) == ["$: unexpected field 'b'"]

    def test_parse_accepts_fenced_and_embedded_json(self):
        good = '{"file": "a.py", "severity": "low"}'
        for text in (good, f"```json\n{good}\n```", f"Here it is: {good} — done."):
            data, why = sch.parse(text, FINDINGS)
            assert why is None and data["file"] == "a.py"

    def test_parse_skips_a_nonconforming_value_before_the_real_one(self):
        data, why = sch.parse('see {"note": 1} then {"file": "b", "severity": "high"}', FINDINGS)
        assert why is None and data["file"] == "b"

    def test_parse_reports_why(self):
        assert sch.parse("", FINDINGS) == (None, "empty answer")
        assert sch.parse("I found nothing.", FINDINGS)[1] == "no JSON value found in the answer"
        assert "missing required field 'severity'" in sch.parse('{"file": "a"}', FINDINGS)[1]


def _scripted(monkeypatch, answers):
    """run_turn fake that replies with *answers* in order; records each call."""
    calls = []

    async def turn(messages, excluded_tools=None, **_kw):
        calls.append({"messages": messages, "excluded": excluded_tools})
        return answers[len(calls) - 1], messages

    monkeypatch.setattr(turn_mod, "run_turn", turn)
    return calls


async def test_schema_answer_is_parsed_into_data(tmp_path, monkeypatch):
    calls = _scripted(monkeypatch, ['{"file": "a.py", "severity": "high"}'])
    res = await par._run_worker("find it", "w", _config(tmp_path), None, 5, set(),
                                output_schema=FINDINGS)
    assert res["data"] == {"file": "a.py", "severity": "high"}
    assert "schema_error" not in res and len(calls) == 1
    assert "JSON Schema" in calls[0]["messages"][-1]["content"]


async def test_schema_miss_gets_one_toolless_repair_round(tmp_path, monkeypatch):
    calls = _scripted(monkeypatch, ["It is in a.py, quite severe.",
                                    '{"file": "a.py", "severity": "high"}'])
    res = await par._run_worker("find it", "w", _config(tmp_path), None, 5, set(),
                                output_schema=FINDINGS)
    assert res["data"]["file"] == "a.py" and res["error"] is None
    assert len(calls) == 2
    tail = calls[1]["messages"][-2:]
    assert tail[0] == {"role": "assistant", "content": "It is in a.py, quite severe."}
    assert "no JSON value found" in tail[1]["content"]
    assert "read_file" in calls[1]["excluded"]


async def test_schema_miss_twice_is_reported_with_the_original_text(tmp_path, monkeypatch):
    calls = _scripted(monkeypatch, ["prose", '{"file": 1}'])
    res = await par._run_worker("find it", "w", _config(tmp_path), None, 5, set(),
                                output_schema=FINDINGS)
    assert len(calls) == 2
    assert res["output"] == "prose" and "data" not in res
    assert "expected string, got integer" in res["schema_error"]
