import asyncio
import json
from types import SimpleNamespace

import pytest

from agent.core.loop_detector import LoopDetector
from agent.core.turn import run_turn
from agent.core.turn_guards import patch_read_file_result, patch_edit_file_result
from agent.config import Config


def _fake_tool_call(name: str, args: dict | None = None):
    args = args or {}
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_signature_stable_for_equivalent_args():
    a = LoopDetector.signature("read_file", '{"path": "x.py", "n": 1}')
    b = LoopDetector.signature("read_file", '{"n": 1, "path": "x.py"}')
    assert a == b


def test_signature_differs_by_args():
    a = LoopDetector.signature("read_file", '{"path": "x.py"}')
    b = LoopDetector.signature("read_file", '{"path": "y.py"}')
    assert a != b


def test_triggers_after_threshold():
    d = LoopDetector(window=10, threshold=3)
    sig = LoopDetector.signature("grep", '{"q":"foo"}')
    assert d.triggered(sig, d.observe(sig)) is False
    assert d.triggered(sig, d.observe(sig)) is False
    assert d.triggered(sig, d.observe(sig)) is True


def test_acknowledge_silences_signature():
    d = LoopDetector(window=10, threshold=2)
    sig = LoopDetector.signature("ls", "{}")
    d.observe(sig); d.observe(sig)
    assert d.triggered(sig, 2) is True
    d.acknowledge(sig)
    assert d.triggered(sig, 5) is False


def test_per_tool_threshold_overrides_default():
    d = LoopDetector(window=20, threshold=3, per_tool_threshold={"list_files": 5})
    sig = LoopDetector.signature("list_files", '{"path": "."}')
    # First 4 observations should not trigger (override = 5)
    for _ in range(4):
        assert d.triggered(sig, d.observe(sig)) is False
    # 5th observation triggers
    assert d.triggered(sig, d.observe(sig)) is True


def test_per_tool_threshold_only_applies_to_named_tool():
    d = LoopDetector(window=20, threshold=3, per_tool_threshold={"list_files": 10})
    other = LoopDetector.signature("read_file", '{"path": "x.py"}')
    # read_file still uses default threshold=3
    d.observe(other); d.observe(other)
    assert d.triggered(other, 3) is True


def test_window_evicts_old_signatures():
    d = LoopDetector(window=3, threshold=2)
    sig = LoopDetector.signature("a", "{}")
    other = LoopDetector.signature("b", "{}")
    d.observe(sig)                      # buf: [a]
    for _ in range(3):
        d.observe(other)                # buf: [other,other,other]; sig evicted
    assert d.observe(sig) == 1          # only this latest occurrence remains


class _StubChoice:
    def __init__(self, tool_calls):
        self.message = SimpleNamespace(content=None, tool_calls=tool_calls)
        self.finish_reason = "tool_calls"


class _StubResponse:
    def __init__(self, tool_calls):
        self.choices = [_StubChoice(tool_calls)]
        self.usage = None


class _StubCompletions:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    async def create(self, **kw):
        # Always return the same tool call to force a loop
        return _StubResponse([_fake_tool_call("read_file", {"path": "x.py"}) for _ in range(1)])


class _StubClient:
    def __init__(self, tool_calls):
        self.chat = SimpleNamespace(completions=_StubCompletions(tool_calls))


@pytest.mark.asyncio
async def test_run_turn_stops_on_loop_with_no_callback(monkeypatch):
    cfg = Config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 50  # ensure loop guard, not iter cap, is what stops us

    # Stub the tool execution to avoid hitting the real registry
    async def _fake_execute(tc, config=None):
        return json.dumps({"ok": True})

    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(None)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, out_messages = await run_turn(messages, cfg, client)
    assert "loop guard" in response
    # Should have stopped before hitting max_iterations
    assert sum(1 for m in out_messages if m.get("role") == "tool") < cfg.llm.max_iterations


@pytest.mark.asyncio
async def test_run_turn_continues_when_callback_returns_true(monkeypatch):
    cfg = Config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 5  # bound the test

    async def _fake_execute(tc, config=None):
        return json.dumps({"ok": True})

    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])

    client = _StubClient(None)
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    calls = {"n": 0}

    def _allow(summary, count):
        calls["n"] += 1
        return True

    response, _ = await run_turn(messages, cfg, client, on_loop_detected=_allow)
    # Callback fired at least once; we still terminated via max_iterations note
    assert calls["n"] >= 1
    assert "iteration limit" in response or "loop guard" not in response


# ── read_file auto-advance ─────────────────────────────────────────────────


WARN = 3
STOP = 8


class TestReadAutoAdvance:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Point the file tools at an isolated working dir."""
        from agent.tools.files import setup as files_setup
        from agent.tools.rules import load_rules

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = str(tmp_path / ".agent")
        files_setup(cfg)
        load_rules(str(tmp_path))
        self.work = tmp_path

    def _make_file(self, name: str, n_lines: int) -> str:
        (self.work / name).write_text(
            "\n".join(f"line {i + 1}" for i in range(n_lines)) + "\n"
        )
        return name

    def _read_result(self, path: str, **kw) -> str:
        from agent.tools.files.read import read_file
        return json.dumps(read_file(path, **kw))

    def test_repeat_read_serves_next_block(self):
        path = self._make_file("big.txt", 600)
        tc = _fake_tool_call("read_file", {"path": path})
        result = self._read_result(path)  # head window (file > 500 lines)
        counts, adv = {}, {}

        r1, stop = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert stop is None
        assert r1 == result  # first read passes through untouched
        assert adv == {}

        r2, stop = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert stop is None
        parsed = json.loads(r2)
        assert "_auto_advanced" in parsed
        assert "line 201" in parsed["content"]
        assert adv[path] == 401

        r3, _ = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert "line 401" in json.loads(r3)["content"]
        assert adv[path] == 601

    def test_advance_past_eof_reports_all_lines_shown(self):
        path = self._make_file("mid.txt", 300)
        tc = _fake_tool_call("read_file", {"path": path, "start_line": 1, "end_line": 200})
        result = self._read_result(path, start_line=1, end_line=200)
        counts, adv = {}, {}

        patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        r2, _ = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert "line 201" in json.loads(r2)["content"]  # clamped 201-300 block

        r3, stop = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert stop is None
        parsed = json.loads(r3)
        assert parsed.get("end_of_file") is True
        assert "all 300 lines" in parsed["content"]

    def test_successful_edit_resets_advance_cursor(self):
        path = self._make_file("edited.txt", 600)
        read_tc = _fake_tool_call("read_file", {"path": path})
        result = self._read_result(path)
        counts, adv = {}, {}

        patch_read_file_result(read_tc, result, counts, WARN, STOP, adv)
        patch_read_file_result(read_tc, result, counts, WARN, STOP, adv)
        assert adv[path] == 401 and counts

        edit_tc = _fake_tool_call("edit_file", {"path": path})
        patch_edit_file_result(edit_tc, json.dumps({"ok": True}), counts, {}, 2, adv)
        assert adv == {}
        assert counts == {}

    def test_errored_read_does_not_advance(self):
        tc = _fake_tool_call("read_file", {"path": "missing.txt"})
        result = json.dumps({"error": "File not found: missing.txt"})
        counts, adv = {}, {}

        patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        r2, stop = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert stop is None
        assert r2 == result  # auto-advance read errored too → passthrough
        assert adv == {}

    def test_warn_not_injected_on_auto_advanced_result(self):
        path = self._make_file("warn.txt", 600)
        tc = _fake_tool_call("read_file", {"path": path})
        result = self._read_result(path)
        counts, adv = {}, {}

        for _ in range(2):
            patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        r3, _ = patch_read_file_result(tc, result, counts, WARN, STOP, adv)  # count=3
        parsed = json.loads(r3)
        assert "_auto_advanced" in parsed
        assert "_loop_warning" not in parsed

    def test_warn_still_injected_without_auto_advance(self):
        path = self._make_file("legacy.txt", 600)
        tc = _fake_tool_call("read_file", {"path": path})
        result = self._read_result(path)
        counts = {}

        for _ in range(2):
            patch_read_file_result(tc, result, counts, WARN, STOP, None)
        r3, _ = patch_read_file_result(tc, result, counts, WARN, STOP, None)
        assert "_loop_warning" in json.loads(r3)

    def test_stop_note_at_hard_ceiling(self):
        path = self._make_file("stop.txt", 600)
        tc = _fake_tool_call("read_file", {"path": path})
        result = self._read_result(path)
        counts, adv = {}, {}

        stop = None
        for _ in range(STOP):
            _, stop = patch_read_file_result(tc, result, counts, WARN, STOP, adv)
        assert stop is not None and "loop guard" in stop
