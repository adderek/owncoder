"""Tests for the 429 retry path, the error-streak guard, and per-tool call caps.

These cover the failure mode of a rate-limited backend: the model rephrases the
same failing call each iteration, so the exact-signature loop detector never
fires — the guards here must stop the turn instead.
"""
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from agent.config import Config
from agent.core.loop_detector import LoopDetector
from agent.core.turn import run_turn


def _fake_tool_call(name: str, args: dict, call_id: str = "call_0"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class _StubChoice:
    def __init__(self, tool_calls=None, content=None, finish_reason="tool_calls"):
        self.message = SimpleNamespace(content=content, tool_calls=tool_calls)
        self.finish_reason = finish_reason


class _StubResponse:
    def __init__(self, choice):
        self.choices = [choice]
        self.usage = None


class _ScriptedCompletions:
    """Yields the next item per create() call; exceptions are raised."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def create(self, **kw):
        self.calls += 1
        item = self._script.pop(0) if self._script else self._script_default()
        if isinstance(item, Exception):
            raise item
        return item

    @staticmethod
    def _script_default():
        return _StubResponse(_StubChoice(content="done", finish_reason="stop"))


class _ScriptedClient:
    def __init__(self, script):
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(script))


def _rate_limit_error(retry_after: str = "0") -> RateLimitError:
    resp = httpx.Response(
        429,
        request=httpx.Request("POST", "http://test/v1/chat/completions"),
        headers={"retry-after": retry_after},
    )
    return RateLimitError("rate limited", response=resp, body=None)


class _VaryingToolCompletions:
    """Always requests the same tool with different args (defeats sig matching)."""

    def __init__(self, name: str):
        self._name = name
        self.calls = 0

    async def create(self, **kw):
        self.calls += 1
        tc = _fake_tool_call(self._name, {"query": f"q{self.calls}"}, f"call_{self.calls}")
        return _StubResponse(_StubChoice(tool_calls=[tc]))


def _base_cfg() -> Config:
    cfg = Config()
    cfg.llm.max_iterations = 50
    cfg.confidence_guard.enabled = False
    return cfg


def _no_sleep(monkeypatch):
    import agent.core.turn as turn_mod

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(turn_mod.asyncio, "sleep", _instant)


def _stub_tools(monkeypatch, execute):
    import agent.core.turn as turn_mod
    monkeypatch.setattr(turn_mod, "execute_tool", execute)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])


# ── 429 handling ────────────────────────────────────────────────────────────

async def test_rate_limit_retries_then_succeeds(monkeypatch):
    cfg = _base_cfg()
    cfg.llm.rate_limit_retries = 3
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch, None)

    ok = _StubResponse(_StubChoice(content="done", finish_reason="stop"))
    client = _ScriptedClient([_rate_limit_error(), _rate_limit_error(), ok])
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, _ = await run_turn(messages, cfg, client)
    assert response == "done"
    # Two 429s were absorbed, then the scripted success (later calls may come
    # from the turn's own nudge/justify follow-ups).
    assert client.chat.completions.calls >= 3


async def test_rate_limit_exhausted_raises_without_failover(monkeypatch):
    cfg = _base_cfg()
    cfg.llm.rate_limit_retries = 1
    assert cfg.failover.enabled is False
    _no_sleep(monkeypatch)
    _stub_tools(monkeypatch, None)

    client = _ScriptedClient([_rate_limit_error(), _rate_limit_error()])
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    # No self-hosted model to degrade to → surfaced as a recoverable
    # NoUsableModelError (retry / enable a model), wrapping the original 429.
    from agent.core.turn import NoUsableModelError
    with pytest.raises(NoUsableModelError) as ei:
        await run_turn(messages, cfg, client)
    assert isinstance(ei.value.cause, RateLimitError)


# ── error-streak guard ──────────────────────────────────────────────────────

async def test_error_streak_stops_turn(monkeypatch):
    cfg = _base_cfg()
    cfg.loop_guard.error_streak_threshold = 3
    # Varying args per call: the signature-based detector must not be what stops us.
    cfg.loop_guard.per_tool_call_cap = {}

    async def _always_error(tc, config=None):
        return json.dumps({"error": "backend rate limited (429)"})

    _stub_tools(monkeypatch, _always_error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_VaryingToolCompletions("web_search")))
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, _ = await run_turn(messages, cfg, client)
    assert "error guard" in response
    assert client.chat.completions.calls == 3


async def test_error_streak_resets_on_success(monkeypatch):
    cfg = _base_cfg()
    cfg.loop_guard.error_streak_threshold = 2
    cfg.loop_guard.per_tool_call_cap = {}
    cfg.llm.max_iterations = 6

    state = {"n": 0}

    async def _alternating(tc, config=None):
        state["n"] += 1
        if state["n"] % 2:
            return json.dumps({"error": "boom"})
        return json.dumps({"ok": True})

    _stub_tools(monkeypatch, _alternating)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_VaryingToolCompletions("web_search")))
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, _ = await run_turn(messages, cfg, client)
    # Streak never reaches 2 consecutively — the iteration cap ends the turn instead.
    assert "error guard" not in response
    assert "iteration limit" in response


# ── per-tool call cap ───────────────────────────────────────────────────────

def test_name_cap_triggers_regardless_of_args():
    d = LoopDetector(window=10, threshold=99, per_tool_call_cap={"web_search": 3})
    for i in range(2):
        assert d.name_capped("web_search", d.observe_name("web_search")) is False
    assert d.name_capped("web_search", d.observe_name("web_search")) is True
    # Uncapped tools never trigger.
    assert d.name_capped("read_file", d.observe_name("read_file")) is False


def test_name_cap_acknowledge_silences():
    d = LoopDetector(window=10, threshold=99, per_tool_call_cap={"web_search": 2})
    d.observe_name("web_search")
    assert d.name_capped("web_search", d.observe_name("web_search")) is True
    d.acknowledge("name:web_search")
    assert d.name_capped("web_search", d.observe_name("web_search")) is False


async def test_name_cap_stops_rephrased_calls(monkeypatch):
    cfg = _base_cfg()
    cfg.loop_guard.error_streak_threshold = 0  # isolate the name cap
    cfg.loop_guard.per_tool_call_cap = {"web_search": 4}

    async def _ok(tc, config=None):
        return json.dumps({"results": ["x"]})

    _stub_tools(monkeypatch, _ok)
    client = SimpleNamespace(chat=SimpleNamespace(completions=_VaryingToolCompletions("web_search")))
    messages = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    response, _ = await run_turn(messages, cfg, client)
    assert "loop guard" in response
    assert client.chat.completions.calls == 4


# ── 429 endpoint cooldown ───────────────────────────────────────────────────

def test_rate_limit_cooldown_marks_entry_unavailable(monkeypatch):
    from agent.config import model_probe as mp
    mp.clear_availability_cache()
    entry = SimpleNamespace(base_url="http://rl", api_key="k", model="m1")
    monkeypatch.setattr(mp, "list_endpoint_models", lambda *a, **k: {"m1"})
    assert mp.entry_available(entry) is True
    mp.mark_rate_limited("http://rl", "m1", cooldown_s=60)
    assert mp.is_rate_limited("http://rl", "m1") is True
    assert mp.entry_available(entry) is False
    # Other models on the same endpoint are unaffected.
    assert mp.is_rate_limited("http://rl", "m2") is False
    mp.clear_availability_cache()
    assert mp.is_rate_limited("http://rl", "m1") is False


# ── session model enable/disable ────────────────────────────────────────────

def test_model_disable_enable_cycle():
    from agent.core.model_control import set_model_enabled, is_disabled
    cfg = Config()
    cfg.model_entries = {"m1": SimpleNamespace(base_url="http://a", model="x", tags=[], dimensions=0)}
    cfg.model_roles = {"default": "m1"}
    ok, msg = set_model_enabled(cfg, "nope", False)
    assert ok is False
    ok, msg = set_model_enabled(cfg, "m1", False)
    assert ok and is_disabled(cfg, "m1")
    ok, msg = set_model_enabled(cfg, "m1", True)
    assert ok and not is_disabled(cfg, "m1")


def test_disabled_entry_skipped_by_ladder(monkeypatch):
    from agent.core.model_tier import build_ladder
    from agent.core.model_control import set_model_enabled
    cfg = Config()
    e1 = SimpleNamespace(base_url="http://a", model="x", tags=[], dimensions=0, local=True)
    e2 = SimpleNamespace(base_url="http://b", model="y", tags=[], dimensions=0, local=True)
    cfg.model_entries = {"m1": e1, "m2": e2}
    cfg.model_roles = {"default": "m1"}
    ladder = build_ladder(cfg, check_available=False)
    assert {n for n, _ in ladder} == {"m1", "m2"}
    set_model_enabled(cfg, "m2", False)
    ladder = build_ladder(cfg, check_available=False)
    assert {n for n, _ in ladder} == {"m1"}


def test_call_kwargs_clamps_max_tokens():
    from agent.core.prompts import _build_call_kwargs
    cfg = Config()
    cfg.llm.ctx_window = 262144
    cfg.llm.max_output_tokens = 262144  # misconfigured: output == whole window
    kw = _build_call_kwargs(cfg)
    assert kw["max_tokens"] == 131072
    cfg.llm.max_output_tokens = 8192
    assert _build_call_kwargs(cfg)["max_tokens"] == 8192
