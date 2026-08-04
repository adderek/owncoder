"""Failure-triggered auto-tier escalation (core/turn.py + core/model_tier.py).

Covers the loop-guard and verify-fail mid-turn fast->strong escalation paths,
the once-per-turn guard shared with the confidence path, and the local-only
privacy gate in escalate_mid_turn.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.config import Config, ModelEntry
import agent.core.turn as turn_mod
import agent.core.model_tier as model_tier_mod
from agent.core.turn import run_turn
from agent.core.model_tier import escalate_mid_turn


# ── stub client plumbing (mirrors test_verify_loop.py) ────────────────────────

def _fake_tool_call(name: str, args: dict | None = None):
    args = args or {}
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _tool_call_response(name: str, args: dict | None = None):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=None, tool_calls=[_fake_tool_call(name, args)]),
        finish_reason="tool_calls",
    )
    return SimpleNamespace(choices=[choice], usage=None)


def _stop_response(content: str = "done"):
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=None),
        finish_reason="stop",
    )
    return SimpleNamespace(choices=[choice], usage=None)


class _RecordingCompletions:
    def __init__(self, responses, models):
        self._responses = iter(responses)
        self._models = models

    async def create(self, **kw):
        self._models.append(kw.get("model"))
        resp = next(self._responses, None)
        if resp is None:
            return _stop_response("done")
        return resp


class _RecordingClient:
    """Records the ``model`` kwarg of every create() call in *models*."""

    def __init__(self, responses, models):
        self.chat = SimpleNamespace(completions=_RecordingCompletions(list(responses), models))


def _base_messages():
    return [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]


async def _fake_execute_ok(tc, config=None):
    return json.dumps({"ok": True})


def _tiered_config(*, strong_local: bool = True) -> Config:
    """Config with a fast (local) and a strong entry, auto-tier enabled."""
    cfg = Config()
    cfg.llm.model = "fast-model"
    cfg.llm.base_url = "http://localhost:8080/v1"
    cfg.model_entries = {
        "fast": ModelEntry(base_url="http://localhost:8080/v1", model="fast-model", local=True),
        "strong": ModelEntry(
            base_url=("http://localhost:9090/v1" if strong_local else "https://api.remote.example/v1"),
            model="strong-model",
            local=strong_local,
        ),
    }
    cfg.auto_tier.enabled = True
    cfg.auto_tier.fast_entry = "fast"
    cfg.auto_tier.strong_entry = "strong"
    return cfg


def _patch_common(monkeypatch):
    monkeypatch.setattr(turn_mod, "execute_tool", _fake_execute_ok)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])


def _stub_escalator(counter: dict, client, model: str = "strong-model"):
    """A drop-in for escalate_mid_turn: mutate config.llm and hand back *client*
    (a stub) instead of a real AsyncOpenAI, so the swapped-in client stays offline.
    """
    def _fake(config, reason="confidence"):
        counter["n"] += 1
        config.llm.model = model
        return client
    return _fake


# ── escalate_mid_turn unit tests (real function) ─────────────────────────────

def test_escalate_disabled_returns_none():
    cfg = _tiered_config()
    cfg.auto_tier.enabled = False
    assert escalate_mid_turn(cfg, reason="loop_guard") is None


def test_escalate_loop_guard_gate_off_returns_none():
    cfg = _tiered_config()
    cfg.auto_tier.escalate_on_loop_guard = False
    assert escalate_mid_turn(cfg, reason="loop_guard") is None


def test_escalate_verify_gate_off_returns_none():
    cfg = _tiered_config()
    cfg.auto_tier.escalate_on_verify_fail = False
    assert escalate_mid_turn(cfg, reason="verify") is None


def test_escalate_swaps_to_strong_model():
    cfg = _tiered_config()
    client = escalate_mid_turn(cfg, reason="loop_guard")
    assert client is not None
    assert cfg.llm.model == "strong-model"


def test_escalate_confidence_default_reason_unchanged():
    # Default reason keeps the pre-existing confidence behavior.
    cfg = _tiered_config()
    cfg.auto_tier.escalate_on_confidence = False
    assert escalate_mid_turn(cfg) is None


# ── privacy gate ─────────────────────────────────────────────────────────────

def test_privacy_gate_blocks_remote_strong_when_local_only():
    cfg = _tiered_config(strong_local=False)
    cfg.runtime_local_only = True  # private mode active
    assert escalate_mid_turn(cfg, reason="loop_guard") is None
    assert cfg.llm.model == "fast-model"  # untouched


def test_privacy_gate_allows_local_strong_when_local_only():
    cfg = _tiered_config(strong_local=True)
    cfg.runtime_local_only = True
    assert escalate_mid_turn(cfg, reason="loop_guard") is not None
    assert cfg.llm.model == "strong-model"


def test_remote_strong_allowed_when_not_local_only():
    cfg = _tiered_config(strong_local=False)
    cfg.runtime_local_only = False
    assert escalate_mid_turn(cfg, reason="loop_guard") is not None


# ── loop-guard escalation through run_turn ───────────────────────────────────

async def test_loop_guard_escalates_instead_of_stopping(monkeypatch):
    cfg = _tiered_config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 10
    _patch_common(monkeypatch)

    models: list = []
    # search_files repeated to trip the loop guard, then a stop response.
    responses = [_tool_call_response("search_files", {"q": "foo"}) for _ in range(4)]
    responses.append(_stop_response("done"))
    stub = _RecordingClient(responses, models)

    counter = {"n": 0}
    monkeypatch.setattr(model_tier_mod, "escalate_mid_turn", _stub_escalator(counter, stub))

    response, out = await run_turn(_base_messages(), cfg, stub)

    assert counter["n"] == 1
    assert cfg.llm.model == "strong-model"
    assert "loop guard: stopped" not in response  # did NOT hard-stop
    injected = [m for m in out if m.get("role") == "user"
                and "[loop guard: switching" in (m.get("content") or "")]
    assert len(injected) == 1
    assert "strong-model" in injected[0]["content"]


async def test_loop_guard_stops_when_auto_tier_disabled(monkeypatch):
    cfg = _tiered_config()
    cfg.auto_tier.enabled = False
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 50
    _patch_common(monkeypatch)

    models: list = []
    stub = _RecordingClient(
        [_tool_call_response("search_files", {"q": "foo"}) for _ in range(50)], models,
    )
    # Should never be called; assert if it is.
    def _boom(config, reason="confidence"):
        raise AssertionError("escalate_mid_turn called while auto_tier disabled")
    monkeypatch.setattr(model_tier_mod, "escalate_mid_turn", _boom)

    response, _ = await run_turn(_base_messages(), cfg, stub)
    assert "loop guard: stopped" in response
    assert cfg.llm.model == "fast-model"  # untouched


async def test_loop_guard_privacy_gate_stops_turn(monkeypatch):
    # Real escalate_mid_turn returns None (remote strong + local-only) -> stop.
    cfg = _tiered_config(strong_local=False)
    cfg.runtime_local_only = True
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 50
    _patch_common(monkeypatch)

    models: list = []
    stub = _RecordingClient(
        [_tool_call_response("search_files", {"q": "foo"}) for _ in range(50)], models,
    )
    response, _ = await run_turn(_base_messages(), cfg, stub)
    assert "loop guard: stopped" in response
    assert cfg.llm.model == "fast-model"  # never escalated to remote


# ── verify-fail escalation ───────────────────────────────────────────────────

async def test_verify_fail_escalates_before_fix_round(monkeypatch):
    cfg = _tiered_config()
    cfg.verify.enabled = True
    cfg.verify.command = "pytest"
    cfg.verify.max_attempts = 2
    _patch_common(monkeypatch)

    # Verify always fails so the fix round is entered.
    monkeypatch.setattr(turn_mod, "_run_verify_command", lambda c, cwd, t: (1, "boom"))

    models: list = []
    stub = _RecordingClient(
        [
            _tool_call_response("edit_file", {"path": "a.py"}),
            _stop_response("first attempt"),
            _stop_response("second attempt"),
        ],
        models,
    )
    counter = {"n": 0}
    monkeypatch.setattr(model_tier_mod, "escalate_mid_turn", _stub_escalator(counter, stub))

    response, out = await run_turn(_base_messages(), cfg, stub)

    assert counter["n"] == 1
    # The fix round (3rd create) ran on the strong model.
    assert models[-1] == "strong-model"
    verify_notes = [m for m in out if m.get("role") == "user" and "[verify]" in (m.get("content") or "")]
    assert verify_notes
    assert any("switching to a stronger model" in m["content"] for m in verify_notes)


async def test_only_one_escalation_per_turn(monkeypatch):
    # Loop guard escalates first; a later verify failure must not escalate again.
    cfg = _tiered_config()
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 20
    cfg.verify.enabled = True
    cfg.verify.command = "pytest"
    cfg.verify.max_attempts = 2
    _patch_common(monkeypatch)

    monkeypatch.setattr(turn_mod, "_run_verify_command", lambda c, cwd, t: (1, "still broken"))

    models: list = []
    responses = [_tool_call_response("edit_file", {"path": "a.py"}) for _ in range(3)]
    responses += [_stop_response("done1"), _stop_response("done2")]
    stub = _RecordingClient(responses, models)

    counter = {"n": 0}
    monkeypatch.setattr(model_tier_mod, "escalate_mid_turn", _stub_escalator(counter, stub))

    await run_turn(_base_messages(), cfg, stub)
    assert counter["n"] == 1  # loop guard escalated; verify did not re-escalate


async def test_read_guard_escalation_keeps_tool_pairing(monkeypatch):
    """Escalation at the read_file hard ceiling must not orphan tool_calls.

    The assistant tool_calls message is already in history at that point, so
    the escalation note may only land AFTER the batch's tool results — strict
    OpenAI-compatible servers reject an assistant tool_calls message without
    its matching tool messages.
    """
    cfg = _tiered_config()
    cfg.loop_guard.enabled = False           # isolate the read-path hard ceiling
    cfg.llm.max_iterations = 40
    cfg.llm.ctx_window = 200_000
    cfg.llm.compaction_message_threshold = 1000  # keep compaction (needs LLM) out
    _patch_common(monkeypatch)

    models: list = []
    responses = [_tool_call_response("read_file", {"path": "x.py"}) for _ in range(40)]
    client = _RecordingClient(responses, models)
    counter = {"n": 0}
    monkeypatch.setattr(model_tier_mod, "escalate_mid_turn", _stub_escalator(counter, client))

    response, out = await run_turn(_base_messages(), cfg, client)

    assert counter["n"] == 1
    assert "loop guard" in response          # second ceiling (post-escalation) ends the turn
    # The escalation note sits AFTER its batch's tool result…
    idx = next(i for i, m in enumerate(out)
               if m.get("role") == "user" and "switching to a stronger model" in (m.get("content") or ""))
    assert out[idx - 1].get("role") == "tool"
    # …and every NON-terminal assistant tool_calls message is immediately
    # followed by its tool results (pairing invariant — this history is re-sent
    # mid-turn after escalation). The terminal batch is exempt: the hard-stop
    # path has always ended the turn without executing that batch.
    tc_idxs = [i for i, m in enumerate(out) if m.get("role") == "assistant" and m.get("tool_calls")]
    for i in tc_idxs[:-1]:
        for j in range(len(out[i]["tool_calls"])):
            assert out[i + 1 + j].get("role") == "tool", (
                f"orphaned tool_calls at message {i}: followed by {out[i + 1 + j].get('role')}"
            )
