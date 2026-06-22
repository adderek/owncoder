"""Unit tests for agent.prompt_compiler.

Covers cache key derivation, exclude list, env-disable override, the
fallback path on a fresh cache miss, the auto-suspect/recompile loop on
high error rate, and the permanent-disable cap on repeated compile
failures.

Compilation itself (which would call out to a real LLM) is replaced by a
synchronous stub so these tests have no network dependency.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import prompt_compiler as pc


@pytest.fixture(autouse=True)
def _reset_state():
    """Each test gets a clean in-memory cache."""
    pc.reset_state_for_tests()
    yield
    pc.reset_state_for_tests()


def _stub_compile(text: str) -> str:
    """Deterministic shrink: drop every other word so output is always shorter."""
    parts = text.split()
    kept = [w for i, w in enumerate(parts) if i % 2 == 0]
    return " ".join(kept) or text[: max(1, len(text) // 2)]


def _patch_compile_blocking():
    """Make `_spawn_compile` run synchronously and use the deterministic stub.

    Kept for tests that monkey-patch _spawn_compile specifically.
    """
    def _run_inline(key, name, original, config):
        with pc._lock:
            pc._in_flight.add(key)
        try:
            compiled = _stub_compile(original)
            pc._store_compiled(key, original, compiled, config)
        finally:
            with pc._lock:
                pc._in_flight.discard(key)
    return patch.object(pc, "_spawn_compile", side_effect=_run_inline)


def _warm_cache(name: str, text: str, cfg) -> None:
    """Seed entry + store compiled text. `load()` no longer auto-spawns compile
    (llama.cpp serialisation) so tests warm the cache explicitly."""
    pc.load(name, text, cfg)  # ensure entry exists
    key = pc._cache_key(cfg.llm.base_url, cfg.llm.model, text)
    pc._store_compiled(key, text, _stub_compile(text), cfg)


# ── Tests ──────────────────────────────────────────────────────────────────

def test_disabled_returns_original(cfg):
    cfg.compile_prompts.enabled = False
    out = pc.load("system.txt", "hello world this is the prompt", cfg)
    assert out == "hello world this is the prompt"


def test_env_overrides_enabled(cfg, monkeypatch):
    cfg.compile_prompts.enabled = True
    monkeypatch.setenv("AGENT_COMPILE_PROMPTS", "0")
    out = pc.load("system.txt", "hello world", cfg)
    assert out == "hello world"


def test_exclude_list_skips_compilation(cfg):
    cfg.compile_prompts.exclude = ["system.txt"]
    with _patch_compile_blocking() as m:
        out = pc.load("system.txt", "long prompt text here yes please compile me", cfg)
    assert out == "long prompt text here yes please compile me"
    m.assert_not_called()


def test_first_call_returns_original_then_caches(cfg):
    text = "this is a moderately long prompt that should compile fine"
    first = pc.load("system.txt", text, cfg)      # miss → original
    _warm_cache("system.txt", text, cfg)
    second = pc.load("system.txt", text, cfg)     # hit → compiled
    assert first == text
    assert second != text
    assert len(second) < len(text)


def test_cache_key_differs_per_model(cfg):
    text = "shared prompt body that both models will see"
    with _patch_compile_blocking():
        cfg.llm.model = "model-a"
        pc.load("system.txt", text, cfg)
        pc.load("system.txt", text, cfg)          # warm
        cfg.llm.model = "model-b"
        out_b = pc.load("system.txt", text, cfg)  # different cache key → miss
    assert out_b == text                          # not yet cached for model-b
    rows = pc.status(cfg)
    assert len(rows) == 2
    assert {r["model"] for r in rows} == {"model-a", "model-b"}


def test_cache_key_differs_per_api_base(cfg):
    text = "same prompt body sent to two different servers"
    with _patch_compile_blocking():
        cfg.llm.base_url = "http://server-a/v1"
        pc.load("system.txt", text, cfg)
        pc.load("system.txt", text, cfg)
        cfg.llm.base_url = "http://server-b/v1"
        out_b = pc.load("system.txt", text, cfg)
    assert out_b == text
    assert len(pc.status(cfg)) == 2


def _run_arm(name: str, text: str, cfg, arm: str, oks: list[bool]) -> None:
    """Drive `oks` tool-call outcomes through one A/B arm of `name`.

    Forces the arm by pinning holdout to 0 (compiled) or 1 (original), then
    clears the per-session arm cache so load() re-picks for this batch.
    """
    cfg.compile_prompts.holdout_ratio = 1.0 if arm == "original" else 0.0
    pc._s._active_arm.clear()
    pc._s._active.clear()
    pc.load(name, text, cfg)            # assigns the arm for this "session"
    for ok in oks:
        pc.record_call(ok, cfg)


def test_ab_regression_recompiles_then_pins(cfg):
    """Compiled arm worse than the control by the margin → recompile, then pin."""
    text = "prompt body to compile and then degrade over time please"
    cfg.compile_prompts.min_samples = 4
    cfg.compile_prompts.error_rate_threshold = 0.3
    cfg.compile_prompts.regression_margin = 0.2
    cfg.compile_prompts.max_recompile_attempts = 2

    def _drive_one_regression_round():
        _warm_cache("system.txt", text, cfg)                       # attempts +1, compiled
        _run_arm("system.txt", text, cfg, "compiled", [False, False, False, True])  # 75% err
        _run_arm("system.txt", text, cfg, "original", [True, True, True, True])     # 0% err
        return pc.evaluate(cfg)

    # Round 1: attempts==1 (< max 2) → recompile (suspect).
    actions = _drive_one_regression_round()
    assert actions and actions[0]["action"] == "recompile"
    assert {r["name"]: r for r in pc.status(cfg)}["system.txt"]["status"] == "suspect"

    # load() drops suspect → pending but KEEPS attempts so the cap can trip.
    out = pc.load("system.txt", text, cfg)
    assert out == text                                              # falls back to original

    # Round 2: recompiling bumps attempts to 2 == max → pin to original forever.
    actions = _drive_one_regression_round()
    assert actions[0]["action"] == "pin"
    pinned = {r["name"]: r for r in pc.status(cfg)}["system.txt"]
    assert pinned["status"] == "pinned"
    assert pinned["disabled_reason"] == "regression"

    # A pinned prompt always serves the original, never the compiled variant.
    assert pc.load("system.txt", text, cfg) == text


def test_ab_no_regression_keeps_compiled(cfg):
    """When the compiled arm is no worse than control, the variant is kept."""
    text = "prompt body that compresses cleanly without hurting behaviour"
    cfg.compile_prompts.min_samples = 4
    cfg.compile_prompts.error_rate_threshold = 0.3
    cfg.compile_prompts.regression_margin = 0.2
    _warm_cache("system.txt", text, cfg)
    # Both arms degrade equally → regression ≈ 0 → ambient noise, not the prompt.
    _run_arm("system.txt", text, cfg, "compiled", [False, False, True, True])
    _run_arm("system.txt", text, cfg, "original", [False, False, True, True])

    actions = pc.evaluate(cfg)
    assert actions and actions[0]["action"] == "keep"
    assert {r["name"]: r for r in pc.status(cfg)}["system.txt"]["status"] == "compiled"


def test_ab_waits_for_both_arms(cfg):
    """No verdict until BOTH arms clear min_samples — control alone isn't enough."""
    text = "prompt that only ever runs on the compiled treatment arm here"
    cfg.compile_prompts.min_samples = 4
    _warm_cache("system.txt", text, cfg)
    _run_arm("system.txt", text, cfg, "compiled", [False, False, False, False])
    # No original-arm samples at all → evaluate must abstain.
    actions = pc.evaluate(cfg)
    assert actions == []
    assert {r["name"]: r for r in pc.status(cfg)}["system.txt"]["status"] == "compiled"


def test_compile_failure_eventually_disables(cfg):
    """After max_recompile_attempts failures the entry is permanently disabled."""
    text = "prompt that will keep failing to compile"
    cfg.compile_prompts.max_recompile_attempts = 2

    pc.load("system.txt", text, cfg)  # seed entry
    key = pc._cache_key(cfg.llm.base_url, cfg.llm.model, text)
    pc._record_compile_failure(key, cfg)          # attempt 1
    pc._record_compile_failure(key, cfg)          # attempt 2 -> disabled

    rows = {r["name"]: r for r in pc.status(cfg)}
    assert rows["system.txt"]["status"] == "disabled"

    # Further loads should be quiet (no more spawned compiles).
    with patch.object(pc, "_spawn_compile") as m:
        out = pc.load("system.txt", text, cfg)
    assert out == text
    m.assert_not_called()


def test_clear_removes_entries(cfg):
    text = "compile me and then forget me"
    with _patch_compile_blocking():
        pc.load("system.txt", text, cfg)
        pc.load("system.txt", text, cfg)
    assert len(pc.status(cfg)) == 1
    removed = pc.clear(cfg)
    assert removed == 1
    assert pc.status(cfg) == []


def test_recompile_resets_attempts(cfg):
    """`recompile` should put entries back in pending with a clean attempts count."""
    text = "compile, then ask to recompile"
    with _patch_compile_blocking():
        pc.load("system.txt", text, cfg)
        pc.load("system.txt", text, cfg)
    pc.recompile(cfg)
    rows = {r["name"]: r for r in pc.status(cfg)}
    assert rows["system.txt"]["status"] == "pending"
    assert rows["system.txt"]["attempts"] == 0


def _fake_openai_returning(text: str):
    """Build a fake OpenAI client whose chat.completions.create returns *text*."""
    resp = type("R", (), {
        "choices": [type("C", (), {"message": type("M", (), {"content": text})()})()]
    })()
    client = type("Cli", (), {})()
    client.chat = type("Ch", (), {})()
    client.chat.completions = type("Co", (), {})()
    client.chat.completions.create = lambda **kw: resp
    return client


def test_compiled_must_beat_min_savings_ratio(cfg):
    """`_do_compile` raises _NoSavings when the token reduction is too small."""
    # Original tokenises to ~50 tokens; "compiled" output is nearly identical
    # length so token savings are far below the 10% default threshold.
    original = "alpha beta gamma " * 30
    near_same = "alpha beta gamma " * 28        # ~7% shorter — under threshold

    with patch("openai.OpenAI", return_value=_fake_openai_returning(near_same)):
        with pytest.raises(pc._NoSavings):
            pc._do_compile("system.txt", original, cfg)


def test_no_savings_disables_immediately(cfg):
    """When the model can't compress enough, the entry is permanently disabled
    on the first attempt — we don't burn the full max_recompile_attempts on a
    structural fact about (prompt, model)."""
    original = "alpha beta gamma " * 30
    near_same = "alpha beta gamma " * 28

    pc.load("system.txt", original, cfg)  # seed entry
    key = pc._cache_key(cfg.llm.base_url, cfg.llm.model, original)
    with patch("openai.OpenAI", return_value=_fake_openai_returning(near_same)):
        try:
            pc._do_compile("system.txt", original, cfg)
        except pc._NoSavings:
            pc._disable_for(key, "no_savings", cfg)

    rows = {r["name"]: r for r in pc.status(cfg)}
    assert rows["system.txt"]["status"] == "disabled"
    assert rows["system.txt"]["disabled_reason"] == "no_savings"


def test_cumulative_tokens_saved_bumps_per_hit(cfg):
    """Every cache hit should increment tokens_saved_total by the per-call delta."""
    text = (
        "the quick brown fox jumps over the lazy dog. " * 20
    )  # large enough that the stub shrink genuinely beats the 10% threshold
    _warm_cache("system.txt", text, cfg)
    pc.load("system.txt", text, cfg)             # hit  -> compiled (bump 1)
    pc.load("system.txt", text, cfg)             # hit  -> compiled (bump 2)

    rows = {r["name"]: r for r in pc.status(cfg)}
    saved_per_call = rows["system.txt"]["original_tokens"] - rows["system.txt"]["compiled_tokens"]
    assert saved_per_call > 0
    assert rows["system.txt"]["tokens_saved_total"] == saved_per_call * 2
