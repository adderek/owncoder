"""Unit tests for spawn_agents decision-maker (pure scorer)."""
from __future__ import annotations

import pytest
from agent.config.models import DecisionConfig, ModelEntry
from agent.tools.parallel.decision import pick_model, _score


def _cfg(**kw) -> DecisionConfig:
    cfg = DecisionConfig()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _entry(**kw) -> ModelEntry:
    e = ModelEntry()
    for k, v in kw.items():
        setattr(e, k, v)
    return e


# ── hard filters ─────────────────────────────────────────────────────────────

def test_filters_ctx_too_small():
    entries = {"big": _entry(ctx_window=32768), "small": _entry(ctx_window=4096)}
    result = pick_model(entries, {"est_in_tokens": 8000}, _cfg(enabled=True))
    assert result == "big"


def test_filters_needs_thinking():
    entries = {
        "thinker": _entry(thinking=True),
        "plain": _entry(thinking=False),
    }
    result = pick_model(entries, {"needs_thinking": True}, _cfg(enabled=True))
    assert result == "thinker"


def test_filters_min_strength():
    entries = {
        "strong": _entry(params_b=70.0),
        "weak": _entry(params_b=7.0),
    }
    result = pick_model(entries, {"min_strength": 30.0}, _cfg(enabled=True))
    assert result == "strong"


def test_all_filtered_returns_none():
    entries = {"tiny": _entry(ctx_window=1024)}
    result = pick_model(entries, {"est_in_tokens": 8000}, _cfg(enabled=True))
    assert result is None


# ── scoring ───────────────────────────────────────────────────────────────────

def test_prefer_local():
    cfg = _cfg(prefer_local=2.0, cost_weight=1.0, strength_weight=0.5)
    entries = {
        "cloud": _entry(cost_in_per_1k=0.001, cost_out_per_1k=0.002, local=False),
        "local": _entry(cost_in_per_1k=0.0, cost_out_per_1k=0.0, local=True),
    }
    result = pick_model(entries, {}, cfg)
    assert result == "local"


def test_cost_weight_picks_cheaper():
    cfg = _cfg(prefer_local=0.0, cost_weight=1.0, strength_weight=0.0)
    entries = {
        "expensive": _entry(cost_in_per_1k=0.01, cost_out_per_1k=0.02, local=False),
        "cheap": _entry(cost_in_per_1k=0.001, cost_out_per_1k=0.002, local=False),
    }
    hint = {"est_in_tokens": 1000, "est_out_tokens": 500}
    result = pick_model(entries, hint, cfg)
    assert result == "cheap"


def test_strength_penalty_pushes_weak():
    cfg = _cfg(prefer_local=0.0, cost_weight=1.0, strength_weight=10.0)
    entries = {
        "weak": _entry(params_b=3.0, cost_in_per_1k=0.0, cost_out_per_1k=0.0),
        "strong": _entry(params_b=70.0, cost_in_per_1k=0.001, cost_out_per_1k=0.002),
    }
    # min_strength=30: weak gets large penalty; strong has small cost but no penalty
    hint = {"est_in_tokens": 100, "est_out_tokens": 100, "min_strength": 30.0}
    result = pick_model(entries, hint, cfg)
    assert result == "strong"


def test_tie_break_alphabetical():
    cfg = _cfg(prefer_local=0.0, cost_weight=1.0, strength_weight=0.0)
    entries = {
        "beta": _entry(),
        "alpha": _entry(),
    }
    result = pick_model(entries, {}, cfg)
    assert result == "alpha"


# ── candidates restriction ────────────────────────────────────────────────────

def test_candidates_restricts_pool():
    cfg = _cfg(prefer_local=0.0, cost_weight=1.0, strength_weight=0.0)
    entries = {
        "a": _entry(cost_in_per_1k=0.0),
        "b": _entry(cost_in_per_1k=10.0),
    }
    result = pick_model(entries, {}, cfg, candidates=["b"])
    assert result == "b"


def test_candidates_unknown_name_skipped():
    cfg = _cfg()
    result = pick_model({"a": _entry()}, {}, cfg, candidates=["nonexistent"])
    assert result is None


# ── zero-token hint (no cost signal) ─────────────────────────────────────────

def test_zero_tokens_prefers_local_over_cloud():
    cfg = _cfg(prefer_local=1.0, cost_weight=1.0, strength_weight=0.0)
    entries = {
        "cloud": _entry(local=False, cost_in_per_1k=0.0, cost_out_per_1k=0.0),
        "local": _entry(local=True, cost_in_per_1k=0.0, cost_out_per_1k=0.0),
    }
    result = pick_model(entries, {}, cfg)
    assert result == "local"
