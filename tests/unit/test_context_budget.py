"""Config-derived token budgets — core/context_budget.py.

The regression these pin: ctx_window == 0 ("auto", or a failed probe) used to
produce a budget of 1, so every turn looked over-budget and compacted on every
iteration.
"""
import logging

import pytest

from agent.config.models import Config
from agent.core import context_budget as cb
from agent.core.context_budget import (
    DEFAULT_CTX_WINDOW,
    effective_ctx_window,
    input_token_budget,
)


@pytest.fixture(autouse=True)
def _reset_warn_latch():
    """The auto-ctx warning fires once per process; unlatch it per test."""
    cb._warned_auto_ctx = False
    yield
    cb._warned_auto_ctx = False


def _cfg(ctx_window=8192, max_output_tokens=1024, model="test-model"):
    c = Config()
    c.llm.ctx_window = ctx_window
    c.llm.max_output_tokens = max_output_tokens
    c.llm.model = model
    return c


# ── effective_ctx_window ──────────────────────────────────────────────────────

def test_configured_window_passes_through():
    assert effective_ctx_window(_cfg(ctx_window=8192)) == 8192


def test_zero_window_falls_back_to_default():
    assert effective_ctx_window(_cfg(ctx_window=0)) == DEFAULT_CTX_WINDOW


def test_negative_window_falls_back_to_default():
    assert effective_ctx_window(_cfg(ctx_window=-1)) == DEFAULT_CTX_WINDOW


def test_missing_llm_section_falls_back():
    class _Bare:
        pass

    assert effective_ctx_window(_Bare()) == DEFAULT_CTX_WINDOW


def test_auto_window_warns_once_naming_the_model(caplog):
    cfg = _cfg(ctx_window=0, model="qwen-probe-failed")
    with caplog.at_level(logging.WARNING, logger=cb.__name__):
        effective_ctx_window(cfg)
        effective_ctx_window(cfg)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "auto-ctx warning must not repeat every turn"
    assert "qwen-probe-failed" in warnings[0].getMessage()


def test_configured_window_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger=cb.__name__):
        effective_ctx_window(_cfg(ctx_window=4096))
    assert not caplog.records


# ── input_token_budget ────────────────────────────────────────────────────────

def test_budget_subtracts_reserve_and_overhead():
    assert input_token_budget(_cfg(ctx_window=8192, max_output_tokens=1024)) == (
        8192 - 1024 - cb._PROMPT_OVERHEAD)


def test_auto_window_yields_a_usable_budget():
    """The actual regression: 0 - 8192 - 500 used to clamp to 1."""
    budget = input_token_budget(_cfg(ctx_window=0, max_output_tokens=8192))
    assert budget > 10_000
    assert budget == DEFAULT_CTX_WINDOW - 8192 - cb._PROMPT_OVERHEAD


def test_oversized_output_reserve_is_clamped_to_half():
    """max_output_tokens >= ctx_window must not drive the budget to ~0."""
    budget = input_token_budget(_cfg(ctx_window=8192, max_output_tokens=99_000))
    assert budget == 8192 - 4096 - cb._PROMPT_OVERHEAD


def test_budget_never_below_floor():
    assert input_token_budget(_cfg(ctx_window=512, max_output_tokens=512)) >= 1024


def test_zero_output_reserve_uses_whole_window():
    assert input_token_budget(_cfg(ctx_window=8192, max_output_tokens=0)) == (
        8192 - cb._PROMPT_OVERHEAD)


def test_budget_is_below_the_window_it_derives_from():
    for ctx in (2048, 8192, 32768, 200_000):
        cfg = _cfg(ctx_window=ctx, max_output_tokens=4096)
        assert input_token_budget(cfg) < effective_ctx_window(cfg)
