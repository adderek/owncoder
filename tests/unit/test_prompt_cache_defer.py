"""Cache-aware compaction deferral — core/prompt_cache.py defer_for_cache (P5).

Compaction rewrites the message prefix, so running it while the provider's
prompt cache is warm throws that cache away and the next request re-pays full
price. Deferring until the cache expires makes the same compaction free.
"""
import pytest

from agent.config.models import Config
from agent.core import cache_tracker, prompt_cache
from agent.core.prompt_cache import cached_prefix_intact, defer_for_cache


@pytest.fixture(autouse=True)
def _clean_state():
    prompt_cache.reset()
    cache_tracker.clear_cache()
    yield
    prompt_cache.reset()
    cache_tracker.clear_cache()


def _cfg(defer=True, ttl=300):
    c = Config()
    c.llm.base_url = "https://api.deepseek.com/v1"
    c.llm.model = "deepseek-chat"
    c.llm.cache_ttl = ttl
    c.llm.defer_compaction_for_cache = defer
    return c


_MESSAGES = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "hello"},
]


def _warm(cfg):
    """Put the endpoint's cache in the warm-and-intact state."""
    cache_tracker.mark_request(cfg.llm.base_url, cfg.llm.model)
    from agent.core.turn_setup import normalize_api_messages
    prompt_cache.check_prefix_stable(normalize_api_messages(_MESSAGES), cfg)


# ── the deferral decision ─────────────────────────────────────────────────────

def test_defers_when_cache_is_warm_and_intact():
    cfg = _cfg()
    _warm(cfg)
    defer, why = defer_for_cache(cfg, _MESSAGES, token_est=1050, budget=1000)
    assert defer and "cache warm" in why


def test_off_by_default():
    cfg = Config()
    assert getattr(cfg.llm, "defer_compaction_for_cache") is False


def test_flag_off_never_defers():
    cfg = _cfg(defer=False)
    _warm(cfg)
    assert defer_for_cache(cfg, _MESSAGES, 1050, 1000) == (False, "")


def test_cache_tracking_disabled_never_defers():
    cfg = _cfg(ttl=0)
    _warm(cfg)
    assert defer_for_cache(cfg, _MESSAGES, 1050, 1000) == (False, "")


def test_cold_cache_does_not_defer():
    """Nothing to protect: compact now."""
    cfg = _cfg()
    assert defer_for_cache(cfg, _MESSAGES, 1050, 1000)[1] == "cache cold"


def test_expired_cache_does_not_defer(monkeypatch):
    cfg = _cfg(ttl=300)
    _warm(cfg)
    import time as _t
    real = _t.time()
    monkeypatch.setattr(cache_tracker.time, "time", lambda: real + 301)
    assert defer_for_cache(cfg, _MESSAGES, 1050, 1000)[0] is False


def test_changed_prefix_does_not_defer():
    """The cache is already lost — deferring would only delay the inevitable."""
    cfg = _cfg()
    _warm(cfg)
    moved = [{"role": "system", "content": "Different preamble entirely."},
             {"role": "user", "content": "hello"}]
    defer, why = defer_for_cache(cfg, moved, 1050, 1000)
    assert not defer and why == "prefix already changed"


# ── the overflow guard ────────────────────────────────────────────────────────

def test_large_overshoot_compacts_anyway():
    """A deferral must never be the reason a turn blows the context window."""
    cfg = _cfg()
    _warm(cfg)
    defer, why = defer_for_cache(cfg, _MESSAGES, token_est=5000, budget=1000)
    assert not defer and why == "overshoot too large"


def test_overshoot_boundary_is_respected():
    cfg = _cfg()
    _warm(cfg)
    limit = int(1000 * prompt_cache._DEFER_MAX_OVERSHOOT)
    assert defer_for_cache(cfg, _MESSAGES, limit, 1000)[0] is True
    assert defer_for_cache(cfg, _MESSAGES, limit + 1, 1000)[0] is False


def test_no_output_reserve_leaves_no_room_to_defer():
    """With max_output_tokens 0 the budget already sits against the window."""
    cfg = _cfg()
    cfg.llm.ctx_window = 8192
    cfg.llm.max_output_tokens = 0
    _warm(cfg)
    from agent.core.context_budget import input_token_budget
    budget = input_token_budget(cfg)          # 7692, i.e. window - overhead
    token_est = int(budget * 1.1)             # inside the relative overshoot
    defer, why = defer_for_cache(cfg, _MESSAGES, token_est, budget)
    assert not defer and why == "would eat the output reserve"


def test_deferral_may_borrow_up_to_half_the_reserve():
    cfg = _cfg()
    cfg.llm.ctx_window = 32768
    cfg.llm.max_output_tokens = 4096
    _warm(cfg)
    from agent.core.context_budget import input_token_budget, _PROMPT_OVERHEAD
    budget = input_token_budget(cfg)                       # 28172
    limit = 32768 - 2048 - _PROMPT_OVERHEAD                # 30220
    assert defer_for_cache(cfg, _MESSAGES, limit, budget)[0] is True
    assert defer_for_cache(cfg, _MESSAGES, limit + 1, budget)[1] == (
        "would eat the output reserve")


def test_realistic_default_config_can_actually_defer():
    """Guards against a guard so strict that the feature never fires."""
    cfg = _cfg()
    cfg.llm.ctx_window = 32768
    cfg.llm.max_output_tokens = 4096
    _warm(cfg)
    from agent.core.context_budget import input_token_budget
    budget = input_token_budget(cfg)
    assert defer_for_cache(cfg, _MESSAGES, int(budget * 1.05), budget)[0] is True


def test_zero_budget_does_not_defer():
    cfg = _cfg()
    _warm(cfg)
    assert defer_for_cache(cfg, _MESSAGES, 10, 0)[0] is False


def test_deferral_is_bounded_by_growth():
    """A permanently warm cache must not defer forever — overshoot ends it."""
    cfg = _cfg()
    _warm(cfg)
    budget = 1000
    deferrals = 0
    for token_est in range(1001, 2000, 25):     # context keeps growing
        cache_tracker.mark_request(cfg.llm.base_url, cfg.llm.model)  # stays warm
        if defer_for_cache(cfg, _MESSAGES, token_est, budget)[0]:
            deferrals += 1
        else:
            break
    assert 0 < deferrals < 10


# ── cached_prefix_intact ──────────────────────────────────────────────────────

class TestCachedPrefixIntact:
    def test_false_when_nothing_recorded_yet(self):
        """A cache that does not exist cannot be lost."""
        assert not cached_prefix_intact(_MESSAGES, _cfg())

    def test_true_after_the_same_prefix_was_sent(self):
        cfg = _cfg()
        prompt_cache.check_prefix_stable(_MESSAGES, cfg)
        assert cached_prefix_intact(_MESSAGES, cfg)

    def test_false_after_the_prefix_changes(self):
        cfg = _cfg()
        prompt_cache.check_prefix_stable(_MESSAGES, cfg)
        assert not cached_prefix_intact(
            [{"role": "system", "content": "new"}], cfg)

    def test_empty_messages_are_not_intact(self):
        assert not cached_prefix_intact([], _cfg())

    def test_does_not_record_anything(self):
        """Read-only: asking must not make the next check_prefix_stable lie."""
        cfg = _cfg()
        cached_prefix_intact(_MESSAGES, cfg)
        assert not cached_prefix_intact(_MESSAGES, cfg)

    def test_tracked_per_endpoint(self):
        cfg = _cfg()
        prompt_cache.check_prefix_stable(_MESSAGES, cfg)
        other = _cfg()
        other.llm.model = "deepseek-reasoner"
        assert cached_prefix_intact(_MESSAGES, cfg)
        assert not cached_prefix_intact(_MESSAGES, other)
