"""Tests for tier-conditional system-prompt overlays (S5)."""
from __future__ import annotations

from unittest.mock import patch

from agent.config.models import Config
from agent.core import prompts


class TestTierOverlay:
    def test_off_returns_none(self):
        cfg = Config()
        cfg.agent.tier_prompt_overlay = "off"
        assert prompts._tier_overlay(cfg) is None

    def test_default_is_off(self):
        # A fresh config must not change prompt behavior.
        assert Config().agent.tier_prompt_overlay == "off"
        assert prompts._tier_overlay(Config()) is None

    def test_forced_local_loads_and_strips_comments(self):
        cfg = Config()
        cfg.agent.tier_prompt_overlay = "local"
        ov = prompts._tier_overlay(cfg)
        assert ov is not None
        fname, text = ov
        assert fname == "tier_local.txt"
        assert "WORKING RULES" in text
        # Comment lines (dev notes) must not reach the model.
        assert not any(l.startswith("#") for l in text.splitlines())
        assert "Lines starting with" not in text

    def test_forced_unknown_tier_returns_none(self):
        cfg = Config()
        cfg.agent.tier_prompt_overlay = "paid"  # no overlay file for paid
        assert prompts._tier_overlay(cfg) is None

    def test_auto_uses_resolved_tier(self):
        cfg = Config()
        cfg.agent.tier_prompt_overlay = "auto"
        with patch.object(prompts, "_resolve_default_tier", return_value="local"):
            ov = prompts._tier_overlay(cfg)
        assert ov is not None and ov[0] == "tier_local.txt"

    def test_auto_non_local_tier_returns_none(self):
        cfg = Config()
        cfg.agent.tier_prompt_overlay = "auto"
        with patch.object(prompts, "_resolve_default_tier", return_value="paid"):
            assert prompts._tier_overlay(cfg) is None

    def test_resolve_tier_never_raises(self):
        # A malformed config must degrade to "" (no overlay), not crash the
        # whole system-prompt build.
        assert prompts._resolve_default_tier(object()) == ""
