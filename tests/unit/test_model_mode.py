"""Unit tests for model-mode tiering, registry selection, and /mode command."""
from __future__ import annotations

from agent.config import Config, ModelEntry, make_registry, entry_tier, mode_allows
from agent.config.registry import MODE_TIERS
from agent.core.model_mode import run_mode_command


def _cfg(mode: str = "any") -> Config:
    cfg = Config()
    cfg.agent.model_mode = mode
    cfg.model_entries = {
        "local-coder": ModelEntry(base_url="http://localhost:8080/v1", model="qwen", local=True),
        "cerebras": ModelEntry(base_url="https://api.cerebras.ai/v1", model="llama-3.3-70b",
                               tags=["background", "summarizer"]),
        "gpt4o": ModelEntry(base_url="https://api.openai.com/v1", model="gpt-4o",
                            cost_in_per_1k=0.005, cost_out_per_1k=0.015),
        "deepseek": ModelEntry(base_url="https://api.deepseek.com/v1", model="r1", tier="paid"),
    }
    cfg.model_roles = {"default": "local-coder", "summarizer": "local-coder"}
    return cfg


class TestEntryTier:
    def test_localhost_is_local(self):
        assert entry_tier(ModelEntry(base_url="http://localhost:8080/v1")) == "local"

    def test_local_flag_is_local(self):
        assert entry_tier(ModelEntry(base_url="https://x/v1", local=True)) == "local"

    def test_priced_cloud_is_paid(self):
        assert entry_tier(ModelEntry(base_url="https://x/v1", cost_in_per_1k=0.001)) == "paid"

    def test_free_cloud(self):
        assert entry_tier(ModelEntry(base_url="https://api.groq.com/v1")) == "free"

    def test_explicit_tier_wins(self):
        e = ModelEntry(base_url="http://localhost:8080/v1", tier="paid")
        assert entry_tier(e) == "paid"


class TestModeAllows:
    def test_local_only(self):
        loc = ModelEntry(local=True)
        free = ModelEntry(base_url="https://api.groq.com/v1")
        assert mode_allows(loc, "local-only")
        assert not mode_allows(free, "local-only")

    def test_free_hybrid(self):
        loc = ModelEntry(local=True)
        free = ModelEntry(base_url="https://api.groq.com/v1")
        paid = ModelEntry(base_url="https://x/v1", cost_in_per_1k=0.01)
        assert mode_allows(loc, "free-hybrid")
        assert mode_allows(free, "free-hybrid")
        assert not mode_allows(paid, "free-hybrid")

    def test_unknown_mode_permissive(self):
        assert mode_allows(ModelEntry(base_url="https://x/v1", cost_in_per_1k=1.0), "bogus")


class TestRegistryAllowed:
    def test_allowed_names_local_only(self):
        reg = make_registry(_cfg("local-only"))
        assert reg.allowed_names() == ["local-coder"]

    def test_allowed_names_free_cloud(self):
        reg = make_registry(_cfg("free-cloud"))
        assert reg.allowed_names() == ["cerebras"]

    def test_allowed_names_any(self):
        reg = make_registry(_cfg("any"))
        assert set(reg.allowed_names()) == {"local-coder", "cerebras", "gpt4o", "deepseek"}


class TestBackgroundSelection:
    def test_explicit_background_role_wins(self):
        cfg = _cfg("free-hybrid")
        cfg.model_roles["background"] = "deepseek"
        assert make_registry(cfg).background.model == "r1"

    def test_free_hybrid_prefers_free_cloud(self):
        # No background role pinned → auto-pick the free cloud model to offload.
        reg = make_registry(_cfg("free-hybrid"))
        assert reg.background.model == "llama-3.3-70b"

    def test_local_only_falls_back_to_summarizer(self):
        # No free model permitted → fall back to summarizer role.
        reg = make_registry(_cfg("local-only"))
        assert reg.background.model == "qwen"


class TestModeCommand:
    def test_show_lists_tiers(self):
        out = run_mode_command(_cfg("any"), "")
        assert "model-mode: any" in out
        assert "cerebras" in out

    def test_switch_sets_mode(self):
        cfg = _cfg("any")
        out = run_mode_command(cfg, "free-cloud")
        assert cfg.agent.model_mode == "free-cloud"
        assert "free-cloud" in out

    def test_invalid_mode_rejected(self):
        cfg = _cfg("any")
        out = run_mode_command(cfg, "bogus")
        assert cfg.agent.model_mode == "any"
        assert "unknown mode" in out

    def test_all_modes_defined(self):
        assert set(MODE_TIERS) == {
            "local-only", "free-cloud", "free-hybrid", "paid-cloud", "manual", "any"
        }


class TestRoleMatrix:
    def test_unpinned_role_falls_back_to_default(self):
        reg = make_registry(_cfg("any"))
        assert reg.role("review").model == "qwen"  # default

    def test_explicit_role_pin_wins(self):
        cfg = _cfg("any")
        cfg.model_roles["review"] = "deepseek"
        assert make_registry(cfg).role("review").model == "r1"

    def test_namer_inherits_background_offload(self):
        # Unpinned namer routes through background → free cloud under free-hybrid.
        reg = make_registry(_cfg("free-hybrid"))
        assert reg.role("namer").model == "llama-3.3-70b"

    def test_pinned_namer_overrides_offload(self):
        cfg = _cfg("free-hybrid")
        cfg.model_roles["namer"] = "local-coder"
        assert make_registry(cfg).role("namer").model == "qwen"

    def test_manual_mode_disables_offload(self):
        # manual: background falls to summarizer, no free-cloud auto-pick.
        reg = make_registry(_cfg("manual"))
        assert reg.background.model == "qwen"  # summarizer fallback
        assert reg.role("namer").model == "qwen"

    def test_matrix_covers_roles(self):
        m = make_registry(_cfg("any")).matrix()
        for r in ("default", "summarizer", "background", "review", "commit"):
            assert r in m
        assert m["review"][0] == "local-coder"  # resolved entry name
        assert m["review"][1] == "local"        # tier
