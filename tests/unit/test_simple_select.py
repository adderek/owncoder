"""Simple model selection (/use): where · model · effort over the precise knobs."""
from __future__ import annotations

import pytest

from agent.config import Config, ModelEntry, mode_allows
from agent.core import simple_select as ss


def _cfg(mode: str = "free-hybrid") -> Config:
    cfg = Config()
    cfg.agent.model_mode = mode
    cfg.model_entries = {
        "local-small": ModelEntry(base_url="http://localhost:8080/v1", model="q9", params_b=9),
        "lan-big": ModelEntry(base_url="http://192.168.31.42:8081/v1", model="q35", params_b=35),
        "groq": ModelEntry(base_url="https://api.groq.com/v1", model="llama-70b", params_b=70),
        "paid": ModelEntry(base_url="https://api.x.com/v1", model="big", params_b=400,
                           tier="paid"),
    }
    cfg.model_roles = {"default": "local-small"}
    cfg.model_pools = {}
    return cfg


@pytest.fixture(autouse=True)
def _no_probes(monkeypatch):
    # /use where=… re-pins roles, which probes endpoints; never hit the network.
    monkeypatch.setattr("agent.config.loader._probe_models", lambda *a, **k: ["ok"])


class TestLocationModes:
    def test_private_is_machine_plus_lan(self):
        cfg = _cfg()
        e = cfg.model_entries
        assert mode_allows(e["local-small"], "private")
        assert mode_allows(e["lan-big"], "private")
        assert not mode_allows(e["groq"], "private")
        assert not mode_allows(e["paid"], "private")
        assert mode_allows(ModelEntry(base_url="https://box.example/v1", local=True), "private")

    def test_cloud_is_the_rest(self):
        e = _cfg().model_entries
        assert [n for n, x in e.items() if mode_allows(x, "cloud")] == ["groq", "paid"]


class TestParse:
    def test_tokens_any_order_and_aliases(self):
        assert ss.parse("xhigh cloud strong") == (
            {"where": "cloud", "model": "strong", "effort": "xhigh"}, [])
        assert ss.parse("lan quick max")[0] == {"where": "local", "model": "fast", "effort": "xhigh"}
        assert ss.parse("effort=high where=local")[0] == {"effort": "high", "where": "local"}

    def test_auto_resets_where_and_model_only(self):
        assert ss.parse("auto")[0] == {"where": "auto", "model": "auto"}
        assert ss.parse("auto high")[0] == {"where": "auto", "model": "auto", "effort": "high"}

    def test_unknown(self):
        assert ss.parse("cloud turbo")[1] == ["turbo"]
        assert ss.parse("effort=fast")[1] == ["effort=fast"]


class TestApplyAndState:
    def test_use_sets_underlying_knobs(self):
        cfg = _cfg()
        out = ss.run_use_command(cfg, "cloud strong xhigh")
        assert cfg.agent.model_mode == "cloud"
        assert cfg.auto_tier.enabled and cfg.auto_tier.ladder and cfg.auto_tier.effort == "deep"
        assert cfg.llm.think_level == "max"
        assert out.startswith("using: cloud · strong · xhigh")
        assert "→ paid" in out          # strongest cloud entry

    def test_auto_returns_to_startup_mode(self):
        cfg = _cfg("free-hybrid")
        ss.run_use_command(cfg, "local")
        assert cfg.agent.model_mode == "private"
        ss.run_use_command(cfg, "auto")
        assert cfg.agent.model_mode == "free-hybrid"
        assert ss.state(cfg)["where"] == "auto"

    def test_balanced_is_ladder_middle(self):
        cfg = _cfg("any")
        out = ss.run_use_command(cfg, "balanced")
        assert cfg.auto_tier.effort == "balanced"
        assert "→ groq" in out           # 4 entries: middle index 2

    def test_state_derives_custom_from_precise_knobs(self):
        cfg = _cfg("paid-cloud")
        cfg.llm.think_level = "normal"
        st = ss.state(cfg)
        assert st["where"] == "auto" and st["effort"] == "medium" and st["model"] == "default"
        ss.run_use_command(cfg, "local")
        cfg.agent.model_mode = "lan-only"        # set by /mode
        assert ss.state(cfg)["where"] == "custom"
        assert ss.label(ss.state(cfg)).startswith("lan-only")
        cfg.runtime_model_pinned = True           # /model <entry>
        assert ss.state(cfg)["model"] == "custom"
        assert "📌" in ss.label(ss.state(cfg))

    def test_model_choice_releases_pin(self):
        cfg = _cfg()
        cfg.runtime_model_pinned = True
        out = ss.run_use_command(cfg, "fast")
        assert cfg.runtime_model_pinned is False and "pin released" in out

    def test_bad_input_changes_nothing(self):
        cfg = _cfg()
        out = ss.run_use_command(cfg, "cloud nonsense")
        assert out.startswith("unknown") and cfg.agent.model_mode == "free-hybrid"

    def test_no_arg_shows_help(self):
        out = ss.run_use_command(_cfg(), "")
        assert "/use [auto|local|cloud]" in out and "/think" in out
