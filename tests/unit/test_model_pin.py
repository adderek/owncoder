"""A hand-picked model stays picked.

`/model <entry>` used to last exactly one turn: auto-tier's ladder re-ran at the
top of the next turn (core/agent.py) and `apply_entry` overwrote the choice, so
the user's explicit pick silently evaporated. Pinning now stands auto-tier down
until the user hands control back.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.config.models import ModelEntry
from agent.core.model_tier import escalate_mid_turn, run_effort_command, select_for_turn
from agent.ui.slash import _apply_model


@pytest.fixture
def cfg(monkeypatch) -> Config:
    c = Config()
    c.agent.model_mode = "any"
    c.auto_tier.enabled = True
    c.auto_tier.ladder = True
    c.auto_tier.effort = "smart"
    c.model_entries = {
        "weak": ModelEntry(base_url="https://a.example.com", model="weak-1", params_b=7.0),
        "strong": ModelEntry(base_url="https://b.example.com", model="strong-1", params_b=200.0),
    }
    c.llm.base_url = "https://a.example.com"
    c.llm.model = "weak-1"
    monkeypatch.setattr("agent.config.model_probe.entry_available", lambda *a, **k: True)
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", lambda *a, **k: object())
    return c


def _agent(cfg: Config) -> SimpleNamespace:
    return SimpleNamespace(config=cfg, _client=None, token_estimate=lambda: 0)


def test_unpinned_auto_tier_still_chooses(cfg):
    assert select_for_turn(cfg, "refactor this module", "local") is not None


def test_every_ui_offered_role_is_pinnable(cfg):
    """/model must accept every role the HTTP UI's dropdown can offer.

    The GUI builds its role list from registry.matrix() (ROLE_FALLBACKS); when
    the two drifted, `judge` was shown in the panel but rejected as
    "Unknown role 'judge'". Regression guard against that drift.
    """
    from agent.config.registry import ROLE_FALLBACKS

    agent = _agent(cfg)
    # registry.matrix(), which backs the GUI dropdown, is built from
    # ROLE_FALLBACKS — that is the list the two must not drift on.
    offered = set(ROLE_FALLBACKS) | {"default", "summarizer"}
    for role in offered:
        if role == "embeddings":
            continue  # not a pinnable chat role (vector service)
        ok, msg = _apply_model(agent, f"{role}=strong")
        assert ok, f"role {role!r} rejected: {msg}"
        assert cfg.model_roles[role] == "strong"


def test_pin_stands_auto_tier_down(cfg):
    ok, msg = _apply_model(_agent(cfg), "strong")
    assert ok and cfg.runtime_model_pinned is True
    assert "auto-tier stands down" in msg
    assert select_for_turn(cfg, "refactor this module", "local") is None


def test_pin_blocks_mid_turn_escalation(cfg):
    _apply_model(_agent(cfg), "weak")
    assert escalate_mid_turn(cfg, "loop_guard") is None
    # …and without the pin the same signal still escalates.
    cfg.runtime_model_pinned = False
    assert escalate_mid_turn(cfg, "loop_guard") is not None


def test_model_auto_releases_the_pin(cfg):
    agent = _agent(cfg)
    _apply_model(agent, "strong")
    ok, msg = _apply_model(agent, "auto")
    assert ok and "released" in msg
    assert cfg.runtime_model_pinned is False
    assert select_for_turn(cfg, "refactor this module", "local") is not None


def test_model_auto_is_idempotent(cfg):
    ok, msg = _apply_model(_agent(cfg), "auto")
    assert ok and "no model pin" in msg


def test_role_auto_releases_the_role_pin(cfg):
    agent = _agent(cfg)
    ok, _ = _apply_model(agent, "summarizer=strong")
    assert ok and cfg.model_roles["summarizer"] == "strong"
    ok, msg = _apply_model(agent, "summarizer=auto")
    assert ok and "summarizer" in msg and "auto" in msg
    assert "summarizer" not in cfg.model_roles


def test_role_auto_is_idempotent(cfg):
    ok, msg = _apply_model(_agent(cfg), "summarizer=auto")
    assert ok and "not pinned" in msg


def test_default_auto_via_role_syntax_releases_the_pin(cfg):
    agent = _agent(cfg)
    _apply_model(agent, "strong")
    ok, msg = _apply_model(agent, "default=auto")
    assert ok and "released" in msg
    assert cfg.runtime_model_pinned is False


def test_effort_change_releases_the_pin(cfg):
    _apply_model(_agent(cfg), "strong")
    out = run_effort_command(cfg, "deep")
    assert "pin released" in out
    assert cfg.runtime_model_pinned is False


def test_role_pin_does_not_pin_the_default(cfg):
    ok, _ = _apply_model(_agent(cfg), "summarizer=strong")
    assert ok and cfg.runtime_model_pinned is False


def test_status_reports_the_pin(cfg):
    agent = _agent(cfg)
    _apply_model(agent, "strong")
    ok, out = _apply_model(agent, "")
    assert ok and "pin: pinned by /model" in out


def test_mode_switch_releases_a_pin_it_overrides(cfg, monkeypatch):
    """A mode that forbids the pinned entry re-pins default — pin is gone."""
    from agent.core import model_mode

    cfg.model_entries["local-1"] = ModelEntry(base_url="http://localhost:9/v1",
                                              model="local-1", local=True)
    cfg.model_pools["default"] = ["local-1"]
    _apply_model(_agent(cfg), "strong")   # cloud entry, tier "free"
    monkeypatch.setattr("agent.config.loader._probe_models", lambda *a, **k: {"data": []})

    model_mode.run_mode_command(cfg, "local-only")

    assert cfg.model_roles["default"] == "local-1"
    assert cfg.runtime_model_pinned is False


def test_env_forced_role_rejects_a_pin(cfg, monkeypatch):
    """AGENT_LLM_MODEL forces the default endpoint, so pinning default is a
    no-op the env would silently overrule — /model must refuse it, and leave
    the role untouched (no session pin recorded)."""
    from agent.config.loader import env_locked_roles
    monkeypatch.setenv("AGENT_LLM_MODEL", "env-model")
    assert "default" in env_locked_roles()

    ok, msg = _apply_model(_agent(cfg), "strong")
    assert not ok and "environment" in msg
    assert "default" not in cfg.model_roles


def test_env_forced_role_can_still_pin_other_roles(cfg, monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MODEL", "env-model")
    ok, _ = _apply_model(_agent(cfg), "summarizer=strong")
    assert ok and cfg.model_roles["summarizer"] == "strong"
