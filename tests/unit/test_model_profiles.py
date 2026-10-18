"""Model setups: capture/restore of hand-made pins, profiles, custom ladder."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import load_config
from agent.core import model_profiles as mp
from agent.core.model_tier import build_ladder, next_stronger
from agent.memory.session import Session, _session_from_data, _session_to_data
from agent.ui.slash import _apply_model


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    proj = tmp_path / "agent.toml"
    proj.write_text("""
[models.small]
base_url = "http://localhost:9001/v1"
model = "small-v1"
params_b = 4

[models.mid]
base_url = "http://localhost:9002/v1"
model = "mid-v1"
params_b = 14

[models.big]
base_url = "http://localhost:9003/v1"
model = "big-v1"
params_b = 70

[models]
default = "mid"
""", encoding="utf-8")
    return SimpleNamespace(config=load_config(proj), token_estimate=lambda: 0, _client=None)


def test_nothing_hand_made_captures_empty(agent):
    assert mp.capture(agent.config) == {}


def test_capture_then_apply_on_fresh_process(agent, tmp_path):
    _apply_model(agent, "summarizer=small")
    _apply_model(agent, "big")
    agent.config.auto_tier.ladder_order = ["big", "small"]
    setup = mp.capture(agent.config)
    assert setup["pins"] == {"default": "big", "summarizer": "small"}
    assert setup["entries"]["big"] == "big-v1@http://localhost:9003/v1"

    # Restart = a config with none of it; restore brings it all back.
    fresh = SimpleNamespace(config=load_config(tmp_path / "agent.toml"),
                            token_estimate=lambda: 0, _client=None)
    assert not mp.same(mp.capture(fresh.config), setup)
    mp.apply(fresh, setup)
    assert mp.same(mp.capture(fresh.config), setup)
    assert fresh.config.runtime_model_pinned
    assert fresh.config.llm.model == "big-v1"


def test_apply_releases_pins_not_in_setup(agent):
    _apply_model(agent, "review=small")
    mp.apply(agent, {"pins": {"summarizer": "big"}})
    assert agent.config.session_role_pins == {"summarizer"}
    assert "review" not in agent.config.model_roles


def test_resolve_remaps_rename_and_flags_changed_model(agent):
    setup = {"pins": {"default": "old-big", "review": "small"},
             "entries": {"old-big": "big-v1@http://localhost:9003/v1",
                         "small": "small-v2@http://elsewhere/v1"}}
    got, problems = mp.resolve(agent.config, setup)
    assert got["pins"] == {"default": "big"}
    assert len(problems) == 1 and "small" in problems[0]


def test_session_roundtrip_keeps_setup():
    s = Session(id="x", model_setup={"pins": {"default": "a"}})
    assert _session_from_data(_session_to_data(s, [])).model_setup == {"pins": {"default": "a"}}
    assert "model_setup" not in _session_to_data(Session(id="y"), [])


def test_profiles_store(agent):
    mp.save_profile(agent.config, "fast", {"pins": {"default": "small"}, "ladder": ["small", "big"]})
    prof = mp.load_profiles()["fast"]
    assert prof["entries"] == {"small": "small-v1@http://localhost:9001/v1",
                               "big": "big-v1@http://localhost:9003/v1"}
    assert mp.delete_profile("fast") and mp.load_profiles() == {}


def test_custom_ladder_order_and_climb(agent):
    cfg = agent.config
    assert [n for n, _ in build_ladder(cfg, check_available=False)] == ["small", "mid", "big"]
    cfg.auto_tier.enabled = cfg.auto_tier.ladder = True
    cfg.auto_tier.ladder_order = ["big", "small"]   # deliberately against power
    assert [n for n, _ in build_ladder(cfg, check_available=False)] == ["big", "small"]
    _apply_model(agent, "big")
    cfg.runtime_model_pinned = False
    from agent.config import model_probe
    orig = model_probe.entry_available
    model_probe.entry_available = lambda e: True
    try:
        assert next_stronger(cfg) == "small"
    finally:
        model_probe.entry_available = orig


def test_http_resume_offers_and_restores(agent, tmp_path):
    """Restart flow: a session saved with pins is resumed by a process that has
    none → the page gets an offer; accepting it re-applies the pins."""
    import asyncio
    import threading
    from agent.ui.http_loop import _HttpUI

    _apply_model(agent, "review=small")
    saved = Session(id="s1", model_setup=mp.capture(agent.config))

    fresh = SimpleNamespace(config=load_config(tmp_path / "agent.toml"),
                            token_estimate=lambda: 0, _client=None)
    server = SimpleNamespace(_agent=fresh, models_overview=lambda: {})
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        ui = _HttpUI(server, saved, loop)
        ui._offer_model_restore()
        assert ui.model_restore and ui.model_restore["problems"] == []
        # The unanswered offer must not be overwritten by the next save.
        ui._sync_grants()
        assert saved.model_setup["pins"] == {"review": "small"}
        r = ui.model_action({"action": "restore", "accept": True})
        assert r["ok"], r
        assert fresh.config.model_roles["review"] == "small"
        assert ui.model_restore is None
        info = ui.model_setup_info()
        assert info["current"]["pins"] == {"review": "small"}
        # Nothing differs any more → no second offer.
        ui._offer_model_restore()
        assert ui.model_restore is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
