"""Local→local failover: rescue a turn whose local endpoint is failing (e.g. a
router preset whose weights are missing 500s every request) by switching to
another live local entry — never toward remote.
"""
import agent.config.model_probe as mp
from agent.config import Config
from agent.config.models import ModelEntry
from agent.core import model_routing


def _cfg():
    cfg = Config()
    cfg.failover.enabled = True
    cfg.failover.local_entry = "backup"
    cfg.model_entries = {
        "broken": ModelEntry(base_url="http://localhost:8081/v1", model="qwen-14b", local=True),
        "backup": ModelEntry(base_url="http://localhost:8001/v1", model="qwen3.6-27b", local=True),
        "cloud": ModelEntry(base_url="https://api.example.com/v1", model="big", tier="paid"),
    }
    cfg.llm.base_url = "http://localhost:8081/v1"
    cfg.llm.model = "qwen-14b"
    return cfg


def test_alternative_switches_to_preferred_local(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: True)
    client = model_routing.failover_to_alternative(cfg)
    assert client is not None
    assert cfg.llm.base_url == "http://localhost:8001/v1"
    assert cfg.llm.model == "qwen3.6-27b"


def test_alternative_never_routes_remote(monkeypatch):
    cfg = _cfg()
    # Only the cloud entry is live besides the failing one → no candidate.
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: "example.com" in e.base_url)
    assert model_routing.failover_to_alternative(cfg) is None


def test_alternative_skips_unavailable_entries(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: False)
    assert model_routing.failover_to_alternative(cfg) is None


def test_alternative_disabled_failover():
    cfg = _cfg()
    cfg.failover.enabled = False
    assert model_routing.failover_to_alternative(cfg) is None


def _cfg_lan():
    """Active endpoint is cloud; a LAN (private-IP) self-hosted box is available."""
    cfg = Config()
    cfg.failover.enabled = True
    cfg.model_entries = {
        "lan": ModelEntry(base_url="http://192.168.31.42:8001/v1", model="qwen3.6-27b"),
        "cloud": ModelEntry(base_url="https://openrouter.ai/api/v1", model="ds:free", tier="free"),
    }
    cfg.llm.base_url = "https://openrouter.ai/api/v1"
    cfg.llm.model = "ds:free"
    return cfg


def test_failover_degrades_to_lan_box(monkeypatch):
    # Cloud endpoint died; the LAN box (tier "remote") must rescue the turn.
    cfg = _cfg_lan()
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: "192.168.31.42" in e.base_url)
    client = model_routing.failover_to_local(cfg)
    assert client is not None
    assert cfg.llm.base_url == "http://192.168.31.42:8001/v1"
    assert cfg.llm.model == "qwen3.6-27b"


def test_failover_skips_dead_lan_box(monkeypatch):
    # LAN box configured but unreachable → no blind switch, surface None.
    cfg = _cfg_lan()
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: False)
    assert model_routing.failover_to_local(cfg) is None


def test_local_only_pin_excludes_lan(monkeypatch):
    # Private session (runtime_local_only) must not degrade off the loopback iface.
    cfg = _cfg_lan()
    cfg.runtime_local_only = True
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: True)
    assert model_routing.failover_to_local(cfg) is None
