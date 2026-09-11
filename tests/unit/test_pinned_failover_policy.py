"""Pinned-entry failover policy: an automatic switch off a hand-picked model is
a spending decision, not a repair. ``failover.pinned_policy`` decides whether a
pinned turn may drift onto another (possibly paid) entry, only onto a free one,
or not at all — the last two pause and let the user choose.
"""
import agent.config.model_probe as mp
from agent.config import Config
from agent.config.models import ModelEntry
from agent.core import model_routing
from agent.core import turn_errors


def _cfg(*, active_cloud=True):
    cfg = Config()
    cfg.failover.enabled = True
    cfg.failover.max_retries = 1
    entries = {
        "free-peer": ModelEntry(base_url="https://free.example.com/v1", model="small", tier="free"),
        "paid-peer": ModelEntry(base_url="https://paid.example.com/v1", model="big", tier="paid"),
        "lan-peer": ModelEntry(base_url="http://192.168.31.42:8081/v1", model="lan-model"),
    }
    if active_cloud:
        active = ModelEntry(base_url="https://active.example.com/v1", model="active-model", tier="free")
        entries["active"] = active
        cfg.llm.base_url = active.base_url
        cfg.llm.model = active.model
    else:
        active = ModelEntry(base_url="http://192.168.31.42:8083/v1", model="ornith-10-35B")
        entries["active"] = active
        cfg.llm.base_url = active.base_url
        cfg.llm.model = active.model
    cfg.model_entries = entries
    return cfg


def _all_live(monkeypatch):
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: True)


def test_pin_detected_from_runtime_flag_and_session_pins():
    cfg = _cfg()
    assert model_routing.is_active_default_pinned(cfg) is False
    cfg.runtime_model_pinned = True
    assert model_routing.is_active_default_pinned(cfg) is True
    cfg.runtime_model_pinned = False
    cfg.session_role_pins = {"default"}
    assert model_routing.is_active_default_pinned(cfg) is True


def test_ask_policy_refuses_to_switch(monkeypatch):
    # Pinned + "ask": peers are live, but the answer is "stop and ask" (None),
    # not a silent switch off the model the user chose.
    cfg = _cfg()
    cfg.runtime_model_pinned = True
    cfg.failover.pinned_policy = "ask"
    _all_live(monkeypatch)
    assert turn_errors.try_failover(cfg) is None


def test_free_only_policy_skips_paid_peer(monkeypatch):
    # Only the paid cloud peer is live besides the failing active entry: a pinned
    # turn must stop rather than spend.
    cfg = _cfg()
    cfg.runtime_model_pinned = True
    cfg.failover.pinned_policy = "free-only"
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: "paid.example.com" in e.base_url)
    assert turn_errors.try_failover(cfg) is None


def test_free_only_policy_still_uses_free_peer(monkeypatch):
    cfg = _cfg()
    cfg.runtime_model_pinned = True
    cfg.failover.pinned_policy = "free-only"
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: "free.example.com" in e.base_url)
    client = turn_errors.try_failover(cfg)
    assert client is not None
    assert cfg.llm.base_url == "https://free.example.com/v1"


def test_fallback_policy_allows_paid_peer(monkeypatch):
    # Default preserves the pre-policy behaviour: any live entry may take over.
    cfg = _cfg()
    cfg.runtime_model_pinned = True
    cfg.failover.pinned_policy = "fallback"
    monkeypatch.setattr(mp, "entry_available", lambda e, **kw: "paid.example.com" in e.base_url)
    client = turn_errors.try_failover(cfg)
    assert client is not None
    assert cfg.llm.base_url == "https://paid.example.com/v1"


def test_unpinned_turn_ignores_policy(monkeypatch):
    # An auto-tier turn was never pinned, so the policy does not gate it.
    cfg = _cfg()
    cfg.failover.pinned_policy = "ask"
    _all_live(monkeypatch)
    assert turn_errors.try_failover(cfg) is not None


class _FakeBadRequest(Exception):
    def __init__(self, body):
        self.body = body
        super().__init__("bad request")


def test_model_not_found_classification():
    from agent.core.turn import _is_model_not_found

    observed = _FakeBadRequest({"error": {"code": 400, "type": "invalid_request_error",
                                          "message": "model 'ornith-10-35B' not found"}})
    assert _is_model_not_found(observed) is True
    # A context-window rejection is the same status but not a missing model.
    ctx = _FakeBadRequest({"error": {"code": 400, "type": "exceed_context_size_error",
                                     "message": "the request exceeds the available context size"}})
    assert _is_model_not_found(ctx) is False


def test_no_usable_model_error_accepts_reason():
    cfg = _cfg()
    err = turn_errors.no_usable_model_error(
        cfg, RuntimeError("boom"), reason="endpoint X does not serve model 'y'")
    assert "does not serve model" in str(err)
