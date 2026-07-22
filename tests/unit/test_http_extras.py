"""Tests for the HTTP-mode polish/sidecar additions:
agent/metrics/model_calls.session_cost_usd, agent/ui/http_loop._HttpUI.diff_info,
agent/ui/http_sidecar._SidecarServer.chat fanout."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as N

import pytest

from agent.metrics import model_calls as mc


@pytest.fixture(autouse=True)
def _reset():
    mc.run_modelcalls_command("reset")
    mc.reset_round()
    yield
    mc.run_modelcalls_command("reset")
    mc.reset_round()


# ---------------------------------------------------------------------------
# session_cost_usd
# ---------------------------------------------------------------------------

def _entry(**kw):
    base = dict(cost_in_per_1k=0.0, cost_out_per_1k=0.0)
    base.update(kw)
    return N(**base)


def test_session_cost_usd_sums_priced_models():
    mc.record("paid", role="main", model="gpt-x", in_tokens=10_000, out_tokens=2_000)
    cfg = N(model_entries={"gpt-x": _entry(cost_in_per_1k=0.005, cost_out_per_1k=0.015)})
    assert mc.session_cost_usd(cfg) == pytest.approx(10_000 / 1000 * 0.005 + 2_000 / 1000 * 0.015)


def test_session_cost_usd_zero_for_unpriced_or_local_models():
    mc.record("local", role="main", model="llama-local", in_tokens=5_000, out_tokens=1_000)
    cfg = N(model_entries={"llama-local": _entry()})  # 0.0 pricing
    assert mc.session_cost_usd(cfg) == 0.0


def test_session_cost_usd_skips_unknown_model():
    mc.record("paid", role="main", model="mystery-model", in_tokens=1_000, out_tokens=1_000)
    cfg = N(model_entries={})  # entry not registered
    assert mc.session_cost_usd(cfg) == 0.0


# ---------------------------------------------------------------------------
# _HttpUI.diff_info — path-traversal guard
# ---------------------------------------------------------------------------

def _make_http_ui():
    from agent.ui.http_loop import _HttpUI

    class _FakeServer:
        def get_ui_config(self, session_id=""):
            return {}

    loop = asyncio.new_event_loop()
    try:
        return _HttpUI(_FakeServer(), None, loop)
    finally:
        loop.close()


def test_diff_info_rejects_path_traversal():
    ui = _make_http_ui()
    for bad in ("../etc/passwd", "a/../../b", "..", ""):
        result = ui.diff_info(bad)
        assert result["diff"] == ""
        assert result.get("error") == "invalid path"


def test_diff_info_accepts_plain_relative_path(tmp_path, monkeypatch):
    ui = _make_http_ui()
    monkeypatch.setattr(ui, "workdir", lambda: str(tmp_path))
    # No git repo at tmp_path — git exits non-zero, diff comes back empty,
    # but the call must not raise and must not report the path as invalid.
    result = ui.diff_info("some/file.py")
    assert result["path"] == "some/file.py"
    assert "error" not in result or result.get("error") != "invalid path"


# ---------------------------------------------------------------------------
# _HttpUI.upload_file — attachment save + sanitization
# ---------------------------------------------------------------------------

def test_upload_file_sanitizes_name_and_saves_content(tmp_path):
    import base64

    ui = _make_http_ui()
    ui.workdir = lambda: str(tmp_path)
    data = base64.b64encode(b"payload bytes").decode()
    result = ui.upload_file("../../etc/evil name.png", data)
    assert result["ok"] is True
    assert ".." not in result["path"]
    saved = tmp_path / result["path"]
    assert saved.exists()
    assert saved.read_bytes() == b"payload bytes"
    assert saved.parent == tmp_path / ".agent" / "uploads"


def test_upload_file_rejects_oversize():
    import base64

    ui = _make_http_ui()
    big = base64.b64encode(b"x" * (21 * 1024 * 1024)).decode()
    result = ui.upload_file("big.bin", big)
    assert result["ok"] is False


def test_upload_file_rejects_bad_base64():
    ui = _make_http_ui()
    result = ui.upload_file("f.txt", "not-base64!!")
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# _SidecarServer.chat — event fanout + busy tracking
# ---------------------------------------------------------------------------

class _FakeInner:
    def __init__(self):
        self.injected = []

    async def chat(self, text, session_id="", on_token=None, on_tool_call=None,
                    on_tool_result=None, on_usage=None, on_progress=None,
                    on_loop_detected=None, on_phase=None, on_reasoning=None,
                    on_context_size=None, on_user_message=None, on_signal=None,
                    source="terminal"):
        if on_token:
            on_token("hel")
            on_token("lo")
        if on_tool_call:
            on_tool_call("read_file", '{"path": "x.py"}')
        if on_tool_result:
            on_tool_result("read_file", True)
        return "hello"

    def token_estimate(self, session_id=""):
        return 99

    def get_llm_info(self, session_id=""):
        return {"model": "m", "ctx_window": 1000, "compaction_threshold": 0.75}

    def inject(self, text, session_id=""):
        self.injected.append(text)


@pytest.mark.asyncio
async def test_sidecar_server_fanouts_events_and_tracks_busy():
    from agent.ui.http_sidecar import _SidecarServer

    inner = _FakeInner()
    wrapped = _SidecarServer(inner)
    q = wrapped.bus.subscribe()

    assert wrapped.busy is False
    seen_tokens = []
    busy_during = []

    def _on_token(t):
        seen_tokens.append(t)
        busy_during.append(wrapped.busy)

    response = await wrapped.chat("hi", on_token=_on_token)
    assert response == "hello"
    assert wrapped.busy is False
    assert busy_during == [True, True]  # busy for the whole in-flight turn

    # Caller's own callback still fires alongside the bus mirror.
    assert seen_tokens == ["hel", "lo"]

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    types = [e["type"] for e in events]
    assert types[0] == "user"
    assert "token" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "response" in types
    assert types[-1] == "state" and events[-1]["state"] == "idle"


@pytest.mark.asyncio
async def test_sidecar_server_delegates_unknown_attrs_to_inner():
    from agent.ui.http_sidecar import _SidecarServer

    inner = _FakeInner()
    wrapped = _SidecarServer(inner)
    wrapped.inject("nudge")
    assert inner.injected == ["nudge"]
    assert wrapped.token_estimate() == 99
