"""Per-endpoint probe for `tool_choice: "required"`.

"required" is not one mechanism. On llama.cpp it is a sampling constraint built
from the tool schemas, so a hallucinated call is unreachable. In the cloud it is an
API contract, and a contract can be honoured by returning tool_calls with the
content dropped — which silently removes the model's ability to explain. The probe
therefore asserts four things, and anything short of all four degrades to "auto",
leaving the nudge ladder as the backstop.
"""
from __future__ import annotations

import io
import json
import types

import pytest

from agent.config import model_probe as mp


def _entry(base_url="http://x/v1", model="m", api_key=""):
    return types.SimpleNamespace(base_url=base_url, model=model, api_key=api_key)


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch(monkeypatch, payload=None, exc=None, capture=None):
    def fake(req, timeout=None):
        if capture is not None:
            capture.append(json.loads(req.data.decode()))
        if exc is not None:
            raise exc
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(mp.urllib.request, "urlopen", fake)


def _msg(content, calls):
    return {"choices": [{"message": {"content": content, "tool_calls": calls}}]}


_CALL = [{"function": {"name": "no_tool_needed", "arguments": '{"reason":"x"}'}}]


@pytest.fixture(autouse=True)
def _clear():
    mp.clear_tool_choice_cache()
    yield
    mp.clear_tool_choice_cache()


def test_required_when_call_and_prose_both_come_back(monkeypatch):
    _patch(monkeypatch, _msg("A race condition is ...", _CALL))
    assert mp.tool_choice_support(_entry()) == "required"


def test_auto_when_the_endpoint_rejects_required(monkeypatch):
    """The common 4xx: required unsupported, or required+tools unsupported."""
    _patch(monkeypatch, exc=Exception("HTTP 400"))
    assert mp.tool_choice_support(_entry()) == "auto"


def test_auto_when_prose_is_dropped(monkeypatch):
    """Contract honoured, explanation channel eaten — the failure the probe exists
    for, and the one a status-code check would miss."""
    _patch(monkeypatch, _msg("", _CALL))
    assert mp.tool_choice_support(_entry()) == "auto"


def test_auto_when_no_call_is_returned(monkeypatch):
    """required asked for, none delivered: the contract is not being enforced."""
    _patch(monkeypatch, _msg("A race condition is ...", []))
    assert mp.tool_choice_support(_entry()) == "auto"


def test_auto_on_a_malformed_response(monkeypatch):
    _patch(monkeypatch, {"unexpected": True})
    assert mp.tool_choice_support(_entry()) == "auto"


def test_probe_sends_required_tools_and_a_nonzero_temperature(monkeypatch):
    """Temperature matters: prose sits before the tool-call section, so an endpoint
    can pass at temperature 0 and drop it once sampling is in play."""
    seen: list[dict] = []
    _patch(monkeypatch, _msg("ok", _CALL), capture=seen)
    mp.tool_choice_support(_entry())
    body = seen[0]
    assert body["tool_choice"] == "required"
    assert [t["function"]["name"] for t in body["tools"]] == ["read_file", "no_tool_needed"]
    assert body["temperature"] > 0


def test_verdict_is_cached_per_endpoint_and_model(monkeypatch):
    seen: list[dict] = []
    _patch(monkeypatch, _msg("ok", _CALL), capture=seen)
    e = _entry()
    assert mp.tool_choice_support(e) == "required"
    assert mp.tool_choice_support(e) == "required"
    assert len(seen) == 1, "second call should be served from cache"
    # a different model on the same endpoint is a separate question
    mp.tool_choice_support(_entry(model="other"))
    assert len(seen) == 2


def test_no_base_url_is_auto_without_a_request(monkeypatch):
    _patch(monkeypatch, exc=AssertionError("must not be called"))
    assert mp.tool_choice_support(_entry(base_url="")) == "auto"


def test_rate_limited_endpoint_is_not_probed(monkeypatch):
    monkeypatch.setattr(mp, "is_rate_limited", lambda *a, **k: True)
    _patch(monkeypatch, exc=AssertionError("must not be called"))
    assert mp.tool_choice_support(_entry()) == "auto"


# ── the send site ───────────────────────────────────────────────────────────
# Two gates, both required: the config flag opts in, the probe confirms this
# endpoint honours "required" without dropping content. Either one saying no
# leaves the request exactly as it was, so the nudge ladder stays the backstop.

def _cfg(mode):
    llm = types.SimpleNamespace(model="m", max_output_tokens=1024, ctx_window=8192,
                                temperature=0.2, tool_choice_required=mode,
                                base_url="http://x/v1", api_key="")
    return types.SimpleNamespace(llm=llm)


@pytest.mark.parametrize("mode,verdict,expected", [
    ("off",  "required", None),        # not opted in: never sent
    ("off",  "auto",     None),
    ("auto", "auto",     None),        # opted in, endpoint unsafe: still not sent
    ("auto", "required", "required"),  # both gates pass
])
def test_tool_choice_is_sent_only_when_both_gates_pass(monkeypatch, mode, verdict, expected):
    from agent.core import prompts
    monkeypatch.setattr(mp, "tool_choice_support", lambda e, **k: verdict)
    kw = prompts._build_call_kwargs(_cfg(mode))
    assert kw.get("tool_choice") == expected


def test_a_probe_failure_does_not_break_the_request(monkeypatch):
    """The probe is best-effort; an exception must leave the call unchanged rather
    than take down the turn."""
    from agent.core import prompts
    def boom(*a, **k):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(mp, "tool_choice_support", boom)
    kw = prompts._build_call_kwargs(_cfg("auto"))
    assert "tool_choice" not in kw
    assert kw["model"] == "m"
