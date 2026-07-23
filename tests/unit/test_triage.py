"""Tests for LLM triage of security findings (agent.security.triage)."""
from __future__ import annotations

import sys
import types

import pytest

from agent.security import secaudit, triage


def _make_result(tmp_path, body="eval(x)\npickle.loads(y)\n"):
    (tmp_path / "f.py").write_text(body)
    return secaudit.scan(str(tmp_path))


def test_empty_findings_short_circuits(tmp_path):
    (tmp_path / "ok.py").write_text("def f():\n    return 1\n")
    res = secaudit.scan(str(tmp_path))
    assert res.findings == []
    # No LLM call needed; sync wrapper returns the canned message.
    assert triage.run_triage(object(), res) == "No findings to triage."


class _FakeMsg:
    def __init__(self, content): self.message = types.SimpleNamespace(content=content)


class _FakeResp:
    def __init__(self, content): self.choices = [_FakeMsg(content)]


class _FakeClient:
    last_user = None

    def __init__(self, *a, **k):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, *, model, messages, **k):
        _FakeClient.last_user = messages[-1]["content"]
        return _FakeResp("## Top risks\n1. finding 0 — eval RCE")

    async def close(self):
        pass


@pytest.fixture
def _fake_llm(monkeypatch):
    # Fake make_registry().default
    entry = types.SimpleNamespace(base_url="http://localhost:8080/v1", api_key="local", model="m")
    reg = types.SimpleNamespace(default=entry, summarizer=entry, role=lambda *_a, **_k: entry)
    monkeypatch.setattr("agent.config.make_registry", lambda cfg: reg)
    # call_role_with_failover builds its client via make_llm_client (not a
    # bare AsyncOpenAI()) — patch that factory instead of faking the openai
    # module, which would also break the real RateLimitError/… imports it uses.
    monkeypatch.setattr("agent.core.llm_client.make_llm_client",
                        lambda cfg, base_url="", api_key="": _FakeClient())
    return entry


def test_triage_annotates_findings(tmp_path, _fake_llm):
    res = _make_result(tmp_path)
    out = triage.run_triage(object(), res)
    assert "Top risks" in out
    # Findings fed to model include index + severity.
    assert '"i":' in _FakeClient.last_user or '"i": 0' in _FakeClient.last_user
    # Triage must not mutate the deterministic findings.
    assert len(res.findings) >= 2


def test_triage_never_raises_on_client_error(tmp_path, monkeypatch):
    res = _make_result(tmp_path)

    def _boom(cfg, base_url="", api_key=""):
        raise RuntimeError("no endpoint")

    entry = types.SimpleNamespace(base_url="x", api_key="x", model="m")
    monkeypatch.setattr("agent.config.make_registry",
                        lambda cfg: types.SimpleNamespace(default=entry, role=lambda *_a, **_k: entry))
    monkeypatch.setattr("agent.core.llm_client.make_llm_client", _boom)
    out = triage.run_triage(object(), res)
    assert "triage call failed" in out
