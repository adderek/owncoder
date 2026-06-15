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
    # Fake openai.AsyncOpenAI
    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    # Fake make_registry().default
    entry = types.SimpleNamespace(base_url="http://localhost:8080/v1", api_key="local", model="m")
    reg = types.SimpleNamespace(default=entry, summarizer=entry)
    monkeypatch.setattr("agent.config.make_registry", lambda cfg: reg)
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

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no endpoint")

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _Boom
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    entry = types.SimpleNamespace(base_url="x", api_key="x", model="m")
    monkeypatch.setattr("agent.config.make_registry",
                        lambda cfg: types.SimpleNamespace(default=entry))
    out = triage.run_triage(object(), res)
    assert "triage unavailable" in out
