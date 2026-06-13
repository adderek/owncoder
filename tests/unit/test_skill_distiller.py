"""Unit tests for memory/skill_distiller — session-end skill self-evolution."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.memory import skill_distiller
from agent.skills import SkillLoader


class _Round:
    def __init__(self, draft="", summary=""):
        self.knowledge_draft = draft
        self.summary = summary


class _FactsStore:
    def __init__(self, round_):
        self._round = round_

    def latest_round(self):
        return self._round


def _make_config(tmp_path, distill=True):
    return SimpleNamespace(
        agent=SimpleNamespace(distill_skills=distill),
        tools=SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        llm=SimpleNamespace(base_url="http://x/v1", api_key="local", model="m"),
    )


def _patch_llm(monkeypatch, payload: str):
    """Patch openai.OpenAI so the client returns `payload` as content."""
    msg = SimpleNamespace(content=payload)
    choice = SimpleNamespace(message=msg)
    resp = SimpleNamespace(choices=[choice])

    class _Completions:
        def create(self, **_):
            return resp

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **_):
            self.chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)


_LONG = "x" * 100  # exceeds the 80-char source threshold


def test_creates_new_skill(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    payload = json.dumps({"skills": [{
        "name": "run-suite",
        "description": "run the integration suite",
        "content": "## Steps\n1. activate venv\n2. pytest",
        "confidence": 0.9,
    }]})
    _patch_llm(monkeypatch, payload)

    n = skill_distiller.distill_session_skills("sess1234", cfg, _FactsStore(_Round(draft=_LONG)))
    assert n == 1
    loader = SkillLoader(cfg)
    assert "run-suite" in [name for name, _ in loader.available()]
    assert "pytest" in loader.load(["run-suite"])


def test_disabled_flag_skips(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, distill=False)
    _patch_llm(monkeypatch, json.dumps({"skills": [{
        "name": "x", "description": "d", "content": "c", "confidence": 0.95}]}))
    assert skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG))) == 0


def test_short_source_skips(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    _patch_llm(monkeypatch, json.dumps({"skills": []}))
    assert skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft="tiny"))) == 0


def test_low_confidence_skipped(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    _patch_llm(monkeypatch, json.dumps({"skills": [{
        "name": "weak", "description": "d", "content": "body", "confidence": 0.5}]}))
    assert skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG))) == 0


def test_existing_skill_refined_bumps_version(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    loader = SkillLoader(cfg)
    loader.save("deploy", "## old\n1. old step", description="deploy")

    payload = json.dumps({"skills": [{
        "name": "deploy",
        "description": "deploy better",
        "content": "## new\n1. improved step\n2. verify",
        "confidence": 0.9,
    }]})
    _patch_llm(monkeypatch, payload)

    n = skill_distiller.distill_session_skills("sess", cfg, _FactsStore(_Round(draft=_LONG)))
    assert n == 1
    hist = loader.history("deploy")
    assert hist[-1]["version"] == 2
    assert "improved step" in loader.load(["deploy"])


def test_identical_body_no_churn(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    loader = SkillLoader(cfg)
    body = "## steps\n1. do the thing\n2. done"
    loader.save("flow", body, description="d")

    _patch_llm(monkeypatch, json.dumps({"skills": [{
        "name": "flow", "description": "d", "content": body, "confidence": 0.95}]}))

    n = skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG)))
    assert n == 0
    # still v1 — no archived history created
    assert loader.history("flow")[-1]["version"] == 1


def test_caps_at_two_per_session(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    skills = [
        {"name": f"s{i}", "description": "d", "content": f"## c{i}\nstep", "confidence": 0.9}
        for i in range(4)
    ]
    _patch_llm(monkeypatch, json.dumps({"skills": skills}))
    n = skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG)))
    assert n == 2


def test_malformed_json_fenced(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    payload = "```json\n" + json.dumps({"skills": [{
        "name": "fenced", "description": "d", "content": "## c\nstep", "confidence": 0.9}]}) + "\n```"
    _patch_llm(monkeypatch, payload)
    n = skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG)))
    assert n == 1


def test_llm_error_returns_zero(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)

    class _Boom:
        def __init__(self, **_):
            raise RuntimeError("no endpoint")

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Boom)
    assert skill_distiller.distill_session_skills("s", cfg, _FactsStore(_Round(draft=_LONG))) == 0
