"""Tests for LLM deep-read vulnerability audit (agent.security.review)."""
from __future__ import annotations

import sys
import types

from agent.security import review


def _cfg(tmp_path):
    return types.SimpleNamespace(
        tools=types.SimpleNamespace(working_dir=str(tmp_path), agent_dir=".agent"),
        security=types.SimpleNamespace(airgap=False),
    )


def test_select_files_filters_and_skips(tmp_path):
    (tmp_path / "a.c").write_text("int main(){}\n")
    (tmp_path / "readme.md").write_text("# hi\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("var x=1\n")
    files = review._select_files(str(tmp_path))
    names = {f.name for f in files}
    assert names == {"a.c"}


def test_windows_overlap_for_large_file():
    lines = [f"l{i}" for i in range(800)]
    wins = list(review._windows(lines))
    assert len(wins) >= 2
    # First window starts at line 1.
    assert wins[0][0] == 1
    # Overlap: second window starts before the first one ended.
    first_end = wins[0][0] + len(wins[0][1]) - 1
    assert wins[1][0] <= first_end


def test_parse_handles_fences_and_garbage():
    assert review._parse('```json\n[{"line":1}]\n```') == [{"line": 1}]
    assert review._parse("[]") == []
    assert review._parse("no json here") == []
    assert review._parse('text [{"line":5,"severity":"high"}] tail')[0]["line"] == 5


class _FakeClient:
    payload = '[{"line": 3, "severity": "critical", "class": "overflow", "detail": "no bounds check before memcpy"}]'

    def __init__(self, *a, **k):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **k):
        msg = types.SimpleNamespace(content=_FakeClient.payload)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    async def close(self):
        pass


def _patch_llm(monkeypatch, payload=None):
    if payload is not None:
        _FakeClient.payload = payload
    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake)
    entry = types.SimpleNamespace(base_url="http://localhost:8081/v1", api_key="local", model="m")
    monkeypatch.setattr("agent.config.make_registry",
                        lambda c: types.SimpleNamespace(default=entry))


def test_review_reports_llm_findings(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)
    (tmp_path / "scanner.c").write_text("a\nb\nmemcpy(d,s,n);\nd\n")
    out = review.run_review_command(_cfg(tmp_path), str(tmp_path))
    assert "LLM vulnerability review" in out
    assert "LLM-REPORTED, UNVERIFIED" in out
    assert "overflow" in out
    assert "critical" in out


def test_review_empty_when_model_finds_nothing(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, payload="[]")
    (tmp_path / "x.c").write_text("int safe(void){return 0;}\n")
    out = review.run_review_command(_cfg(tmp_path), str(tmp_path))
    assert "No vulnerabilities reported" in out
    assert "NOT proof of safety" in out


def test_review_no_source_files(tmp_path, monkeypatch):
    _patch_llm(monkeypatch)
    (tmp_path / "readme.md").write_text("# doc\n")
    out = review.run_review_command(_cfg(tmp_path), str(tmp_path))
    assert "No source files" in out


def test_airgap_refuses_remote(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.security.airgap = True
    (tmp_path / "x.c").write_text("int main(){}\n")
    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake)
    entry = types.SimpleNamespace(base_url="https://api.example.com", api_key="k", model="m")
    monkeypatch.setattr("agent.config.make_registry",
                        lambda c: types.SimpleNamespace(default=entry))
    out = review.run_review_command(cfg, str(tmp_path))
    assert "air-gap" in out


def test_missing_path(tmp_path):
    out = review.run_review_command(_cfg(tmp_path), str(tmp_path / "nope"))
    assert "path not found" in out
