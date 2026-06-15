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


def test_estimate_counts_files_and_windows(tmp_path):
    (tmp_path / "small.c").write_text("\n".join(f"l{i}" for i in range(10)))
    big = "\n".join(f"l{i}" for i in range(900))
    (tmp_path / "big.c").write_text(big)
    nfiles, nwin = review.estimate(_cfg(tmp_path), str(tmp_path))
    assert nfiles == 2
    assert nwin >= 3  # small=1 window, big=multiple


def test_estimate_caps_at_max_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "_MAX_WINDOWS", 5)
    monkeypatch.setattr(review, "_WINDOW_LINES", 10)
    monkeypatch.setattr(review, "_OVERLAP", 0)
    (tmp_path / "huge.c").write_text("\n".join(f"l{i}" for i in range(1000)))
    _, nwin = review.estimate(_cfg(tmp_path), str(tmp_path))
    assert nwin == 5


def test_progress_callback_fires(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, payload="[]")
    (tmp_path / "a.c").write_text("\n".join(f"l{i}" for i in range(5)))
    msgs = []
    review.run_review_command(_cfg(tmp_path), str(tmp_path), msgs.append)
    assert msgs and any("a.c" in m for m in msgs)
    assert any(m.startswith("[1/") for m in msgs)


def test_incremental_skips_unchanged(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, payload="[]")
    cfg = _cfg(tmp_path)
    (tmp_path / "a.c").write_text("int x;\n")
    # First full review of '.' records state.
    review.run_review_command(cfg, ".")
    # No-arg incremental: nothing changed -> skipped.
    out = review.run_review_command(cfg, "")
    assert "Nothing changed" in out or "already reviewed" in out


def test_incremental_reviews_modified(tmp_path, monkeypatch):
    _patch_llm(monkeypatch, payload="[]")
    cfg = _cfg(tmp_path)
    f = tmp_path / "a.c"
    f.write_text("int x;\n")
    review.run_review_command(cfg, ".")
    import os, time
    # Bump mtime to look modified.
    future = time.time() + 100
    os.utime(f, (future, future))
    out = review.run_review_command(cfg, "")
    assert "LLM vulnerability review" in out  # ran, did not skip


def test_resolve_target_confined_by_grants(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    # Simulate configured policy that denies a path outside the project.
    monkeypatch.setattr("agent.security.policy.is_configured", lambda: True)
    monkeypatch.setattr("agent.security.path_grants.grant_for", lambda p: None)
    p, err = review._resolve_target(cfg, "/etc")
    assert p is None
    assert "not granted" in err
    assert "/paths add" in err


def test_resolve_target_dot_is_workdir(tmp_path):
    cfg = _cfg(tmp_path)
    p, err = review._resolve_target(cfg, ".")
    assert err is None
    assert p == tmp_path.resolve()


def test_boundary_windows_cut_at_function_starts(monkeypatch):
    monkeypatch.setattr(review, "_WINDOW_LINES", 10)
    lines = [f"l{i}" for i in range(30)]
    # Function starts at lines 8 and 18 (0-based).
    wins = list(review._boundary_windows(lines, [8, 18]))
    # Every line covered exactly once, in order.
    rebuilt = []
    for bl, chunk in wins:
        rebuilt += chunk
    assert rebuilt == lines
    # A cut lands on a boundary (window 1 ends at line 8).
    starts = [bl for bl, _ in wins]
    assert 9 in starts or 19 in starts  # 1-based start after a boundary cut


def test_boundary_windows_no_boundaries_still_covers(monkeypatch):
    monkeypatch.setattr(review, "_WINDOW_LINES", 10)
    lines = [f"l{i}" for i in range(25)]
    wins = list(review._boundary_windows(lines, []))
    rebuilt = []
    for _, chunk in wins:
        rebuilt += chunk
    assert rebuilt == lines


def test_file_windows_fallback_without_rag(tmp_path):
    # _cfg has no .rag -> falls back to line windows, still covers the file.
    f = tmp_path / "x.c"
    f.write_text("\n".join(f"l{i}" for i in range(50)))
    wins = review._file_windows(_cfg(tmp_path), f)
    assert wins
    total = sum(len(c) for _, c in wins)
    assert total >= 50


def test_persist_and_diff_new_fixed(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    (tmp_path / "a.c").write_text("memcpy(d,s,n);\n")
    # First run reports one finding.
    _patch_llm(monkeypatch, payload='[{"line":1,"severity":"high","class":"overflow","detail":"x"}]')
    out1 = review.run_review_command(cfg, ".")
    assert "reported issues: 1" in out1
    # Second run reports nothing -> the prior finding is "fixed/gone".
    _patch_llm(monkeypatch, payload="[]")
    out2 = review.run_review_command(cfg, ".")
    assert "-1 fixed" in out2


def test_clear_history(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    (tmp_path / "a.c").write_text("int x;\n")
    _patch_llm(monkeypatch, payload="[]")
    review.run_review_command(cfg, ".")
    assert review._history_dir(cfg).exists()
    out = review.run_review_command(cfg, "clear")
    assert "Cleared" in out
    assert not review._history_dir(cfg).exists()
    assert not review._state_path(cfg).exists()


def test_history_listing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    (tmp_path / "a.c").write_text("int x;\n")
    _patch_llm(monkeypatch, payload="[]")
    assert "No review history" in review.run_review_command(cfg, "history")
    review.run_review_command(cfg, ".")
    assert "Review history:" in review.run_review_command(cfg, "history")
