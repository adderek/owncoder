"""Unit tests for agent/tools/review_changes/ — pre-commit diff review."""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.tools.review_changes.main import (
    _collect_diff,
    _parse_review,
    _pick_reviewer,
    _truncate_by_file,
    review_changes,
    setup,
)


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    import agent.tools.review_changes.main as mod
    monkeypatch.setattr(mod, "_config", None)
    yield


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def _make_config(repo: Path):
    cfg = MagicMock()
    cfg.tools.working_dir = str(repo)
    cfg.tools.agent_dir = ".agent"
    return cfg


class TestCollectDiff:
    def test_unstaged_change_included(self, repo):
        (repo / "a.py").write_text("x = 2\n")
        diff, _notes = _collect_diff(str(repo), staged=False)
        assert "x = 2" in diff

    def test_untracked_file_included_as_pseudo_diff(self, repo):
        (repo / "new.py").write_text("y = 3\n")
        diff, _notes = _collect_diff(str(repo), staged=False)
        assert "new.py" in diff
        assert "+y = 3" in diff

    def test_staged_only(self, repo):
        (repo / "a.py").write_text("x = 2\n")
        _git(repo, "add", "a.py")
        (repo / "a.py").write_text("x = 3\n")
        diff, _ = _collect_diff(str(repo), staged=True)
        assert "x = 2" in diff and "x = 3" not in diff


class TestTruncateByFile:
    def test_under_budget_untouched(self):
        diff = "diff --git a/x b/x\n+1\n"
        out, dropped = _truncate_by_file(diff, 1000)
        assert out == diff and dropped == []

    def test_drops_whole_sections(self):
        s1 = "diff --git a/x b/x\n" + "+a\n" * 10
        s2 = "diff --git a/y b/y\n" + "+b\n" * 10
        out, dropped = _truncate_by_file(s1 + s2, len(s1) + 5)
        assert "a/x" in out and "a/y" not in out
        assert len(dropped) == 1


class TestParseReview:
    def test_plain_json(self):
        obj = _parse_review(json.dumps({"summary": "ok", "findings": []}))
        assert obj["summary"] == "ok"

    def test_fenced_json(self):
        raw = "```json\n{\"summary\": \"ok\", \"findings\": []}\n```"
        assert _parse_review(raw)["summary"] == "ok"

    def test_leading_prose(self):
        raw = "Here is my review: {\"summary\": \"fine\", \"findings\": []}"
        assert _parse_review(raw)["summary"] == "fine"

    def test_garbage_returns_none(self):
        assert _parse_review("looks good to me!") is None

    def test_non_list_findings_coerced(self):
        obj = _parse_review('{"summary": "s", "findings": "none"}')
        assert obj["findings"] == []


class TestPickReviewer:
    def test_prefers_equal_or_stronger_non_active(self):
        cfg = MagicMock()
        cfg.llm.base_url = "http://active:1/v1"
        cfg.llm.model = "m-active"
        other = MagicMock(base_url="http://other:1/v1", model="m-big")
        active = MagicMock(base_url="http://active:1/v1", model="m-active")
        cfg.model_entries = {"other": other, "active": active}
        with patch("agent.core.model_tier.build_ladder",
                   return_value=[("active", 5.0), ("other", 9.0)]):
            name, entry = _pick_reviewer(cfg)
        assert name == "other"

    def test_weaker_alternative_rejected_for_self_review(self):
        # Only non-active entry is weaker than the author → flagged self-review
        # beats a noisy weak-model review.
        cfg = MagicMock()
        cfg.llm.base_url = "http://active:1/v1"
        cfg.llm.model = "m-active"
        weak = MagicMock(base_url="http://other:1/v1", model="m-weak")
        strong = MagicMock(base_url="http://active:1/v1", model="m-active")
        cfg.model_entries = {"weak": weak, "strong": strong}
        with patch("agent.core.model_tier.build_ladder",
                   return_value=[("weak", 1.0), ("strong", 9.0)]):
            name, entry = _pick_reviewer(cfg)
        assert name == "strong"

    def test_falls_back_to_active_when_alone(self):
        cfg = MagicMock()
        cfg.llm.base_url = "http://active:1/v1"
        cfg.llm.model = "m"
        only = MagicMock(base_url="http://active:1/v1", model="m")
        cfg.model_entries = {"only": only}
        with patch("agent.core.model_tier.build_ladder", return_value=[("only", 5.0)]):
            name, _ = _pick_reviewer(cfg)
        assert name == "only"

    def test_empty_ladder_returns_none(self):
        cfg = MagicMock()
        with patch("agent.core.model_tier.build_ladder", return_value=[]):
            assert _pick_reviewer(cfg) is None


class TestReviewChangesTool:
    def test_not_a_repo(self, tmp_path):
        setup(_make_config(tmp_path))
        result = asyncio.run(review_changes(path=str(tmp_path)))
        assert "error" in result

    def test_clean_tree_short_circuits(self, repo):
        setup(_make_config(repo))
        result = asyncio.run(review_changes(path=str(repo)))
        assert result["findings"] == []
        assert "No unstaged changes" in result["summary"]

    def test_degrades_without_reviewer(self, repo):
        cfg = _make_config(repo)
        setup(cfg)
        (repo / "a.py").write_text("x = 2\n")
        with patch("agent.tools.review_changes.main._pick_reviewer", return_value=None), \
             patch("agent.tools.review_changes.main._static_findings", return_value=([], [])):
            result = asyncio.run(review_changes(path=str(repo)))
        assert result["degraded"] is True
        assert "static findings only" in result["summary"]

    def test_llm_review_report(self, repo):
        cfg = _make_config(repo)
        setup(cfg)
        cfg.llm.base_url = "http://active:1/v1"
        cfg.llm.model = "m-active"
        (repo / "a.py").write_text("x = 2\n")
        entry = MagicMock(base_url="http://other:1/v1", model="m-rev")
        parsed = {"summary": "one bug",
                  "findings": [{"file": "a.py", "line": 1, "severity": "high",
                                "kind": "bug", "note": "wrong value"}]}
        with patch("agent.tools.review_changes.main._pick_reviewer",
                   return_value=("rev", entry)), \
             patch("agent.tools.review_changes.main._static_findings",
                   return_value=([], [])), \
             patch("agent.tools.review_changes.main._llm_review",
                   new=AsyncMock(return_value=(parsed, None))):
            result = asyncio.run(review_changes(path=str(repo)))
        assert result["degraded"] is False
        assert result["reviewed_by"] == "rev"
        assert result["self_review"] is False
        assert result["findings"][0]["note"] == "wrong value"

    def test_reviewer_call_failure_degrades(self, repo):
        cfg = _make_config(repo)
        setup(cfg)
        (repo / "a.py").write_text("x = 2\n")
        entry = MagicMock(base_url="http://other:1/v1", model="m-rev")
        with patch("agent.tools.review_changes.main._pick_reviewer",
                   return_value=("rev", entry)), \
             patch("agent.tools.review_changes.main._static_findings",
                   return_value=([], [])), \
             patch("agent.tools.review_changes.main._llm_review",
                   new=AsyncMock(return_value=(None, "ConnectError: down"))):
            result = asyncio.run(review_changes(path=str(repo)))
        assert result["degraded"] is True
        assert "ConnectError" in result["summary"]
