"""Tests for the eval LLM-as-judge scorer (evals/judge.py) and its wiring
into evals/run.py."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evals"))

import judge  # noqa: E402
import run as eval_run  # noqa: E402


def _make_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ── parse_score ───────────────────────────────────────────────────────────


class TestParseScore:
    def test_bare_json(self):
        assert judge.parse_score('{"score": 8, "rationale": "clean"}') == (8, "clean")

    def test_json_in_fence(self):
        text = 'Here you go:\n```json\n{"score": 5, "rationale": "meh"}\n```'
        assert judge.parse_score(text) == (5, "meh")

    def test_regex_fallback(self):
        score, rationale = judge.parse_score("I rate this score: 7 overall")
        assert score == 7
        assert rationale == ""

    def test_clamped_to_range(self):
        assert judge.parse_score('{"score": 15, "rationale": ""}')[0] == 10
        assert judge.parse_score('{"score": -3, "rationale": ""}')[0] == 0

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            judge.parse_score("great work, ten out of ten")


# ── diff mechanics ────────────────────────────────────────────────────────


SAMPLE_DIFF = """\
diff -ruN a/calc.py b/calc.py
--- a/calc.py\t2026-01-01
+++ b/calc.py\t2026-01-01
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a - b
+    return a + b
diff -ruN a/test_calc.py b/test_calc.py
--- a/test_calc.py\t2026-01-01
+++ b/test_calc.py\t2026-01-01
@@ -1,2 +1,2 @@
-def test_add():
+def test_add_renamed():
"""


class TestDiffStats:
    def test_counts_and_files(self):
        stats = judge.diff_stats(SAMPLE_DIFF)
        assert stats["files"] == ["calc.py", "test_calc.py"]
        assert stats["added"] == 2
        assert stats["removed"] == 2
        assert stats["total_lines"] == 4

    def test_empty_diff(self):
        stats = judge.diff_stats("")
        assert stats["files"] == []
        assert stats["total_lines"] == 0


class TestCheckForbidden:
    def test_glob_hit(self):
        assert judge.check_forbidden(SAMPLE_DIFF, ["test_*.py"]) == ["test_calc.py"]

    def test_no_hit(self):
        assert judge.check_forbidden(SAMPLE_DIFF, ["docs/*"]) == []


class TestComputeDiff(object):
    def test_paths_normalized_and_agent_excluded(self, tmp_path):
        fixture = tmp_path / "fixture"
        workspace = tmp_path / "ws"
        fixture.mkdir()
        workspace.mkdir()
        (fixture / "a.py").write_text("x = 1\n")
        (workspace / "a.py").write_text("x = 2\n")
        (workspace / ".agent").mkdir()
        (workspace / ".agent" / "junk").write_text("ignore me")
        diff = judge.compute_diff(fixture, workspace)
        assert "a/a.py" in diff and "b/a.py" in diff
        assert str(tmp_path) not in diff
        assert ".agent/junk" not in diff and "ignore me" not in diff

    def test_identical_dirs_empty(self, tmp_path):
        fixture = tmp_path / "fixture"
        workspace = tmp_path / "ws"
        fixture.mkdir()
        workspace.mkdir()
        (fixture / "a.py").write_text("x = 1\n")
        (workspace / "a.py").write_text("x = 1\n")
        assert judge.compute_diff(fixture, workspace) == ""


# ── judge prompt construction (anti-gaming) ───────────────────────────────


class TestBuildJudgeMessages:
    def test_contains_only_prompt_and_diff(self):
        msgs = judge.build_judge_messages("fix", "fix the bug", SAMPLE_DIFF)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "fix the bug" in msgs[1]["content"]
        assert "return a + b" in msgs[1]["content"]

    def test_unknown_type_falls_back(self):
        msgs = judge.build_judge_messages("nonsense", "p", "d")
        assert "Task type: FIX" in msgs[1]["content"]

    def test_long_diff_clipped(self):
        big = "x" * (judge._DIFF_PROMPT_CAP + 500)
        msgs = judge.build_judge_messages("fix", "p", big)
        assert "[diff clipped]" in msgs[1]["content"]


# ── judge_task ────────────────────────────────────────────────────────────


class TestJudgeTask:
    def _run(self, task, diff, response_text='{"score": 9, "rationale": "ok"}',
             side_effect=None):
        mock = AsyncMock()
        if side_effect is not None:
            mock.side_effect = side_effect
        else:
            mock.return_value = (_make_response(response_text), "entryA", object())
        with patch("agent.core.llm_retry.call_role_with_failover", mock), \
             patch("agent.security.airgap.is_enabled", return_value=False):
            result = asyncio.run(judge_task_call(task, diff))
        return result, mock

    def test_empty_diff_scores_zero_without_llm(self):
        result, mock = self._run({"prompt": "p"}, "")
        assert result["score"] == 0
        assert result["rationale"] == "empty diff"
        mock.assert_not_called()

    def test_forbidden_path_scores_zero_without_llm(self):
        task = {"prompt": "p", "judge": {"forbid_paths": ["test_*.py"]}}
        result, mock = self._run(task, SAMPLE_DIFF)
        assert result["score"] == 0
        assert "test_calc.py" in result["rationale"]
        mock.assert_not_called()

    def test_normal_scoring(self):
        result, mock = self._run({"prompt": "p"}, SAMPLE_DIFF)
        assert result["score"] == 9
        assert result["rationale"] == "ok"
        assert result["entry"] == "entryA"
        mock.assert_called_once()

    def test_oversized_diff_clamps_score(self):
        task = {"prompt": "p", "judge": {"max_diff_lines": 2}}
        result, _ = self._run(task, SAMPLE_DIFF)
        assert result["oversized"] is True
        assert result["score"] == judge.OVERSIZED_SCORE_CAP
        assert "clamped" in result["rationale"]

    def test_llm_failure_reported_not_raised(self):
        result, _ = self._run({"prompt": "p"}, SAMPLE_DIFF,
                              side_effect=RuntimeError("boom"))
        assert result["score"] is None
        assert "boom" in result["error"]

    def test_airgap_passed_as_local_only(self):
        mock = AsyncMock(return_value=(
            _make_response('{"score": 7, "rationale": ""}'), "e", object()))
        with patch("agent.core.llm_retry.call_role_with_failover", mock), \
             patch("agent.security.airgap.is_enabled", return_value=True):
            asyncio.run(judge_task_call({"prompt": "p"}, SAMPLE_DIFF))
        assert mock.call_args.kwargs["local_only"] is True

    def test_judge_never_sees_candidate_commentary(self):
        # The messages are built from task prompt + diff only; there is no
        # code path that could include agent stdout. Assert the actual sent
        # messages contain nothing beyond those two inputs.
        task = {"prompt": "UNIQUE_TASK_PROMPT"}
        result, mock = self._run(task, SAMPLE_DIFF)
        sent = mock.call_args.kwargs["messages"]
        user_content = sent[1]["content"]
        assert "UNIQUE_TASK_PROMPT" in user_content
        assert "return a + b" in user_content


def judge_task_call(task, diff):
    return judge.judge_task(SimpleNamespace(), task, diff)


# ── baseline compare ──────────────────────────────────────────────────────


def _payload(*rows):
    return {"results": [dict(r) for r in rows]}


class TestCompareToBaseline:
    def test_no_regression(self):
        base = _payload({"id": "t1", "status": "pass", "judge": {"score": 8}})
        cur = _payload({"id": "t1", "status": "pass", "judge": {"score": 7}})
        assert judge.compare_to_baseline(cur, base) == []

    def test_mechanical_flip(self):
        base = _payload({"id": "t1", "status": "pass"})
        cur = _payload({"id": "t1", "status": "fail"})
        regs = judge.compare_to_baseline(cur, base)
        assert regs == ["t1: mechanical pass -> fail"]

    def test_judged_drop_over_threshold(self):
        base = _payload({"id": "t1", "status": "pass", "judge": {"score": 9}})
        cur = _payload({"id": "t1", "status": "pass", "judge": {"score": 6}})
        regs = judge.compare_to_baseline(cur, base)
        assert regs == ["t1: judged score 9 -> 6"]

    def test_judged_drop_exactly_threshold_ok(self):
        base = _payload({"id": "t1", "status": "pass", "judge": {"score": 9}})
        cur = _payload({"id": "t1", "status": "pass", "judge": {"score": 7}})
        assert judge.compare_to_baseline(cur, base) == []

    def test_new_and_removed_tasks_ignored(self):
        base = _payload({"id": "gone", "status": "pass"})
        cur = _payload({"id": "new", "status": "fail"})
        assert judge.compare_to_baseline(cur, base) == []

    def test_missing_judge_blocks_score_compare(self):
        base = _payload({"id": "t1", "status": "pass", "judge": {"score": 9}})
        cur = _payload({"id": "t1", "status": "pass", "judge": None})
        assert judge.compare_to_baseline(cur, base) == []


# ── run.py wiring ─────────────────────────────────────────────────────────


class TestRunnerWiring:
    def _dirs(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        fixtures_dir = tmp_path / "fixtures"
        tasks_dir.mkdir()
        (fixtures_dir / "t1").mkdir(parents=True)
        (fixtures_dir / "t1" / "a.txt").write_text("before\n")
        (tasks_dir / "t1.yaml").write_text(
            "id: t1\nprompt: change a.txt\nfixture: t1\n"
            "checks:\n  - type: file_contains\n    path: a.txt\n    text: after\n")
        return tasks_dir, fixtures_dir

    def test_judge_disabled_by_default(self, tmp_path):
        tasks_dir, fixtures_dir = self._dirs(tmp_path)
        rc = eval_run.main([
            "--tasks-dir", str(tasks_dir), "--fixtures-dir", str(fixtures_dir),
            "--agent-cmd", "sh -c 'echo after > a.txt' --",
        ])
        assert rc == 0

    def test_judge_result_in_json_with_stub_config(self, tmp_path):
        tasks_dir, fixtures_dir = self._dirs(tmp_path)
        json_path = tmp_path / "out.json"
        fake_judge = {"score": 8, "rationale": "fine", "entry": "e",
                      "oversized": False, "forbidden": [],
                      "diff_stats": {}, "error": ""}
        with patch.object(eval_run, "_judge_workspace", return_value=fake_judge), \
             patch.object(judge, "load_agent_config", return_value=SimpleNamespace()):
            rc = eval_run.main([
                "--tasks-dir", str(tasks_dir), "--fixtures-dir", str(fixtures_dir),
                "--agent-cmd", "sh -c 'echo after > a.txt' --",
                "--judge", "--json", str(json_path),
            ])
        assert rc == 0
        payload = json.loads(json_path.read_text())
        assert payload["results"][0]["judge"]["score"] == 8

    def test_baseline_regression_fails_run(self, tmp_path):
        tasks_dir, fixtures_dir = self._dirs(tmp_path)
        baseline = tmp_path / "base.json"
        baseline.write_text(json.dumps(
            {"results": [{"id": "t1", "status": "pass"}]}))
        rc = eval_run.main([
            "--tasks-dir", str(tasks_dir), "--fixtures-dir", str(fixtures_dir),
            "--agent-cmd", "true",  # agent does nothing -> check fails
            "--baseline", str(baseline),
        ])
        assert rc == 1

    def test_baseline_clean_passes_even_if_scores_dip_slightly(self, tmp_path):
        tasks_dir, fixtures_dir = self._dirs(tmp_path)
        baseline = tmp_path / "base.json"
        baseline.write_text(json.dumps(
            {"results": [{"id": "t1", "status": "pass"}]}))
        rc = eval_run.main([
            "--tasks-dir", str(tasks_dir), "--fixtures-dir", str(fixtures_dir),
            "--agent-cmd", "sh -c 'echo after > a.txt' --",
            "--baseline", str(baseline),
        ])
        assert rc == 0
