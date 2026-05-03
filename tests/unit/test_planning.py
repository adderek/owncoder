from __future__ import annotations

import json
import pytest

from agent.planning import plan as plan_mod
from agent.planning import recovery
from agent.planning import dag as dag_mod
from agent.planning.compact import compact_plan_sync


@pytest.fixture
def tmp_agent_dir(tmp_path, monkeypatch):
    plan_mod.configure(str(tmp_path), ".agent")
    recovery.configure(str(tmp_path), ".agent")
    yield tmp_path


def test_create_plan_persists(tmp_agent_dir):
    p = plan_mod.create_plan(
        "Add new parser",
        session_id="S1",
        steps=[
            "Write red test for parser skeleton",
            {"description": "Implement parser", "tests": ["unit parses happy path"]},
        ],
    )
    assert p.id
    assert len(p.steps) == 2
    loaded = plan_mod.load_plan(p.id)
    assert loaded is not None
    assert loaded.goal == "Add new parser"
    assert loaded.steps[1].tests == ["unit parses happy path"]


def test_update_step_status_transitions(tmp_agent_dir):
    p = plan_mod.create_plan("Goal", steps=["a", "b"])
    s = plan_mod.update_step(p, "s1", status="in_progress")
    assert s is not None
    assert s.started_at is not None
    plan_mod.update_step(p, "s1", status="completed")
    reloaded = plan_mod.load_plan(p.id)
    assert reloaded.steps[0].status == "completed"
    assert reloaded.steps[0].completed_at is not None


def test_list_plans_orders_newest_first(tmp_agent_dir):
    p1 = plan_mod.create_plan("first")
    p2 = plan_mod.create_plan("second")
    lst = plan_mod.list_plans()
    ids = [p.id for p in lst]
    assert p2.id in ids and p1.id in ids
    assert ids.index(p2.id) <= ids.index(p1.id)


def test_current_step_picks_in_progress_then_pending(tmp_agent_dir):
    p = plan_mod.create_plan("g", steps=["a", "b", "c"])
    assert p.current_step().id == "s1"
    plan_mod.update_step(p, "s1", status="completed")
    assert p.current_step().id == "s2"
    plan_mod.update_step(p, "s3", status="in_progress")
    assert p.current_step().id == "s3"


def test_record_crash_and_scan(tmp_agent_dir):
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        recovery.record_crash("S1", e, plan_id="P1", last_user_message="do X")
    pending = recovery.scan_pending()
    assert len(pending) == 1
    rec = pending[0]
    assert rec.session_id == "S1"
    assert rec.plan_id == "P1"
    assert "boom" in rec.exception

    recovery.set_status("S1", "ignored")
    assert recovery.scan_pending() == []


def test_handle_pending_auto_skip(tmp_agent_dir):
    try:
        raise ValueError("x")
    except ValueError as e:
        recovery.record_crash("A", e)
    try:
        raise ValueError("y")
    except ValueError as e:
        recovery.record_crash("B", e)
    recovered = recovery.handle_pending_at_startup("auto_skip")
    assert recovered == []
    assert recovery.scan_pending() == []


def test_handle_pending_auto_recover(tmp_agent_dir):
    try:
        raise ValueError("x")
    except ValueError as e:
        recovery.record_crash("A", e)
    recovered = recovery.handle_pending_at_startup("auto_recover")
    assert len(recovered) == 1
    assert recovered[0].session_id == "A"


def test_delete_plan(tmp_agent_dir):
    p = plan_mod.create_plan("g", steps=["a"])
    assert plan_mod.delete_plan(p.id) is True
    assert plan_mod.load_plan(p.id) is None
    assert plan_mod.delete_plan(p.id) is False


# ---------------------------------------------------------------------------
# DAG: ready_steps / blocked_steps
# ---------------------------------------------------------------------------

class TestDAGReadySteps:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        plan_mod.configure(str(tmp_path), ".agent")

    def test_no_deps_all_ready(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c"])
        assert len(p.ready_steps()) == 3

    def test_dep_blocks_step(self):
        p = plan_mod.create_plan("g", steps=[
            {"description": "a", "id": "s1"},
            {"description": "b", "id": "s2"},
        ])
        p.steps[1].deps = ["s1"]
        plan_mod.save_plan(p)
        ready = p.ready_steps()
        assert len(ready) == 1
        assert ready[0].id == "s1"
        blocked = p.blocked_steps()
        assert len(blocked) == 1
        assert blocked[0].id == "s2"

    def test_dep_satisfied_unblocks(self):
        p = plan_mod.create_plan("g", steps=[
            {"description": "a", "id": "s1"},
            {"description": "b", "id": "s2"},
        ])
        p.steps[1].deps = ["s1"]
        plan_mod.update_step(p, "s1", status="completed")
        assert len(p.ready_steps()) == 1
        assert p.ready_steps()[0].id == "s2"
        assert p.blocked_steps() == []

    def test_in_progress_not_in_ready(self):
        p = plan_mod.create_plan("g", steps=["a"])
        plan_mod.update_step(p, "s1", status="in_progress")
        assert p.ready_steps() == []


# ---------------------------------------------------------------------------
# DAG: detect_cycles
# ---------------------------------------------------------------------------

class TestDAGCycles:
    def _make_steps(self, deps_map: dict) -> list:
        steps = [plan_mod.Step(id=sid, description=sid) for sid in deps_map]
        for s in steps:
            s.deps = deps_map[s.id]
        return steps

    def test_no_cycle(self):
        steps = self._make_steps({"a": [], "b": ["a"], "c": ["b"]})
        assert dag_mod.detect_cycles(steps) == []

    def test_simple_cycle(self):
        steps = self._make_steps({"a": ["b"], "b": ["a"]})
        cycles = dag_mod.detect_cycles(steps)
        assert "a" in cycles and "b" in cycles

    def test_self_loop(self):
        steps = self._make_steps({"a": ["a"]})
        assert "a" in dag_mod.detect_cycles(steps)

    def test_no_false_positive_chain(self):
        steps = self._make_steps({"a": [], "b": ["a"], "c": ["a", "b"]})
        assert dag_mod.detect_cycles(steps) == []


# ---------------------------------------------------------------------------
# DAG: critical_path
# ---------------------------------------------------------------------------

class TestDAGCriticalPath:
    def _make_steps(self, deps_map: dict) -> list:
        steps = [plan_mod.Step(id=sid, description=sid) for sid in deps_map]
        for s in steps:
            s.deps = deps_map[s.id]
        return steps

    def test_linear_chain(self):
        steps = self._make_steps({"a": [], "b": ["a"], "c": ["b"]})
        cp = dag_mod.critical_path(steps)
        assert cp == ["a", "b", "c"]

    def test_no_deps_single_step(self):
        steps = self._make_steps({"a": []})
        assert dag_mod.critical_path(steps) == ["a"]

    def test_longest_branch_wins(self):
        # a→b→d and a→c→d, both same length; either is valid
        steps = self._make_steps({"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]})
        cp = dag_mod.critical_path(steps)
        assert cp[0] == "a" and cp[-1] == "d"
        assert len(cp) == 3


# ---------------------------------------------------------------------------
# Backward compat: old JSON without new Step fields loads with defaults
# ---------------------------------------------------------------------------

def test_old_json_without_dag_fields_loads(tmp_agent_dir):
    p = plan_mod.create_plan("g", steps=["a"])
    path = (tmp_agent_dir / ".agent" / "plans" / f"{p.id}.json")
    data = json.loads(path.read_text())
    for step_data in data["steps"]:
        step_data.pop("deps", None)
        step_data.pop("assigned_to", None)
        step_data.pop("agent_constraints", None)
    path.write_text(json.dumps(data))
    loaded = plan_mod.load_plan(p.id)
    assert loaded.steps[0].deps == []
    assert loaded.steps[0].assigned_to == ""
    assert loaded.steps[0].agent_constraints == {}


# ---------------------------------------------------------------------------
# assign_to and agent_constraints persist
# ---------------------------------------------------------------------------

def test_assigned_to_persists(tmp_agent_dir):
    p = plan_mod.create_plan("g", steps=["a"])
    plan_mod.update_step(p, "s1", assigned_to="worker-1", agent_constraints={"llm_tags": ["local"]})
    loaded = plan_mod.load_plan(p.id)
    assert loaded.steps[0].assigned_to == "worker-1"
    assert loaded.steps[0].agent_constraints == {"llm_tags": ["local"]}


# ---------------------------------------------------------------------------
# compact_plan_sync
# ---------------------------------------------------------------------------

class TestCompactPlanSync:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        plan_mod.configure(str(tmp_path), ".agent")

    def test_below_threshold_no_op(self):
        p = plan_mod.create_plan("g", steps=["a", "b"])
        plan_mod.update_step(p, "s1", status="completed")
        ok, msg = compact_plan_sync(p, min_completed=3)
        assert not ok
        assert "threshold" in msg
        assert len(p.steps) == 2

    def test_removes_completed_steps(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c", "d"])
        for sid in ("s1", "s2", "s3"):
            plan_mod.update_step(p, sid, status="completed")
        ok, msg = compact_plan_sync(p, min_completed=3)
        assert ok
        assert len(p.steps) == 1
        assert p.steps[0].id == "s4"

    def test_notes_appended(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c"])
        for sid in ("s1", "s2", "s3"):
            plan_mod.update_step(p, sid, status="completed")
        compact_plan_sync(p, min_completed=3)
        assert "Completed steps" in p.notes

    def test_existing_notes_preserved(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c"])
        p.notes = "prior note"
        plan_mod.save_plan(p)
        for sid in ("s1", "s2", "s3"):
            plan_mod.update_step(p, sid, status="completed")
        compact_plan_sync(p, min_completed=3)
        assert "prior note" in p.notes
        assert "Completed steps" in p.notes

    def test_persists_to_disk(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c"])
        for sid in ("s1", "s2", "s3"):
            plan_mod.update_step(p, sid, status="completed")
        compact_plan_sync(p, min_completed=3)
        loaded = plan_mod.load_plan(p.id)
        assert len(loaded.steps) == 0
        assert "Completed steps" in loaded.notes

    def test_skipped_steps_also_compacted(self):
        p = plan_mod.create_plan("g", steps=["a", "b", "c"])
        plan_mod.update_step(p, "s1", status="completed")
        plan_mod.update_step(p, "s2", status="skipped")
        plan_mod.update_step(p, "s3", status="completed")
        ok, _ = compact_plan_sync(p, min_completed=3)
        assert ok
        assert len(p.steps) == 0
