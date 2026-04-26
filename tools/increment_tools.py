"""LLM-callable tools for incremental plan step execution with git snapshots."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools import register

if TYPE_CHECKING:
    from agent.config import Config

_config: "Config | None" = None


def setup(config: "Config") -> None:
    global _config
    _config = config


def _working_dir() -> str:
    return _config.tools.working_dir if _config else "."


def _max_retries(step_max: int) -> int:
    cfg_max = _config.planning.max_step_retries if _config else 3
    return step_max if step_max > 0 else cfg_max


@register(
    "snapshot_step",
    {
        "description": (
            "Create a git snapshot of all repos under the working directory before "
            "starting work on a plan step. Call this before making any file changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID to snapshot"},
            },
            "required": ["plan_id", "step_id"],
        },
    },
)
def tool_snapshot_step(plan_id: str, step_id: str) -> dict:
    from agent.planning import load_plan, save_plan
    from agent.planning.increment import snapshot_step
    from agent.planning.plan import update_step

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    snapshots = snapshot_step(plan_id, step_id, _working_dir())
    refs = [s.to_dict() for s in snapshots]
    update_step(plan, step_id, snapshot_refs=refs, status="in_progress")

    if not snapshots:
        return {
            "ok": True,
            "repos_snapshotted": 0,
            "repos": [],
            "message": "No git repos found — proceeding without snapshot.",
        }

    dirty = sum(1 for s in snapshots if s.was_dirty)
    return {
        "ok": True,
        "repos_snapshotted": len(snapshots),
        "dirty_committed": dirty,
        "repos": refs,
        "message": f"Snapshotted {len(snapshots)} repo(s), {dirty} had uncommitted changes.",
    }


@register(
    "complete_step",
    {
        "description": (
            "Mark a plan step as completed after successful verification. "
            "Optionally squashes the snapshot commit into a clean commit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID to complete"},
                "notes": {"type": "string", "description": "Optional completion notes"},
            },
            "required": ["plan_id", "step_id"],
        },
    },
)
def tool_complete_step(plan_id: str, step_id: str, notes: str = "") -> dict:
    from agent.planning import load_plan
    from agent.planning.increment import squash_snapshot, RepoSnapshot
    from agent.planning.plan import update_step

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    squash_results: list[dict] = []
    if _config and _config.planning.squash_snap_on_success:
        squash_msg = f"feat: {step.description}"
        for ref in step.snapshot_refs:
            snap = RepoSnapshot.from_dict(ref)
            if snap.was_dirty:
                ok, msg = squash_snapshot(snap.repo, squash_msg)
                squash_results.append({"repo": snap.repo, "ok": ok, "message": msg})

    kw: dict = {"status": "completed"}
    if notes:
        kw["notes"] = notes
    update_step(plan, step_id, **kw)

    return {"ok": True, "message": f"Step {step_id} completed.", "squashed": squash_results}


@register(
    "revert_step",
    {
        "description": (
            "Revert all git repos to the snapshot taken before this step, then "
            "reset the step to pending for a retry. Call when verification fails. "
            "Returns exhausted=true when max retries reached (step marked failed)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID to revert"},
            },
            "required": ["plan_id", "step_id"],
        },
    },
)
def tool_revert_step(plan_id: str, step_id: str) -> dict:
    from agent.planning import load_plan
    from agent.planning.increment import revert_to_snapshots, RepoSnapshot
    from agent.planning.plan import update_step

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    if not step.snapshot_refs:
        return {"ok": False, "exhausted": False, "message": "No snapshot to revert to."}

    new_retry_count = step.retry_count + 1
    max_ret = _max_retries(step.max_retries)

    snapshots = [RepoSnapshot.from_dict(r) for r in step.snapshot_refs]
    revert_results = revert_to_snapshots(snapshots)

    if new_retry_count >= max_ret:
        update_step(plan, step_id, status="failed", retry_count=new_retry_count)
        return {
            "ok": False,
            "exhausted": True,
            "retry_count": new_retry_count,
            "max_retries": max_ret,
            "reverted": [{"repo": r, "ok": ok, "message": msg} for r, ok, msg in revert_results],
            "message": f"Max retries ({max_ret}) reached. Step marked failed.",
        }

    update_step(plan, step_id, status="pending", retry_count=new_retry_count, snapshot_refs=[])
    return {
        "ok": True,
        "exhausted": False,
        "retry_count": new_retry_count,
        "max_retries": max_ret,
        "reverted": [{"repo": r, "ok": ok, "message": msg} for r, ok, msg in revert_results],
        "message": f"Reverted. Retry {new_retry_count}/{max_ret}.",
    }
