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

    done, total = plan.progress()
    if done == total and total > 0:
        plan.status = "completed"
        from agent.planning import save_plan
        save_plan(plan)
        resume_msg = ""
        if plan.resume_to:
            prev = load_plan(plan.resume_to)
            if prev is not None:
                resume_msg = (
                    f" Plan complete. Previous plan '{prev.goal}' ({prev.id}) is paused"
                    f" — call set_current_plan('{prev.id}') to return to it."
                )
        return {
            "ok": True,
            "plan_completed": True,
            "message": f"Step {step_id} completed. All {total} steps done.{resume_msg}",
            "squashed": squash_results,
        }

    return {"ok": True, "message": f"Step {step_id} completed. ({done}/{total} steps done)", "squashed": squash_results}


@register(
    "plan_ready_steps",
    {
        "description": (
            "Return all pending steps that have no unresolved dependencies. "
            "Use this to decide which steps can be worked on in parallel or next."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
            },
            "required": ["plan_id"],
        },
    },
)
def tool_plan_ready_steps(plan_id: str) -> dict:
    from agent.planning import load_plan

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    ready = plan.ready_steps()
    blocked = plan.blocked_steps()
    return {
        "ok": True,
        "ready": [{"id": s.id, "description": s.description, "assigned_to": s.assigned_to} for s in ready],
        "blocked": [{"id": s.id, "description": s.description, "deps": s.deps} for s in blocked],
        "ready_count": len(ready),
        "blocked_count": len(blocked),
    }


@register(
    "plan_add_dep",
    {
        "description": (
            "Add a dependency between two steps in a plan. "
            "step_id will not start until dep_step_id is completed/skipped."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step that depends on dep_step_id"},
                "dep_step_id": {"type": "string", "description": "Step that must complete first"},
            },
            "required": ["plan_id", "step_id", "dep_step_id"],
        },
    },
)
def tool_plan_add_dep(plan_id: str, step_id: str, dep_step_id: str) -> dict:
    from agent.planning import load_plan, save_plan
    from agent.planning.dag import detect_cycles

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}
    if not any(s.id == dep_step_id for s in plan.steps):
        return {"ok": False, "error": f"dep step not found: {dep_step_id}"}
    if dep_step_id in step.deps:
        return {"ok": True, "message": "dep already exists"}

    step.deps.append(dep_step_id)
    cycles = detect_cycles(plan.steps)
    if cycles:
        step.deps.remove(dep_step_id)
        return {"ok": False, "error": f"would create cycle involving: {cycles}"}

    save_plan(plan)
    return {"ok": True, "message": f"{step_id} now depends on {dep_step_id}"}


@register(
    "plan_assign_step",
    {
        "description": (
            "Assign a step to a specific agent and optionally record LLM/env constraints. "
            "Used for multi-agent coordination."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID to assign"},
                "agent_id": {"type": "string", "description": "Agent identifier"},
                "constraints": {
                    "type": "object",
                    "description": "Routing hints e.g. {\"llm_tags\": [\"local\"], \"env\": \"gpu-box\"}",
                },
            },
            "required": ["plan_id", "step_id", "agent_id"],
        },
    },
)
def tool_plan_assign_step(
    plan_id: str, step_id: str, agent_id: str, constraints: dict | None = None
) -> dict:
    from agent.planning import load_plan
    from agent.planning.plan import update_step

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    fields: dict = {"assigned_to": agent_id}
    if constraints:
        fields["agent_constraints"] = constraints
    update_step(plan, step_id, **fields)
    return {"ok": True, "message": f"{step_id} assigned to {agent_id}"}


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


@register(
    "get_step_brief",
    {
        "description": (
            "Return a focused context bundle for a single plan step: plan goal, shared context, "
            "step introduction, description, acceptance criteria, and tests. "
            "Designed to be passed as `context` to spawn_agents for isolated step execution, "
            "or used to refocus after context drift."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID"},
            },
            "required": ["plan_id", "step_id"],
        },
    },
)
def tool_get_step_brief(plan_id: str, step_id: str) -> dict:
    from agent.planning import load_plan

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    done, total = plan.progress()
    brief_lines = [
        f"# Step Brief: {step.id} — {step.description}",
        "",
        f"## Plan: {plan.goal}",
        f"Progress: {done}/{total} steps complete",
    ]
    if plan.context:
        brief_lines += ["", "## Shared Context", plan.context]
    if step.introduction:
        brief_lines += ["", "## Introduction", step.introduction]
    brief_lines += ["", "## Task", step.description]
    if step.acceptance_criteria:
        brief_lines += ["", "## Acceptance Criteria"]
        brief_lines += [f"- {c}" for c in step.acceptance_criteria]
    if step.tests:
        brief_lines += ["", "## Verification Steps"]
        brief_lines += [f"- {t}" for t in step.tests]
    if plan.final_tests and all(
        s.status in ("completed", "skipped") for s in plan.steps if s.id != step_id
    ):
        brief_lines += ["", "## Final Acceptance Tests (run after all steps complete)"]
        brief_lines += [f"- {t}" for t in plan.final_tests]

    brief = "\n".join(brief_lines)
    return {
        "ok": True,
        "plan_id": plan_id,
        "step_id": step_id,
        "brief": brief,
    }


@register(
    "report_blocking_issue",
    {
        "description": (
            "Report a blocking issue that prevents step completion. "
            "Marks the step as 'blocked', writes an escalation document with full context "
            "(plan goal, shared context, step brief, what was tried, the blocker) "
            "to .agent/plans/{plan_id}_{step_id}_escalation.md. "
            "Call when revert_step exhausted=true OR when stuck on something outside your capability."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "Plan ID"},
                "step_id": {"type": "string", "description": "Step ID"},
                "issue": {"type": "string", "description": "Clear description of the blocking issue"},
                "what_was_tried": {
                    "type": "string",
                    "description": "What approaches were attempted before escalating",
                },
                "suggested_resolution": {
                    "type": "string",
                    "description": "Optional: what you think might resolve this",
                },
            },
            "required": ["plan_id", "step_id", "issue"],
        },
    },
)
def tool_report_blocking_issue(
    plan_id: str,
    step_id: str,
    issue: str,
    what_was_tried: str = "",
    suggested_resolution: str = "",
) -> dict:
    import time as _time
    from agent.planning import load_plan
    from agent.planning.plan import update_step

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    step = next((s for s in plan.steps if s.id == step_id), None)
    if step is None:
        return {"ok": False, "error": f"step not found: {step_id}"}

    ts = _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime())

    doc_lines = [
        f"# Blocking Issue: {step.description}",
        "",
        f"**Created:** {ts}  ",
        f"**Plan ID:** `{plan_id}`  ",
        f"**Step ID:** `{step_id}`",
        "",
        "---",
        "",
        "## Plan Goal",
        plan.goal,
    ]
    if plan.context:
        doc_lines += ["", "## Shared Context", plan.context]
    if step.introduction:
        doc_lines += ["", "## Step Introduction", step.introduction]
    doc_lines += ["", "## Step Description", step.description]
    if step.acceptance_criteria:
        doc_lines += ["", "## Acceptance Criteria"]
        doc_lines += [f"- {c}" for c in step.acceptance_criteria]
    if step.tests:
        doc_lines += ["", "## Verification Steps"]
        doc_lines += [f"- {t}" for t in step.tests]
    doc_lines += ["", "---", "", "## Blocking Issue", issue]
    if what_was_tried:
        doc_lines += ["", "## What Was Tried", what_was_tried]
    if suggested_resolution:
        doc_lines += ["", "## Suggested Resolution", suggested_resolution]
    doc_lines += [
        "",
        "---",
        "",
        "## How to Escalate",
        "1. Share this document with a stronger model or human reviewer.",
        "2. Provide any relevant file contents or error logs.",
        "3. Ask them to resume with: `load plan and continue step`",
        f"   Plan: `{plan_id}` / Step: `{step_id}`",
        "",
        "See `docs/agent/escalation.md` for full escalation guide.",
    ]

    doc = "\n".join(doc_lines)

    d = _get_plans_dir()
    d.mkdir(parents=True, exist_ok=True)
    safe_step = step_id.replace("/", "_")
    doc_path = d / f"{plan_id}_{safe_step}_escalation.md"
    doc_path.write_text(doc, encoding="utf-8")

    update_step(plan, step_id, status="blocked", notes=f"Blocked: {issue[:200]}")

    return {
        "ok": True,
        "escalation_doc": str(doc_path),
        "doc_content": doc,
        "message": f"Step {step_id} marked blocked. Escalation doc: {doc_path}",
    }


@register(
    "create_plan",
    {
        "description": (
            "Create a structured execution plan with atomic steps. "
            "Automatically becomes the active plan (previous active plan is paused; "
            "after this plan completes you will be prompted to return to it). "
            "Use plan_ready_steps to see which steps can start, "
            "then snapshot_step → implement → complete_step for each."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "High-level goal this plan achieves.",
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered list of atomic steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Short step id e.g. 's1'. Auto-assigned if omitted."},
                            "description": {"type": "string", "description": "What this step does."},
                            "introduction": {"type": "string", "description": "Why this step exists / what it builds on."},
                            "acceptance_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Human-readable done conditions.",
                            },
                            "tests": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Verification commands or checks.",
                            },
                            "deps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Step IDs that must complete before this one.",
                            },
                        },
                        "required": ["description"],
                    },
                },
                "context": {
                    "type": "string",
                    "description": "Shared context injected into every step brief (key files, constraints, arch decisions).",
                },
                "final_tests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Acceptance checks run after ALL steps complete.",
                },
            },
            "required": ["goal", "steps"],
        },
    },
)
def tool_create_plan(
    goal: str,
    steps: list,
    context: str = "",
    final_tests: list | None = None,
) -> dict:
    from agent.planning.plan import create_plan as _create_plan, _switch_to_plan

    plan = _create_plan(
        goal=goal,
        session_id="",
        steps=steps,
        context=context,
        final_tests=final_tests,
        created_by="agent",
    )
    _switch_to_plan(plan)

    cs = plan.current_step()
    step_list = [{"id": s.id, "description": s.description} for s in plan.steps]
    next_hint = (
        f"Start with: get_step_brief(plan_id='{plan.id}', step_id='{cs.id}')"
        f" then snapshot_step(plan_id='{plan.id}', step_id='{cs.id}')"
        if cs else ""
    )
    return {
        "ok": True,
        "plan_id": plan.id,
        "goal": plan.goal,
        "steps_count": len(plan.steps),
        "steps": step_list,
        "first_step": {"id": cs.id, "description": cs.description} if cs else None,
        "message": f"Plan '{plan.id}' created and active. {next_hint}".strip(),
    }


@register(
    "set_current_plan",
    {
        "description": (
            "Switch to a different existing plan, pausing the currently active one. "
            "After the new plan completes, you will be prompted to return to the paused one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string", "description": "ID of the plan to make active."},
            },
            "required": ["plan_id"],
        },
    },
)
def tool_set_current_plan(plan_id: str) -> dict:
    from agent.planning import load_plan
    from agent.planning.plan import _switch_to_plan

    plan = load_plan(plan_id)
    if plan is None:
        return {"ok": False, "error": f"plan not found: {plan_id}"}

    paused = _switch_to_plan(plan)
    cs = plan.current_step()
    done, total = plan.progress()
    return {
        "ok": True,
        "plan_id": plan.id,
        "goal": plan.goal,
        "status": plan.status,
        "progress": f"{done}/{total}",
        "paused_plan": paused.id if paused else None,
        "next_step": {"id": cs.id, "description": cs.description} if cs else None,
        "message": f"Switched to plan '{plan.id}'. {f'Previous plan {paused.id!r} paused.' if paused else ''}".strip(),
    }
