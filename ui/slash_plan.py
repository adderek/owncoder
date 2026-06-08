"""Slash command handler for /plan and related planning commands."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _active_plan(agent):
    from agent.planning import list_plans
    sid = getattr(getattr(agent, "session", None), "id", "") or ""
    for p in list_plans():
        if p.status in ("active", "pending") and (not sid or p.session_id == sid):
            return p
    for p in list_plans():
        if p.status == "active":
            return p
    return None


def _render_plan(plan) -> str:
    lines = [
        f"[bold]plan[/bold] {plan.id}  [dim]status={plan.status}[/dim]",
        f"  goal: {plan.goal}",
    ]
    if plan.context:
        preview = plan.context[:80].replace("\n", " ")
        lines.append(f"  context: {preview}{'…' if len(plan.context) > 80 else ''}")
    done, total = plan.progress()
    lines.append(f"  progress: {done}/{total} steps")
    ready_ids = {s.id for s in plan.ready_steps()}
    for s in plan.steps:
        marker = {
            "pending": "·", "in_progress": "▶", "completed": "✓",
            "failed": "✗", "skipped": "—", "blocked": "⚠",
        }.get(s.status, "?")
        suffix = ""
        if s.deps:
            suffix += f"  [dim]needs: {', '.join(s.deps)}[/dim]"
        if s.assigned_to:
            suffix += f"  [dim]@{s.assigned_to}[/dim]"
        if s.status == "pending" and s.id in ready_ids:
            suffix += "  [green]ready[/green]"
        if s.status == "blocked":
            suffix += "  [red]BLOCKED[/red]"
        lines.append(f"   {marker} [{s.id}] {s.description}{suffix}")
        for c in s.acceptance_criteria:
            lines.append(f"        ✓ accept: {c}")
        for t_desc in s.tests:
            lines.append(f"        · test: {t_desc}")
    cp = plan.critical_path()
    if len(cp) > 1:
        lines.append(f"  critical path: {' → '.join(cp)}")
    if plan.final_tests:
        lines.append(f"  final tests: {len(plan.final_tests)} check(s) pending after all steps")
    return "\n".join(lines)


def _apply_plan(agent, arg: str) -> tuple[bool, str]:
    """Handle /plan <sub> …. Returns (ok, message)."""
    from agent.planning import (
        create_plan, load_plan, save_plan, list_plans,
    )
    from agent.planning.plan import update_step

    parts = arg.strip().split(maxsplit=1)
    if not parts:
        plan = _active_plan(agent)
        if plan is None:
            return True, "No active plan. Usage: /plan new <goal>"
        return True, _render_plan(plan)

    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if sub in ("new", "create"):
        if not rest.strip():
            return False, "Usage: /plan new <goal>"
        from agent.planning.plan import _switch_to_plan
        sid = getattr(getattr(agent, "session", None), "id", "") or ""
        plan = create_plan(goal=rest.strip(), session_id=sid, created_by="user")
        paused = _switch_to_plan(plan)
        suffix = f" (paused plan {paused.id})" if paused else ""
        return True, f"Created plan {plan.id}.{suffix} Add steps via /plan step add <description>."

    if sub == "show":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        return True, _render_plan(plan)

    if sub == "steps":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        return True, _render_plan(plan)

    if sub == "step":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        sp = rest.split(maxsplit=2)
        if not sp:
            return False, "Usage: /plan step add <desc> | <id> <status> [note]"
        action = sp[0]
        if action == "add":
            desc = sp[1] if len(sp) > 1 else ""
            if len(sp) > 2:
                desc = f"{sp[1]} {sp[2]}"
            if not desc:
                return False, "Usage: /plan step add <desc>"
            from agent.planning.plan import Step
            sid = f"s{len(plan.steps) + 1}"
            plan.steps.append(Step(id=sid, description=desc))
            save_plan(plan)
            return True, f"Added step {sid}: {desc}"
        step_id = sp[0]
        if len(sp) < 2:
            return False, "Usage: /plan step <id> <status>"
        status = sp[1]
        note = sp[2] if len(sp) > 2 else ""
        if status not in ("pending", "in_progress", "completed", "failed", "skipped"):
            return False, f"Bad status '{status}'."
        fields: dict = {"status": status}
        if note:
            fields["notes"] = note
        updated = update_step(plan, step_id, **fields)
        if updated is None:
            return False, f"No step {step_id}."
        done = all(s.status in ("completed", "skipped") for s in plan.steps)
        if done and plan.steps:
            plan.status = "completed"
            save_plan(plan)
        return True, f"step {step_id} → {status}"

    if sub == "dep":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        dp = rest.split()
        if len(dp) < 2:
            return False, "Usage: /plan dep <step_id> <dep_step_id>"
        step_id, dep_step_id = dp[0], dp[1]
        step = next((s for s in plan.steps if s.id == step_id), None)
        if step is None:
            return False, f"No step {step_id}."
        if not any(s.id == dep_step_id for s in plan.steps):
            return False, f"No step {dep_step_id}."
        if dep_step_id in step.deps:
            return True, "Dep already exists."
        step.deps.append(dep_step_id)
        from agent.planning.dag import detect_cycles
        cycles = detect_cycles(plan.steps)
        if cycles:
            step.deps.remove(dep_step_id)
            return False, f"Would create cycle: {', '.join(cycles)}"
        save_plan(plan)
        return True, f"{step_id} now depends on {dep_step_id}"

    if sub == "assign":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        ap = rest.split(maxsplit=1)
        if len(ap) < 2:
            return False, "Usage: /plan assign <step_id> <agent_id>"
        step_id, agent_id = ap[0], ap[1]
        updated = update_step(plan, step_id, assigned_to=agent_id)
        if updated is None:
            return False, f"No step {step_id}."
        return True, f"{step_id} assigned to {agent_id}"

    if sub == "compact":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        from agent.planning.compact import compact_plan_sync
        min_c = int(rest.strip()) if rest.strip().isdigit() else 3
        ok, msg = compact_plan_sync(plan, min_completed=min_c)
        return ok, msg

    if sub == "abort":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        plan.status = "aborted"
        save_plan(plan)
        return True, f"Plan {plan.id} aborted."

    if sub == "pause":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        plan.status = "paused"
        save_plan(plan)
        return True, f"Plan {plan.id} paused."

    if sub == "stash":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        import subprocess
        try:
            r = subprocess.run(
                ["git", "stash", "push", "-u", "-m", f"plan:{plan.id}"],
                capture_output=True, text=True, timeout=15,
                cwd=agent.config.tools.working_dir,
            )
            stash_out = r.stdout.strip() or r.stderr.strip()
        except Exception as e:
            stash_out = f"git stash failed: {e}"
        plan.status = "stashed"
        save_plan(plan)
        return True, f"Plan {plan.id} stashed.\n{stash_out}"

    if sub == "switch":
        target = rest.strip()
        if not target:
            return False, "Usage: /plan switch <plan_id>"
        from agent.planning.plan import _switch_to_plan
        plan = load_plan(target)
        if plan is None:
            return False, f"Plan not found: {target}"
        paused = _switch_to_plan(plan)
        suffix = f" (paused plan {paused.id})" if paused else ""
        return True, f"Switched to plan {plan.id}.{suffix}"

    if sub == "resume":
        target = rest.strip()
        if target:
            plan = load_plan(target)
        else:
            # prefer the resume_to target of the current active plan
            plan = None
            active = _active_plan(agent)
            if active and active.resume_to:
                plan = load_plan(active.resume_to)
            if plan is None:
                for p in list_plans():
                    if p.status in ("paused", "stashed"):
                        plan = p
                        break
        if plan is None:
            return False, "No plan to resume."
        from agent.planning.plan import _switch_to_plan
        _switch_to_plan(plan)
        return True, f"Plan {plan.id} resumed."

    if sub == "context":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        if not rest.strip():
            if plan.context:
                return True, f"Plan context:\n{plan.context}"
            return True, "No shared context set. Usage: /plan context <text>"
        plan.context = rest.strip()
        save_plan(plan)
        return True, "Shared context updated."

    if sub == "final":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        fp = rest.strip().split(maxsplit=1)
        if not fp or fp[0] == "show":
            if not plan.final_tests:
                return True, "No final tests defined. Add with: /plan final add <check>"
            lines = ["Final acceptance tests:"]
            for i, t in enumerate(plan.final_tests, 1):
                lines.append(f"  {i}. {t}")
            return True, "\n".join(lines)
        if fp[0] == "add":
            check = fp[1] if len(fp) > 1 else ""
            if not check:
                return False, "Usage: /plan final add <check description>"
            plan.final_tests.append(check)
            save_plan(plan)
            return True, f"Added final test: {check}"
        if fp[0] == "clear":
            plan.final_tests = []
            save_plan(plan)
            return True, "Final tests cleared."
        return False, "Usage: /plan final [show|add <check>|clear]"

    if sub == "escalate":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        ep = rest.split(maxsplit=1)
        if len(ep) < 2:
            return False, "Usage: /plan escalate <step_id> <reason>"
        step_id, reason = ep[0], ep[1]
        from agent.planning.plan import update_step
        updated = update_step(plan, step_id, status="blocked", notes=f"Blocked: {reason[:200]}")
        if updated is None:
            return False, f"No step {step_id}."
        import time as _time
        ts = _time.strftime("%Y-%m-%d %H:%M:%S UTC", _time.gmtime())
        step = updated
        doc_lines = [
            f"# Blocking Issue: {step.description}",
            "",
            f"**Created:** {ts}  ",
            f"**Plan ID:** `{plan.id}`  ",
            f"**Step ID:** `{step_id}`",
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
        doc_lines += ["", "## Blocking Issue", reason]
        doc_lines += [
            "",
            "---",
            "See `docs/agent/escalation.md` for escalation guide.",
        ]
        doc = "\n".join(doc_lines)
        from agent.planning.plan import _get_plans_dir
        d = _get_plans_dir()
        d.mkdir(parents=True, exist_ok=True)
        safe_step = step_id.replace("/", "_")
        doc_path = d / f"{plan.id}_{safe_step}_escalation.md"
        doc_path.write_text(doc, encoding="utf-8")
        return True, f"Step {step_id} escalated. Escalation doc: {doc_path}"

    if sub == "brief":
        plan = _active_plan(agent)
        if plan is None:
            return False, "No active plan."
        if not rest.strip():
            return False, "Usage: /plan brief <step_id>"
        step_id = rest.strip().split()[0]
        step = next((s for s in plan.steps if s.id == step_id), None)
        if step is None:
            return False, f"No step {step_id}."
        done, total = plan.progress()
        lines = [
            f"[bold]Step {step.id}[/bold] — {step.description}",
            f"Plan: {plan.goal}  ({done}/{total} done)",
        ]
        if plan.context:
            lines += ["", "[bold]Shared context:[/bold]", plan.context]
        if step.introduction:
            lines += ["", "[bold]Introduction:[/bold]", step.introduction]
        if step.acceptance_criteria:
            lines += ["", "[bold]Acceptance criteria:[/bold]"]
            lines += [f"  ✓ {c}" for c in step.acceptance_criteria]
        if step.tests:
            lines += ["", "[bold]Verification:[/bold]"]
            lines += [f"  · {t}" for t in step.tests]
        return True, "\n".join(lines)

    return False, f"Unknown /plan subcommand '{sub}'."
