"""Slash-command registry and handlers for the terminal UI."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# (primary_name, aliases, short_description, takes_arg)
_SLASH_COMMANDS: list[tuple[str, list[str], str, bool]] = [
    ("/a", [], "switch to A (agent answers) tab", False),
    (
        "/analyze-asm",
        ["/asm"],
        "analyze assembly file  --resume --force --levels N",
        True,
    ),
    ("/apply", [], "write last code block to file", False),
    ("/clear", [], "clear the chat screen", False),
    ("/compact", [], "summarize old messages to free context", False),
    ("/context", ["/ctx", "/legend"], "context breakdown grid + color/marker key", False),
    ("/output", ["/out"], "show model output breakdown (think/tool/reply/other)", True),
    ("/continue", ["/c"], "resume after iteration cap or truncation", False),
    ("/exec", [], "run a shell command", True),
    ("/export", [], "export conversation as markdown", False),
    ("/help", ["/?"], "show this help", False),
    ("/load", [], "load a saved session", True),
    ("/q", [], "switch to Q (user questions) tab", False),
    ("/reset", [], "drop conversation history", False),
    ("/save", [], "save session under a name", False),
    ("/sessions", [], "list saved sessions", False),
    ("/sparse", [], "switch to sparse (condensed dialogue) tab", False),
    (
        "/temperature",
        ["/temp"],
        "set sampling temperature (0.0–2.0, - or default to reset)",
        True,
    ),
    ("/think", ["/effort"], "set thinking level  off|low|normal|high|max", True),
    ("/max_tokens", [], "set max tokens   [out <n> | in <n> | <n> | default]", True),
    ("/wrap", [], "toggle line wrapping", False),
    ("/round-summary", ["/summary"], "toggle gray Q/A summary after each turn", False),
    ("/tools", [], "list available tools", False),
    ("/undo", [], "restore last file snapshot", False),
    ("/plan", [], "plan: new <goal> | show | steps | step <id> <status> | abort | pause | stash | resume", True),
    ("/plans", [], "list saved plans", False),
    ("/abort-plan", [], "mark active plan aborted (no stash)", False),
    ("/stash-plan", [], "git stash current changes + mark plan stashed", False),
    ("/pause-plan", [], "mark active plan paused; resume later", False),
    ("/recoveries", [], "list pending crash-recovery records", False),
    ("/quit", ["/exit", "/q!"], "quit the agent", False),
]


def _apply_think(agent, arg: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    from agent.agent import THINK_LEVELS

    v = arg.strip().lower()
    if not v:
        cur = agent.config.llm.think_level
        return (
            True,
            f"think_level = {cur}  (valid: {', '.join(THINK_LEVELS)}; use '-' or 'default' to reset)",
        )
    if v in ("-", "default"):
        agent.config.llm.think_level = agent._llm_defaults["think_level"]
        return True, f"think_level reset to {agent.config.llm.think_level}"
    if v not in THINK_LEVELS:
        return False, f"Invalid level '{v}'. Allowed: {', '.join(THINK_LEVELS)}"
    agent.config.llm.think_level = v
    return True, f"think_level = {v}"


def _apply_temperature(agent, arg: str) -> tuple[bool, str]:
    v = arg.strip().lower()
    if not v:
        return True, (
            f"temperature = {agent.config.llm.temperature}  "
            f"(float 0.0–2.0; '-' or 'default' to reset to {agent._llm_defaults['temperature']})"
        )
    if v in ("-", "default"):
        agent.config.llm.temperature = agent._llm_defaults["temperature"]
        return True, f"temperature reset to {agent.config.llm.temperature}"
    try:
        f = float(v)
    except ValueError:
        return False, f"Invalid number '{v}'. Usage: /temperature <0.0–2.0>"
    if not (0.0 <= f <= 2.0):
        return False, f"Out of range: {f}. Must be 0.0–2.0."
    agent.config.llm.temperature = f
    return True, f"temperature = {f}"


def _apply_max_tokens(agent, arg: str) -> tuple[bool, str]:
    parts = arg.strip().split()
    if not parts:
        return True, (
            f"max output tokens = {agent.config.llm.max_output_tokens}  "
            f"(default {agent._llm_defaults['max_output_tokens']})\n"
            f"input ctx_window    = {agent.config.llm.ctx_window}  "
            f"(default {agent._llm_defaults['ctx_window']})\n"
            f"Usage: /max_tokens <n>           set output tokens\n"
            f"       /max_tokens out <n>       set output tokens\n"
            f"       /max_tokens in <n>        set input ctx_window\n"
            f"       /max_tokens default       reset both"
        )
    head = parts[0].lower()
    if head in ("-", "default"):
        agent.config.llm.max_output_tokens = agent._llm_defaults["max_output_tokens"]
        agent.config.llm.ctx_window = agent._llm_defaults["ctx_window"]
        return True, (
            f"reset: out={agent.config.llm.max_output_tokens} "
            f"in={agent.config.llm.ctx_window}"
        )
    target = "out"
    num_str = head
    if head in ("in", "out"):
        if len(parts) < 2:
            return False, f"Usage: /max_tokens {head} <n>"
        target = head
        num_str = parts[1]
    if num_str in ("-", "default"):
        key = "max_output_tokens" if target == "out" else "ctx_window"
        attr = "max_output_tokens" if target == "out" else "ctx_window"
        setattr(agent.config.llm, attr, agent._llm_defaults[key])
        return True, f"{target} reset to {getattr(agent.config.llm, attr)}"
    try:
        n = int(num_str)
    except ValueError:
        return False, f"Invalid number '{num_str}'. Expected an integer."
    if n <= 0:
        return False, f"Must be positive, got {n}."
    if target == "out":
        agent.config.llm.max_output_tokens = n
        return True, f"max output tokens = {n}"
    agent.config.llm.ctx_window = n
    return True, f"input ctx_window = {n}"


def _active_plan(agent):
    from agent.planning import list_plans
    sid = getattr(getattr(agent, "session", None), "id", "") or ""
    for p in list_plans():
        if p.status in ("active", "pending") and (not sid or p.session_id == sid):
            return p
    # Fallback: first active plan regardless of session
    for p in list_plans():
        if p.status == "active":
            return p
    return None


def _render_plan(plan) -> str:
    lines = [
        f"[bold]plan[/bold] {plan.id}  [dim]status={plan.status}[/dim]",
        f"  goal: {plan.goal}",
    ]
    done, total = plan.progress()
    lines.append(f"  progress: {done}/{total} steps")
    for s in plan.steps:
        marker = {
            "pending": "·", "in_progress": "▶", "completed": "✓",
            "failed": "✗", "skipped": "—",
        }.get(s.status, "?")
        lines.append(f"   {marker} [{s.id}] {s.description}")
        for t_desc in s.tests:
            lines.append(f"        · test: {t_desc}")
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
        sid = getattr(getattr(agent, "session", None), "id", "") or ""
        plan = create_plan(goal=rest.strip(), session_id=sid)
        plan.status = "active"
        save_plan(plan)
        return True, f"Created plan {plan.id}. Add steps via /plan step add <description>."

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
        # /plan step <id> <status> [note]
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
        # Auto-finalize
        done = all(s.status in ("completed", "skipped") for s in plan.steps)
        if done and plan.steps:
            plan.status = "completed"
            save_plan(plan)
        return True, f"step {step_id} → {status}"

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

    if sub == "resume":
        target = rest.strip()
        if target:
            plan = load_plan(target)
        else:
            # pick latest paused/stashed
            plan = None
            for p in list_plans():
                if p.status in ("paused", "stashed"):
                    plan = p
                    break
        if plan is None:
            return False, "No plan to resume."
        plan.status = "active"
        save_plan(plan)
        return True, f"Plan {plan.id} resumed."

    return False, f"Unknown /plan subcommand '{sub}'."


def _match_commands(prefix: str) -> list[tuple[str, str, bool]]:
    """Return (primary_name, description, takes_arg) for commands whose primary
    name or any alias starts with *prefix* (case-insensitive)."""
    pl = prefix.lower()
    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if primary.startswith(pl) or any(a.startswith(pl) for a in aliases):
            out.append((primary, desc, takes_arg))
    return out
