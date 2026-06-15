"""Slash-command registry and handlers for the terminal UI."""
from __future__ import annotations

from typing import TYPE_CHECKING

from agent.ui.slash_plan import _active_plan, _render_plan, _apply_plan

if TYPE_CHECKING:
    from agent.config.models import Config


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
    ("/goal", [], "set/show/clear completion goal  [<text> | $ <cmd> | clear]", True),
    ("/exec", [], "run a shell command", True),
    ("/export", [], "export conversation as markdown", False),
    ("/help", ["/?"], "show this help", False),
    ("/load", [], "load a saved session", True),
    ("/q", [], "switch to Q (user questions) tab", False),
    ("/reset", [], "drop conversation history", False),
    ("/save", [], "save session under a name", True),
    ("/sessions", [], "list saved sessions", False),
    ("/sparse", [], "switch to sparse (condensed dialogue) tab", False),
    (
        "/temperature",
        ["/temp"],
        "set sampling temperature (0.0–2.0, - or default to reset)",
        True,
    ),
    ("/think", ["/effort"], "set thinking level  off|low|normal|high|max", True),
    ("/autonomy", ["/auto", "/verbose"], "set autonomy level  0.0–1.0 (or %) or supervised|explain|balanced|brisk|autopilot", True),
    ("/max_tokens", [], "set max tokens   [out <n> | in <n> | <n> | default]", True),
    ("/maxiter", ["/max_iter"], "set max tool-call iterations per turn  [<n> | 0/none = unlimited]", True),
    ("/wrap", [], "toggle line wrapping", False),
    ("/round-summary", ["/summary"], "toggle gray Q/A summary after each turn", False),
    ("/tools", [], "list available tools", False),
    ("/skills", [], "skills: list | show <name> | history <name> | rm <name>", True),
    ("/undo", [], "restore last file snapshot", False),
    ("/checkpoint", ["/cp"], "checkpoint: list | new [label] | rollback <id>", True),
    ("/mcp", [], "show MCP server status + their tools", False),
    ("/security", ["/sec", "/audit"], "security: scan|diff|selfaudit|report [path] | baseline [accept|clear|show] | airgap [on|off|status]", True),
    ("/plan", [], "plan: new <goal> | show | steps | step <id> <status> | dep <step> <dep> | assign <step> <agent> | compact | abort | pause | stash | resume", True),
    ("/plans", [], "list saved plans", False),
    ("/abort-plan", [], "mark active plan aborted (no stash)", False),
    ("/stash-plan", [], "git stash current changes + mark plan stashed", False),
    ("/pause-plan", [], "mark active plan paused; resume later", False),
    ("/notify", [], "notification channels  [on | off | status]", True),
    ("/model", [], "switch active model  [<entry> | role=<entry> | role=? | refresh]", True),
    ("/models", [], "show all configured model entries with capabilities", False),
    ("/recoveries", [], "list pending crash-recovery records", False),
    ("/resummarize", [], "re-summarize Q/A entries with stale or missing summaries  [--force]", True),
    ("/idea", [], "ideas: <title> | add [--type T] [--tags t] [--priority N] <title> [| body] | show <id> | update <id> k=v | done <id> | reject <id>", True),
    ("/ideas", [], "list ideas  [status_filter: raw|evaluated|planned|implementing|verifying|done|rejected]", True),
    ("/quit", ["/exit", "/q!"], "quit the agent", False),
]


def _apply_think(agent, arg: str) -> tuple[bool, str]:
    """Returns (ok, message)."""
    from agent.core.prompts import THINK_LEVELS

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


def _apply_autonomy(agent, arg: str) -> tuple[bool, str]:
    from agent.core.prompts import AUTONOMY_LEVELS, _resolve_autonomy

    def _fmt(val: float) -> str:
        anchor = min(4, max(0, round(val * 4)))
        return f"{val:.2f} ({int(val * 100)}%) [{AUTONOMY_LEVELS[anchor]}]"

    v = arg.strip().lower()
    if not v:
        cur = getattr(agent.config.agent, "autonomy", 0.5)
        return True, (
            f"autonomy = {_fmt(_resolve_autonomy(cur))}  "
            f"(0.0=supervised … 1.0=autopilot; also accepts % or name: {', '.join(AUTONOMY_LEVELS)}; '-' to reset)"
        )
    if v in ("-", "default"):
        agent.config.agent.autonomy = agent._llm_defaults["autonomy"]
        return True, f"autonomy reset to {_fmt(_resolve_autonomy(agent.config.agent.autonomy))}"
    resolved = _resolve_autonomy(v)
    agent.config.agent.autonomy = resolved
    return True, f"autonomy = {_fmt(resolved)}"


def _apply_notify(agent, arg: str, broker=None) -> tuple[bool, str]:
    """Returns (ok, message)."""
    v = arg.strip().lower()
    cfg = agent.config.notify
    if v in ("", "status"):
        if broker is not None:
            return True, broker.status()
        return True, f"notify {'on' if cfg.enabled else 'off'} — {len(cfg.channels)} channel(s) configured"
    if v in ("on", "off"):
        cfg.enabled = v == "on"
        if cfg.enabled and not cfg.channels:
            return True, "notify on — but no channels configured ([notify] in agent.toml/agent.yaml)"
        return True, f"notify {v}"
    return False, "Usage: /notify [on | off | status]"


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


def _apply_max_iter(agent, arg: str) -> tuple[bool, str]:
    cur = agent.config.llm.max_iterations
    cur_display = "unlimited" if cur is None or cur == 0 else str(cur)
    v = arg.strip().lower()
    if not v:
        return True, (
            f"max_iterations = {cur_display}  "
            f"(0/none = unlimited; set positive int to cap tool rounds per turn)\n"
            f"Usage: /maxiter <n>   or   /maxiter 0   or   /maxiter none"
        )
    if v in ("0", "none", "null", "unlimited", "-", "default", "∞", "inf"):
        agent.config.llm.max_iterations = None
        return True, "max_iterations = unlimited (Ctrl+C to interrupt)"
    try:
        n = int(v)
    except ValueError:
        return False, f"Invalid value '{arg.strip()}'. Use a positive integer or 0/none for unlimited."
    if n < 1:
        agent.config.llm.max_iterations = None
        return True, "max_iterations = unlimited (Ctrl+C to interrupt)"
    agent.config.llm.max_iterations = n
    return True, f"max_iterations = {n}"


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



_ROLE_ALIASES: dict[str, str] = {
    "llm": "default",
    "default": "default",
    "sum": "summarizer",
    "summarizer": "summarizer",
    "emb": "embeddings",
    "embeddings": "embeddings",
}


def _apply_model(agent, arg: str) -> tuple[bool, str]:
    """Handle /model [role=]<entry-name>.  Returns (ok, message)."""
    from openai import AsyncOpenAI

    cfg = agent.config
    entries = cfg.model_entries

    def _status() -> str:
        cur_model = cfg.llm.model
        cur_url = cfg.llm.base_url
        lines = [
            f"active LLM: [bold]{cur_model}[/bold]  ({cur_url})",
            f"  roles: default={cfg.model_roles.get('default', '—')}  "
            f"summarizer={cfg.model_roles.get('summarizer', '—')}",
            "available entries:",
        ]
        for name, e in sorted(entries.items()):
            tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"  [bold]{name}[/bold]  {e.model}  ({e.base_url}){tags}")
        return "\n".join(lines)

    v = arg.strip()
    if not v:
        return True, _status()

    # Parse optional "role=" prefix
    role = "default"
    entry_name = v
    if "=" in v:
        role_raw, _, entry_name = v.partition("=")
        role_raw = role_raw.strip().lower()
        entry_name = entry_name.strip()
        if role_raw not in _ROLE_ALIASES:
            known = ", ".join(sorted(_ROLE_ALIASES))
            return False, f"Unknown role '{role_raw}'. Known: {known}"
        role = _ROLE_ALIASES[role_raw]

    if entry_name == "?":
        return True, _status()

    if entry_name not in entries:
        known = ", ".join(sorted(entries))
        return False, f"No model entry '{entry_name}'. Available: {known}"

    entry = entries[entry_name]
    cfg.model_roles[role] = entry_name

    if role == "default":
        cfg.llm.base_url = entry.base_url
        cfg.llm.api_key = entry.api_key
        if entry.model:
            cfg.llm.model = entry.model
        cfg.llm.ctx_window = entry.ctx_window
        cfg.llm.max_output_tokens = entry.max_output_tokens
        cfg.llm.temperature = entry.temperature
        # Recreate the OpenAI client with new endpoint/key
        agent._client = AsyncOpenAI(
            base_url=entry.base_url,
            api_key=entry.api_key,
        )
        msg = (
            f"switched default → [bold]{entry_name}[/bold]  "
            f"model={cfg.llm.model}  url={cfg.llm.base_url}"
        )
        used = agent.token_estimate()
        threshold = int(entry.ctx_window * cfg.llm.compaction_threshold)
        if used > threshold:
            msg += (
                f"\n[yellow]Warning: current context ({used} tokens) exceeds "
                f"compaction threshold ({threshold}) for new ctx_window={entry.ctx_window}. "
                f"Compaction will trigger on next turn.[/yellow]"
            )
        return True, msg

    # Non-default role: just update model_roles (no client to recreate)
    return True, f"role '{role}' → [bold]{entry_name}[/bold]  (model={entry.model})"


def _render_models_table(config: "Config"):
    """Return a Rich Table showing all model_entries with capabilities."""
    from rich.table import Table

    entries = config.model_entries
    roles_rev: dict[str, list[str]] = {}
    for role, entry_name in config.model_roles.items():
        roles_rev.setdefault(entry_name, []).append(role)

    active_entry = config.model_roles.get("default", "default")

    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False, collapse_padding=True)
    tbl.add_column("name", style="cyan", no_wrap=True)
    tbl.add_column("model id", no_wrap=True)
    tbl.add_column("endpoint", style="dim", no_wrap=True)
    tbl.add_column("ctx", justify="right", no_wrap=True)
    tbl.add_column("out", justify="right", no_wrap=True)
    tbl.add_column("temp", justify="right", no_wrap=True)
    tbl.add_column("params", justify="right", no_wrap=True)
    tbl.add_column("tok/s", justify="right", no_wrap=True)
    tbl.add_column("L", justify="center", no_wrap=True)  # local
    tbl.add_column("T", justify="center", no_wrap=True)  # thinking
    tbl.add_column("$/in", justify="right", no_wrap=True)
    tbl.add_column("$/out", justify="right", no_wrap=True)
    tbl.add_column("roles/tags", style="dim")

    for name in sorted(entries):
        e = entries[name]
        is_active = name == active_entry

        ctx_str = f"{e.ctx_window // 1024}k" if e.ctx_window >= 1024 else str(e.ctx_window)
        out_str = f"{e.max_output_tokens // 1024}k" if e.max_output_tokens >= 1024 else str(e.max_output_tokens)
        temp_str = f"{e.temperature:.2f}"
        params_str = f"{e.params_b:.0f}B" if e.params_b else "?"
        tps_str = f"{e.tokens_per_sec:.0f}" if e.tokens_per_sec else "—"
        local_str = "[green]✓[/green]" if e.local else ""
        think_str = "[cyan]✓[/cyan]" if e.thinking else ""
        cost_in_str = f"{e.cost_in_per_1k:.4f}" if e.cost_in_per_1k else "—"
        cost_out_str = f"{e.cost_out_per_1k:.4f}" if e.cost_out_per_1k else "—"

        role_parts = roles_rev.get(name, [])
        tag_parts = list(e.tags)
        badge_str = "  ".join(
            [f"[bold yellow]{r}[/bold yellow]" for r in role_parts] + tag_parts
        )

        name_str = f"[bold]{name}[/bold]" if is_active else name
        model_str = (f"[bold]{e.model}[/bold]" if is_active else e.model) or "[dim]—[/dim]"

        tbl.add_row(
            name_str, model_str, e.base_url,
            ctx_str, out_str, temp_str, params_str, tps_str,
            local_str, think_str,
            cost_in_str, cost_out_str, badge_str,
        )

    return tbl


def _apply_goal(agent, arg: str) -> tuple[bool, str]:
    """Handle /goal [<text> | $<cmd> | clear]. Returns (ok, message)."""
    v = arg.strip()
    if not v:
        cur = getattr(agent.config.llm, "goal", None)
        max_i = getattr(agent.config.llm, "goal_max_iterations", 200)
        if cur is None:
            return True, "No goal set. Usage: /goal <description>  or  /goal $ <shell-cmd>"
        return True, f"goal: {cur}\ngoal_max_iterations: {max_i}"
    if v.lower() in ("clear", "off", "none", "-"):
        agent.config.llm.goal = None
        if hasattr(agent.config, "agent"):
            agent.config.agent.goal = None
        return True, "Goal cleared. Agent will stop at max_iterations as usual."
    agent.config.llm.goal = v
    if hasattr(agent.config, "agent"):
        agent.config.agent.goal = v
    max_i = getattr(agent.config.llm, "goal_max_iterations", 200)
    kind = "shell check" if v.startswith("$") else "LLM-evaluated"
    return True, f"Goal set ({kind}): {v}\nAgent will run until goal is achieved (hard ceiling: {max_i} iterations)."


def _match_commands(prefix: str) -> list[tuple[str, str, bool]]:
    """Return (primary_name, description, takes_arg) for commands whose primary
    name or any alias starts with *prefix* (case-insensitive)."""
    pl = prefix.lower()
    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if primary.startswith(pl) or any(a.startswith(pl) for a in aliases):
            out.append((primary, desc, takes_arg))
    return out
