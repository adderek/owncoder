"""Slash-command registry and handlers for the terminal UI."""
from __future__ import annotations

import os

from typing import TYPE_CHECKING

from agent.ui.slash_plan import _active_plan, _render_plan, _apply_plan

if TYPE_CHECKING:
    from agent.config.models import Config


# (primary_name, aliases, short_description, takes_arg)
_SLASH_COMMANDS: list[tuple[str, list[str], str, bool]] = [
    ("/a", [], "switch to A (agent answers) tab", False),
    ("/bg", ["/background"], "background jobs: list | kill <id> | kill all", True),
    ("/loop", [], "repeat prompt in this session: [<30s|5m|1h>] [xN] <prompt> | stop", True),
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
    ("/perf", ["/timing"], "session performance: LLM vs tool time + slowest tools; 'all' = cross-session data-source usage", False),
    ("/modelcalls", ["/mc"], "show model calls this session by cost tier (local/free/bundled/paid); 'detail' adds role × model table; 'reset' clears", True),
    ("/who", ["/agents"], "list other agents active on this worktree", False),
    ("/continue", ["/c"], "resume after iteration cap or truncation", False),
    ("/goal", [], "set/show/clear completion goal  [<text> | $ <cmd> | clear]", True),
    ("/exec", [], "run a shell command", True),
    ("/export", [], "export conversation as markdown", False),
    ("/help", ["/?"], "show this help", False),
    ("/load", [], "load a saved session", True),
    ("/q", [], "switch to Q (user questions) tab", False),
    ("/reset", [], "drop conversation history", False),
    ("/resume", ["/session"], "search past sessions and pick one to load", True),
    ("/save", [], "save session under a name", True),
    ("/sessions", [], "list saved sessions (oldest→newest; [N|all])", True),
    ("/sparse", [], "switch to sparse (condensed dialogue) tab", False),
    (
        "/temperature",
        ["/temp"],
        "set sampling temperature (0.0–2.0, - or default to reset)",
        True,
    ),
    ("/think", [], "set thinking level  off|low|normal|high|max", True),
    ("/autonomy", ["/auto", "/verbose"], "set autonomy level  0.0–1.0 (or %) or supervised|explain|balanced|brisk|autopilot", True),
    ("/mode", [], "show/switch model-mode  local-only|free-cloud|free-hybrid|paid-cloud|manual|any", True),
    ("/effort", [], "show/set per-turn model effort  quick|smart|deep (power ladder)", True),
    ("/max_tokens", ["/maxtokens"], "set max tokens   [out <n> | in <n> | <n> | default]", True),
    ("/tokens", [], "show token usage vs context window", False),
    ("/unlimited", ["/nomax"], "toggle unlimited tool-call iterations", False),
    ("/incognito", [], "toggle incognito mode (session not saved)", False),
    ("/private", [], "toggle private mode (no persistence + local LLMs only)", False),
    ("/vault", [], "toggle vault mode (everything persists, encrypted)  [lock]", True),
    ("/paths", [], "path grants: show | add <path> [ro|rw] | remove <path> | list", True),
    ("/maxiter", ["/max_iter"], "set max tool-call iterations per turn  [<n> | 0/none = unlimited]", True),
    ("/wrap", [], "toggle line wrapping", False),
    ("/round-summary", ["/summary"], "toggle gray Q/A summary after each turn", False),
    ("/tools", [], "list available tools", False),
    ("/skills", [], "skills: list | show <name> | history <name> | rm <name>", True),
    ("/commands", ["/cmds"], "list project ':name' commands from .agent/commands/", False),
    ("/undo", [], "restore last file snapshot", False),
    ("/checkpoint", ["/cp"], "checkpoint: list | new [label] | rollback <id> | prune", True),
    ("/memory", ["/mem"], "memory + index overview  [index | notes [n]]", True),
    ("/mcp", [], "show MCP server status + their tools", False),
    ("/sandbox", [], "sandbox mask scan: limits + last scan cost  [status | scan]", True),
    ("/speech", [], "speech-to-text input status  [status]", True),
    ("/security", ["/sec", "/audit"], "security: scan|diff|triage|selfaudit|report [path] | baseline [...] | airgap [...] | integrity [seal|check] | weights [pin|verify|list] | sbom [path] | taint [path] | evolve | knowledge | verify [<i>|run] | full [path] | review [path]", True),
    ("/plan", [], "plan: new <goal> | show | steps | step <id> <status> | dep <step> <dep> | assign <step> <agent> | compact | abort | pause | stash | resume", True),
    ("/plans", [], "list saved plans", False),
    ("/abort-plan", [], "mark active plan aborted (no stash)", False),
    ("/stash-plan", [], "git stash current changes + mark plan stashed", False),
    ("/pause-plan", [], "mark active plan paused; resume later", False),
    ("/schedule", ["/sched"], "scheduled jobs: list | add <spec> :: <prompt> [:: <name>] | rm <id|name> | on/off <id|name> | runs | run", True),
    ("/watch", [], "event watches: list | add <file|url|cmd|pid> <target> :: <prompt> [:: <name>] | rm | on/off", True),
    ("/credpool", ["/creds"], "credential pool: list|status | add <service> <domain> <username> <password> | remove <service>", True),
    ("/permissions", ["/perms"], "tool permissions: list | add <allow|ask|deny> <tool> [match] | default <verdict> | clear", True),
    ("/hooks", [], "shell hooks: list | approve <n> | revoke <n|digest>", True),
    ("/notify", [], "notification channels  [on | off | status]", True),
    ("/model", [], "switch active model  [<entry> | auto | role=<entry> | role=? | refresh]", True),
    ("/models", [], "model entries: table + toggles  [table | enable <name> | disable <name> | reload [project]]", False),
    ("/heal", ["/introspect", "/diagnose"], "self-diagnose this session's failures and fix the root cause  [<what you observed> | why]", True),
    ("/recoveries", [], "list pending crash-recovery records", False),
    ("/resummarize", [], "re-summarize Q/A entries with stale or missing summaries  [--force]", True),
    ("/idea", [], "ideas: <title> | add [--type T] [--tags t] [--priority N] <title> [| body] | show <id> | update <id> k=v | done <id> | reject <id>", True),
    ("/ideas", [], "list ideas  [status_filter: raw|evaluated|planned|implementing|verifying|done|rejected]", True),
    ("/quit", ["/exit", "/q!"], "quit the agent", False),
]


# ── Menu grouping ───────────────────────────────────────────────────────────
# Typing "/" only helps someone who already knows the name. The browser builds
# a menu from these groups, so the commands are findable by what they are for.
# Order here is the order the menu shows.
_GROUP_ORDER = (
    "session", "memory & search", "model & context", "work", "code & safety",
    "privacy & access", "automation", "diagnostics",
)

_GROUPS: dict[str, tuple[str, ...]] = {
    "session": ("/sessions", "/resume", "/load", "/save", "/reset", "/clear",
                "/export", "/who", "/quit"),
    "memory & search": ("/memory", "/skills", "/commands", "/resummarize",
                        "/compact", "/context", "/tokens"),
    "model & context": ("/model", "/models", "/mode", "/effort", "/think",
                        "/temperature", "/max_tokens", "/autonomy",
                        "/maxiter", "/unlimited", "/mcp"),
    "work": ("/plan", "/plans", "/goal", "/continue", "/idea", "/ideas",
             "/abort-plan", "/stash-plan", "/pause-plan", "/loop", "/bg"),
    "code & safety": ("/security", "/analyze-asm", "/undo", "/checkpoint",
                      "/apply", "/heal", "/recoveries"),
    "privacy & access": ("/incognito", "/private", "/vault", "/paths",
                         "/permissions", "/hooks", "/credpool", "/notify"),
    "automation": ("/schedule", "/watch"),
    "diagnostics": ("/perf", "/modelcalls", "/output", "/speech", "/sandbox", "/wrap",
                    "/round-summary", "/tools", "/help"),
}

# Ready-made arguments for the commands whose useful forms are a short list.
# A menu that only ever fills in "/security " is a longer way to type it.
_PRESETS: dict[str, tuple[tuple[str, str], ...]] = {
    "/security": (("scan", "scan the project"), ("diff", "scan changed files"),
                  ("review", "LLM deep read"), ("full", "full posture"),
                  ("sbom", "dependencies"), ("triage", "rank findings"),
                  ("airgap status", "egress status")),
    "/analyze-asm": (("on", "enable for this session"), ("off", "disable"),
                     ("stop", "interrupt a running analysis")),
    "/memory": (("", "overview"), ("index", "index freshness"),
                ("notes", "recent notes")),
    "/skills": (("list", "list skills"),),
    "/checkpoint": (("list", "list checkpoints"), ("new", "new checkpoint")),
    "/schedule": (("list", "list jobs"), ("runs", "recent runs")),
    "/watch": (("list", "list watches"),),
    "/mode": (("local-only", "local models only"), ("free-hybrid", "local + free cloud"),
              ("paid-cloud", "paid cloud")),
    "/effort": (("quick", "fast"), ("smart", "balanced"), ("deep", "thorough")),
    "/think": (("off", "no thinking"), ("normal", "default"), ("high", "more"),
               ("max", "most")),
    "/bg": (("list", "list background jobs"),),
    "/plan": (("show", "current plan"), ("steps", "step list")),
    "/paths": (("show", "granted paths"),),
    "/vault": (("lock", "lock the vault"),),
    "/sandbox": (("status", "limits + last scan"), ("scan", "scan the tree now")),
}


def command_group(name: str) -> str:
    """Which menu section a command belongs in. Unlisted commands land in
    'other' rather than vanishing — a new command should show up unprompted."""
    for group, names in _GROUPS.items():
        if name in names:
            return group
    return "other"


def command_presets(name: str) -> list[dict]:
    return [{"arg": arg, "label": label}
            for arg, label in _PRESETS.get(name, ())]


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
    "chat": "default",
    "sum": "summarizer",
    "summarizer": "summarizer",
    "emb": "embeddings",
    "embeddings": "embeddings",
    # Per-purpose roles (the matrix). Resolve to default when unpinned.
    "bg": "background",
    "background": "background",
    "idle": "background",
    "namer": "namer",
    "compaction": "compaction",
    "compact": "compaction",
    "review": "review",
    "triage": "triage",
    "verify": "verify",
    "evolve": "evolve",
    "commit": "commit",
}


def _release_role_pin(cfg, role: str) -> None:
    """Forget that *role* was pinned by hand this session.

    `session_role_pins` is what `config.reload.reload_models` consults to decide
    whether a role assignment in the config file may overwrite a live choice:

        if role in session_pins and config.model_roles.get(role) in config.model_entries:
            continue   # live session pin takes precedence

    Clearing `runtime_model_pinned` alone is not enough. The set was write-only
    -- nothing anywhere removed from it -- so a released pin still counted as
    live, and `/models reload` silently ignored a changed `default` role while
    reporting success. Released has to mean released, or the file can never win
    it back.

    Only "default" carries `runtime_model_pinned`; every role carries a session
    pin, so this is the piece that has to be role-generic.
    """
    pins = getattr(cfg, "session_role_pins", None)
    if pins is not None:
        pins.discard(role)


def _apply_model(agent, arg: str) -> tuple[bool, str]:
    """Handle /model [role=]<entry-name>.  Returns (ok, message)."""
    cfg = agent.config
    entries = cfg.model_entries

    def _status() -> str:
        from agent.config import make_registry
        cur_model = cfg.llm.model
        cur_url = cfg.llm.base_url
        reg = make_registry(cfg)
        pinned = ("pinned by /model — auto-tier stands down (release: /model auto)"
                  if getattr(cfg, "runtime_model_pinned", False)
                  else "not pinned — auto-tier picks per turn")
        lines = [
            f"active LLM: [bold]{cur_model}[/bold]  ({cur_url})",
            f"model-mode: {cfg.agent.model_mode}",
            f"pin: {pinned}",
            "purpose → model matrix  (pin: /model <role>=<entry>; release: /model <role>=auto):",
        ]
        for role, (entry_name, tier) in reg.matrix().items():
            pinned = "*" if role in cfg.model_roles else " "
            lines.append(f" {pinned} {role:11s} {entry_name:16s} [{tier}]")
        lines.append("available entries:")
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
        if role_raw in _ROLE_ALIASES:
            role = _ROLE_ALIASES[role_raw]
        else:
            # The HTTP UI builds its role dropdown from the registry matrix
            # (ROLE_FALLBACKS), so every role it can offer has to be pinnable
            # here — otherwise the GUI offers a role `/model` rejects (this is
            # how `judge` was shown yet rejected as "Unknown role"). Derive the
            # fallback from that one source instead of mirroring the list.
            try:
                from agent.config.registry import ROLE_FALLBACKS
            except Exception:
                ROLE_FALLBACKS = {}
            if role_raw not in ROLE_FALLBACKS:
                known = ", ".join(sorted(set(_ROLE_ALIASES) | set(ROLE_FALLBACKS)))
                return False, f"Unknown role '{role_raw}'. Known: {known}"
            role = role_raw

    # A role forced by the environment cannot be pinned or released: the env
    # value is re-applied after the model-entry bridge and on every reload and
    # outranks a live pin, so accepting the command would silently do nothing.
    # The UI greys the control out; this is the CLI parity guard.
    from agent.config.loader import env_locked_roles, _ENV_MODEL_FORCED
    if role in env_locked_roles():
        envs = ", ".join(k for k in _ENV_MODEL_FORCED.get(role, ()) if os.environ.get(k))
        return False, (f"role '{role}' is enforced by the environment ({envs}); "
                       f"it outranks any session pin — unset it to pin from here.")

    if entry_name == "?":
        return True, _status()

    if entry_name == "auto":
        if role == "default":
            was = getattr(cfg, "runtime_model_pinned", False)
            cfg.runtime_model_pinned = False
            _release_role_pin(cfg, role)
            return True, ("model pin released — auto-tier picks per turn again."
                          if was else "no model pin was set; auto-tier is already choosing.")
        # Non-default role: drop the explicit pin so the fallback ladder
        # (ROLE_FALLBACKS, free-cloud offload) decides per call again.
        was = cfg.model_roles.pop(role, None)
        _release_role_pin(cfg, role)
        return True, (
            f"role '{role}' → auto (released '{was}'; ladder picks per call)."
            if was else f"role '{role}' was not pinned; ladder already decides.")

    if entry_name not in entries:
        known = ", ".join(sorted(entries))
        return False, f"No model entry '{entry_name}'. Available: {known}"

    entry = entries[entry_name]
    cfg.model_roles[role] = entry_name
    if not hasattr(cfg, "session_role_pins"):
        cfg.session_role_pins = set()
    cfg.session_role_pins.add(role)

    if role == "default":
        # Hand-picked mid-session: auto-tier stands down (per-turn ladder pick
        # and mid-turn escalation both) until "/model auto" or an /effort change.
        cfg.runtime_model_pinned = True
        cfg.llm.base_url = entry.base_url
        cfg.llm.api_key = entry.api_key
        if entry.model:
            cfg.llm.model = entry.model
        cfg.llm.ctx_window = entry.ctx_window
        cfg.llm.max_output_tokens = entry.max_output_tokens
        cfg.llm.temperature = entry.temperature
        cfg.llm.assume_available = getattr(entry, "assume_available", False)
        # Recreate the OpenAI client with new endpoint/key
        from agent.core.llm_client import make_llm_client
        agent._client = make_llm_client(cfg, base_url=entry.base_url, api_key=entry.api_key)
        msg = (
            f"switched default → [bold]{entry_name}[/bold]  "
            f"model={cfg.llm.model}  url={cfg.llm.base_url}"
        )
        if getattr(cfg.auto_tier, "enabled", False):
            msg += "\n[dim]pinned — auto-tier stands down until /model auto[/dim]"
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


def handle_models_reload(config: "Config", arg: str, agent=None) -> tuple[bool, str]:
    """Handle '/models reload [project]'.

    Re-reads only the [models] section of the config layers (see
    agent.config.reload for why nothing else is reloaded). User-typed only:
    no tool exposes slash commands to the model, and that is what stops the
    agent from re-pointing its own endpoint by writing a config file.
    """
    include_project = arg.strip().lower() in ("project", "--project", "all")
    # A reload swaps model entries under whatever is running; the turn loop
    # reads config.llm mid-flight, so refuse rather than switch endpoints
    # between two calls of the same turn.
    if agent is not None and getattr(agent, "_turn_busy", False):
        return False, "A turn is in progress — retry /models reload when it finishes."
    from agent.config.reload import reload_models
    ok, msg = reload_models(config, include_project=include_project)
    if ok and agent is not None:
        from agent.core.llm_client import make_llm_client
        agent._client = make_llm_client(config)
        try:
            from agent.metrics.model_stats import resolve_entry_name
            agent._model_entry_name = resolve_entry_name(config)
        except Exception:
            pass
    return ok, msg


def handle_models_toggle(config: "Config", arg: str, agent=None) -> tuple[bool, str] | None:
    """Handle '/models enable|disable|enable-save|disable-save <name>' and
    '/models reload'. None when arg isn't one of those (caller shows the
    table instead)."""
    parts = (arg or "").split()
    if parts and parts[0] == "reload":
        return handle_models_reload(config, " ".join(parts[1:]), agent)
    ACTIONS = ("enable", "disable", "enable-save", "disable-save")
    if len(parts) == 2 and parts[0] in ACTIONS:
        save = parts[0].endswith("-save")
        enabled = parts[0].startswith("enable")
        if save:
            from agent.core.model_control import save_model_enabled
            return save_model_enabled(config, parts[1], enabled)
        from agent.core.model_control import set_model_enabled
        return set_model_enabled(config, parts[1], enabled)
    if len(parts) == 1 and parts[0] in ACTIONS:
        return False, f"usage: /models {parts[0]} <entry-name>"
    return None


def _render_models_table(config: "Config", probe: bool = True):
    """Return a Rich Table showing all model_entries with capabilities.

    When *probe* is True, each entry's endpoint is queried once (cached per
    base_url) to report whether the configured model is actually live: a green
    ✓ (served), red ✗ (endpoint up but model missing, or endpoint unreachable).
    """
    from rich.table import Table

    entries = config.model_entries
    roles_rev: dict[str, list[str]] = {}
    for role, entry_name in config.model_roles.items():
        roles_rev.setdefault(entry_name, []).append(role)

    active_entry = config.model_roles.get("default", "default")

    # Best-effort availability probe — one /models call per unique endpoint.
    _models_cache: dict[str, "set | None"] = {}

    def _live_cell(e) -> str:
        if not probe or not e.base_url:
            return ""
        from agent.config.model_probe import list_endpoint_models, model_in_server
        if e.base_url not in _models_cache:
            _models_cache[e.base_url] = list_endpoint_models(
                e.base_url, getattr(e, "api_key", ""), timeout=2
            )
        ids = _models_cache[e.base_url]
        if ids is None:
            return "[red]✗[/red]"  # endpoint unreachable
        if model_in_server(e.model or "", ids):
            return "[green]✓[/green]"
        # Served but not advertised (dated aliases, private deployments): the
        # entry's assume_available waives the catalog check the same way
        # entry_available/check_model_availability do — otherwise every unlisted
        # entry (any role) shows ✗ even though calls to it succeed.
        if getattr(e, "assume_available", False):
            return "[green]✓[/green]"
        return "[red]✗[/red]"

    tbl = Table(show_header=True, header_style="bold", box=None, pad_edge=False, collapse_padding=True)
    tbl.add_column("name", style="cyan", no_wrap=True)
    tbl.add_column("model id", no_wrap=True)
    tbl.add_column("live", justify="center", no_wrap=True)
    tbl.add_column("st", justify="center", no_wrap=True)   # off (disabled) / cool (failure cooldown)
    tbl.add_column("tier", no_wrap=True)                    # local / free / bundled / paid
    tbl.add_column("endpoint", style="dim", no_wrap=True)
    tbl.add_column("ctx", justify="right", no_wrap=True)
    tbl.add_column("out", justify="right", no_wrap=True)
    tbl.add_column("temp", justify="right", no_wrap=True)
    tbl.add_column("params", justify="right", no_wrap=True)
    tbl.add_column("tok/s", justify="right", no_wrap=True)
    tbl.add_column("ok%", justify="right", no_wrap=True)  # success rate, last 24h
    tbl.add_column("L", justify="center", no_wrap=True)  # local
    tbl.add_column("T", justify="center", no_wrap=True)  # thinking
    tbl.add_column("$/in", justify="right", no_wrap=True)
    tbl.add_column("$/out", justify="right", no_wrap=True)
    tbl.add_column("roles/tags", style="dim")

    from agent.config.registry import entry_tier
    from agent.core.model_control import entry_status
    _TIER_STYLE = {"local": "green", "free": "cyan", "bundled": "yellow", "paid": "magenta"}

    for name in sorted(entries):
        e = entries[name]
        is_active = name == active_entry

        _st = entry_status(config, name, e)
        st_str = {"off": "[red]off[/red]", "cool": "[yellow]cool[/yellow]"}.get(_st, "")
        _tier = entry_tier(e)
        tier_str = f"[{_TIER_STYLE.get(_tier, 'white')}]{_tier}[/{_TIER_STYLE.get(_tier, 'white')}]"

        ctx_str = f"{e.ctx_window // 1024}k" if e.ctx_window >= 1024 else str(e.ctx_window)
        out_str = f"{e.max_output_tokens // 1024}k" if e.max_output_tokens >= 1024 else str(e.max_output_tokens)
        temp_str = f"{e.temperature:.2f}"
        params_str = f"{e.params_b:.0f}B" if e.params_b else "?"
        tps_str = f"{e.tokens_per_sec:.0f}" if e.tokens_per_sec else "—"
        try:
            from agent.metrics.model_reliability import reliability_summary
            _rel = reliability_summary(name)
            ok_str = f"{_rel['success_rate'] * 100:.0f}% ({_rel['total']})" if _rel["success_rate"] is not None else "—"
        except Exception:
            ok_str = "—"
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
            name_str, model_str, _live_cell(e), st_str, tier_str, e.base_url,
            ctx_str, out_str, temp_str, params_str, tps_str, ok_str,
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


def _apply_bg(arg: str) -> tuple[bool, str]:
    """Handle /bg [kill <id>|kill all]. Returns (ok, message)."""
    from agent.core import background
    v = arg.strip().lower()
    if v.startswith("kill"):
        target = v[4:].strip()
        if target in ("all", "*"):
            n = background.cancel_all()
            return True, f"cancelled {n} background job(s)."
        if target.isdigit():
            ok = background.cancel(int(target))
            return ok, (f"job {target} cancelled." if ok
                        else f"job {target} not found or not killable.")
        return False, "Usage: /bg [kill <id> | kill all]"
    jobs = background.jobs()
    if not jobs:
        return True, "no background jobs running."
    lines = ["background jobs:"]
    for j in jobs:
        mark = "" if j["killable"] else "  (not killable)"
        lines.append(f"  [{j['id']}] {j['kind']:<10} {j['label']}  {j['age']:.0f}s{mark}")
    lines.append("kill with: /bg kill <id>  |  /bg kill all")
    return True, "\n".join(lines)


def _match_commands(prefix: str) -> list[tuple[str, str, bool]]:
    """Return (primary_name, description, takes_arg) for commands whose primary
    name or any alias contains *prefix* (case-insensitive substring match).
    When prefix is just "/" all commands are returned."""
    pl = prefix.lower()
    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if pl == "/" or pl in primary or any(pl in a for a in aliases):
            out.append((primary, desc, takes_arg))
    return out
