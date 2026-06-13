"""Render helpers: context/output color tables, mini_bar, context report."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# LaTeX math → Unicode substitution table
_LATEX_MAP: dict[str, str] = {
    r"\rightarrow": "→",
    r"\to": "→",
    r"\leftarrow": "←",
    r"\gets": "←",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔",
    r"\uparrow": "↑",
    r"\downarrow": "↓",
    r"\Uparrow": "⇑",
    r"\Downarrow": "⇓",
    r"\nearrow": "↗",
    r"\searrow": "↘",
    r"\swarrow": "↙",
    r"\nwarrow": "↖",
    r"\infty": "∞",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\pm": "±",
    r"\times": "×",
    r"\div": "÷",
    r"\cdot": "·",
    r"\ldots": "…",
    r"\cdots": "⋯",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\supset": "⊃",
    r"\subseteq": "⊆",
    r"\supseteq": "⊇",
    r"\cup": "∪",
    r"\cap": "∩",
    r"\emptyset": "∅",
    r"\nabla": "∇",
    r"\partial": "∂",
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\sqrt": "√",
    r"\neg": "¬",
    r"\land": "∧",
    r"\lor": "∨",
    r"\oplus": "⊕",
    r"\otimes": "⊗",
}

# Matches $\command$ or $\command{...}$ — simple inline math only
_LATEX_RE = re.compile(r"\$\\([A-Za-z]+)(?:\{[^}]*\})?\$")


def _delatex(text: str) -> str:
    """Replace inline LaTeX math tokens (``$\\cmd$``) with Unicode equivalents."""
    def _sub(m: re.Match) -> str:
        key = "\\" + m.group(1)
        return _LATEX_MAP.get(key, m.group(0))
    return _LATEX_RE.sub(_sub, text)


# Single source of truth. Top-bar widgets (ContextBreakdownBar,
# OutputBreakdownBar) and the /context report both read from these — any
# edit here propagates to the bars and the legend in lockstep.
_CTX_SEGMENT_COLORS: dict[str, str] = {
    "agent_prompt": "rgb(26,95,168)",
    "user_context": "rgb(0,155,155)",
    "tools_schema": "rgb(185,130,0)",
    "skills":       "rgb(148,45,190)",
    "user_input":   "rgb(30,150,60)",
    "assistant":    "rgb(100,100,115)",
    "tool_results": "rgb(190,35,35)",
}

_CTX_SEGMENT_LABELS: dict[str, str] = {
    "agent_prompt": "sys",
    "user_context": "user",
    "tools_schema": "tools",
    "skills":       "skills",
    "user_input":   "input",
    "assistant":    "asst",
    "tool_results": "result",
}

_CTX_SEGMENT_DESCS: dict[str, str] = {
    "agent_prompt": "system prompt (agent instructions)",
    "user_context": "injected user context (CLAUDE.md, rules, memory)",
    "tools_schema": "tool JSON schemas sent to the model",
    "skills":       "skill definitions loaded this session",
    "user_input":   "user messages",
    "assistant":    "assistant replies + reasoning",
    "tool_results": "tool call results fed back to the model",
}

_OUT_SEGMENT_COLORS: dict[str, str] = {
    "reasoning": "rgb(148,45,190)",
    "tool":      "rgb(185,130,0)",
    "content":   "rgb(100,100,115)",
    "other":     "rgb(26,95,168)",
}

_OUT_SEGMENT_LABELS: dict[str, str] = {
    "reasoning": "think",
    "tool":      "tool",
    "content":   "text",
    "other":     "other",
}

_OUT_SEGMENT_DESCS: dict[str, str] = {
    "reasoning": "think / chain-of-thought tokens",
    "tool":      "tool call arguments",
    "content":   "user-visible reply text",
    "other":     "everything else (framing, role tags)",
}


# Per-tool emoji icons. Used by all UI layers (textual + readline).
# When a tool name is not found, callers fall back to "⚙".
TOOL_ICONS: dict[str, str] = {
    # File operations
    "read_file":            "📖",
    "list_files":           "📂",
    "write_file":           "📝",
    "edit_file":            "✏️",
    "replace_symbol":       "🔀",
    "undo_file":            "↩️",
    # Shell / execution
    "run_argv":             "▶️",
    # Git
    "git_diff":             "±",
    "git_log":              "📜",
    "git_blame":            "🕵",
    "git_status":           "📋",
    "git_related_files":    "🔗",
    # Search / code navigation
    "grep_code":            "🔍",
    "search_code":          "🔎",
    "search_archive":       "🗄",
    # Memory / recall
    "recall_facts":         "💭",
    "recall_history":       "📚",
    "recall_sessions":      "🕐",
    "save_note":            "📌",
    # Session
    "rate_session":         "⭐",
    "retrieve_output":      "📤",
    # Web
    "web_search":           "🌐",
    "web_fetch":            "🌐",
    # Indexing
    "index_code":           "🗂",
    "graph_status":          "🩺",
    "graph_query":           "🔍",
    "graph_path":            "📡",
    "graph_context":        "👁️",
    "project_file_stats":   "📊",
    "analyze_asm":          "🔬",
    # Skills / agents
    "search_skills":        "🛠",
    "load_skill":           "⚡",
    "save_skill":           "💾",
    "skill_history":        "🕑",
    "rollback_skill":       "⏪",
    "manage_skills":        "🛠",
    "spawn_agents":         "🤖",
    # Knowledge base
    "kb_search":            "🔍",
    "kb_get":               "📖",
    "kb_deps":              "🕸",
    "kb_callers":           "🔗",
    "kb_add_note":          "📌",
    "kb_propose_description": "✍",
    # Ideas
    "submit_idea":          "💡",
    # Planning / increment
    "snapshot_step":        "📸",
    "complete_step":        "✅",
    "plan_ready_steps":     "📐",
    "plan_add_dep":         "🔗",
    "plan_assign_step":     "👤",
    "revert_step":          "↩️",
    "get_step_brief":       "📋",
    "report_blocking_issue": "🚧",
}


def tool_icon(name: str) -> str:
    """Return the emoji icon for tool *name*, or '⚙' if none defined."""
    return TOOL_ICONS.get(name, "⚙")


def _mini_bar(fraction: float, color: str, width: int = 20) -> str:
    frac = max(0.0, min(1.0, fraction))
    filled = int(round(frac * width))
    return (
        f"[{color}]{'█' * filled}[/]"
        f"[dim]{'░' * (width - filled)}[/dim]"
    )


def _labeled_bar_segment(label: str, n: int, color: str) -> str:
    """n terminal cells: label text (truncated/padded) on colored background."""
    text = label[:n].ljust(n)
    return f"[white on {color}]{text}[/]"


def _render_context_report(server, theme) -> str:
    """Grid-style /context report. Combines breakdown + telemetry legend."""
    info = server.get_llm_info()
    ctx = info["ctx_window"] or 1
    breakdown = server.context_breakdown()
    total = sum(max(0, s.get("tokens", 0)) for s in breakdown)
    bar_w = 20

    lines: list[str] = []
    lines.append(f"[bold]Context breakdown[/bold]  [dim](ctx_window={ctx:,})[/dim]")
    lines.append("")
    header = (
        f"  [dim]{'':1} {'segment':<14} {'bar':<{bar_w}}  "
        f"{'tokens':>9} {'%':>6}  description[/dim]"
    )
    lines.append(header)

    for seg in breakdown:
        label = seg["label"]
        tok = max(0, seg.get("tokens", 0))
        pct = (tok / ctx * 100) if ctx else 0.0
        color = _CTX_SEGMENT_COLORS.get(label, "white")
        desc = _CTX_SEGMENT_DESCS.get(label, "")
        bar = _mini_bar(tok / ctx if ctx else 0, color, bar_w)
        lines.append(
            f"  [{color}]█[/] {label:<14} {bar}  "
            f"{tok:>9,} {pct:>5.1f}%  [dim]{desc}[/dim]"
        )

    free = max(0, ctx - total)
    total_pct = (total / ctx * 100) if ctx else 0.0
    lines.append("")
    lines.append(
        f"  [bold]{'total':<16}[/bold] {'':<{bar_w}}  "
        f"{total:>9,} {total_pct:>5.1f}%   [dim]free: {free:,}[/dim]"
    )
    lines.append("")

    # ── Token bar key ──────────────────────────────────────────────────────
    lines.append("[bold]tokens bar[/bold]  [dim]fill of ctx_window[/dim]")
    lines.append(
        "  [rgb(56,142,60) on rgb(56,142,60)]█[/] used under 65%   "
        "[rgb(249,168,37) on rgb(249,168,37)]█[/] 65–85%   "
        "[rgb(198,40,40) on rgb(198,40,40)]█[/] over 85%   "
        "[rgb(70,70,70) on rgb(30,30,30)] [/] free"
    )
    lines.append(
        "  [bold rgb(198,40,40)]🞀[/] compaction threshold "
        "(auto-compact fires here)   "
        "[bold rgb(186,85,211)]▕[/] peak in last agent round"
    )
    lines.append("")

    # ── Output breakdown key ───────────────────────────────────────────────
    out_breakdown = server.output_breakdown("session")
    out_total = sum(max(0, s.get("tokens", 0)) for s in out_breakdown)
    lines.append(
        f"[bold]output breakdown[/bold]  "
        f"[dim]session total: {out_total:,} tokens[/dim]"
    )
    for seg in out_breakdown:
        label = seg["label"]
        tok = max(0, seg.get("tokens", 0))
        pct = (tok / out_total * 100) if out_total else 0.0
        color = _OUT_SEGMENT_COLORS.get(label, "white")
        desc = _OUT_SEGMENT_DESCS.get(label, "")
        bar = _mini_bar(
            tok / out_total if out_total else 0, color, bar_w
        )
        lines.append(
            f"  [{color}]█[/] {label:<14} {bar}  "
            f"{tok:>9,} {pct:>5.1f}%  [dim]{desc}[/dim]"
        )
    c = theme.cmd_color
    lines.append(
        f"  [dim]scope:[/dim] session  "
        f"[dim]— last-turn view:[/dim] [{c}]/output last[/{c}]"
    )
    return "\n".join(lines)
