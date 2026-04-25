"""Render helpers: context/output color tables, mini_bar, context report."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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


def _render_context_report(agent, theme) -> str:
    """Grid-style /context report. Combines breakdown + telemetry legend."""
    cfg = agent.config
    ctx = cfg.llm.ctx_window or 1
    breakdown = agent.context_breakdown()
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
        "  [bold rgb(198,40,40)]▕[/] compaction threshold "
        "(auto-compact fires here)   "
        "[bold rgb(186,85,211)]▕[/] peak in last agent round"
    )
    lines.append("")

    # ── Output breakdown key ───────────────────────────────────────────────
    out_breakdown = agent.output_breakdown("session")
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
