from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.config import Config


# ── Slash-command registry (used by completion and help) ────────────────────

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
    ("/think", [], "set thinking level  off|low|normal|high|max", True),
    ("/max_tokens", [], "set max tokens   [out <n> | in <n> | <n> | default]", True),
    ("/wrap", [], "toggle line wrapping", False),
    ("/tools", [], "list available tools", False),
    ("/undo", [], "restore last file snapshot", False),
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


def _match_commands(prefix: str) -> list[tuple[str, str, bool]]:
    """Return (primary_name, description, takes_arg) for commands whose primary
    name or any alias starts with *prefix* (case-insensitive)."""
    pl = prefix.lower()
    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if primary.startswith(pl) or any(a.startswith(pl) for a in aliases):
            out.append((primary, desc, takes_arg))
    return out


# ── Context/telemetry color tables (shared by top bars + /context report) ───

# Single source of truth. Top-bar widgets (ContextBreakdownBar,
# OutputBreakdownBar) and the /context report both read from these — any
# edit here propagates to the bars and the legend in lockstep.
_CTX_SEGMENT_COLORS: dict[str, str] = {
    "agent_prompt": "blue",
    "user_context": "cyan",
    "tools_schema": "yellow",
    "skills":       "magenta",
    "user_input":   "green",
    "assistant":    "bright_white",
    "tool_results": "red",
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
    "reasoning": "magenta",
    "tool":      "yellow",
    "content":   "bright_white",
    "other":     "blue",
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
        f"[{color}]{'█' * filled}[/{color}]"
        f"[dim]{'░' * (width - filled)}[/dim]"
    )


def _render_context_report(agent, theme) -> str:
    """Grid-style /context report. Combines breakdown + telemetry legend.

    Colors come from _CTX_SEGMENT_COLORS / _OUT_SEGMENT_COLORS so they match
    the top bars exactly.
    """
    cfg = agent.config
    ctx = cfg.llm.ctx_window or 1
    breakdown = agent.context_breakdown()
    total = sum(max(0, s.get("tokens", 0)) for s in breakdown)
    bar_w = 20

    lines: list[str] = []
    lines.append(f"[bold]Context breakdown[/bold]  [dim](ctx_window={ctx:,})[/dim]")
    lines.append("")
    header = (
        f"  [dim]{'':1} {'segment':<14} {'description':<44}"
        f"{'tokens':>9} {'%':>6}  bar[/dim]"
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
            f"  [{color}]█[/{color}] {label:<14} [dim]{desc:<44}[/dim]"
            f"{tok:>9,} {pct:>5.1f}%  {bar}"
        )

    free = max(0, ctx - total)
    total_pct = (total / ctx * 100) if ctx else 0.0
    lines.append("")
    lines.append(
        f"  [bold]{'total':<16}[/bold] {'':<44}"
        f"{total:>9,} {total_pct:>5.1f}%   [dim]free: {free:,}[/dim]"
    )
    lines.append("")

    # ── Token bar key ──────────────────────────────────────────────────────
    lines.append("[bold]tokens bar[/bold]  [dim]fill of ctx_window[/dim]")
    lines.append(
        "  [green]█[/green] used under 65%   "
        "[yellow]█[/yellow] 65–85%   "
        "[red]█[/red] over 85%   "
        "[dim]░[/dim] free"
    )
    lines.append(
        "  [bold red]│[/bold red] compaction threshold "
        "(auto-compact fires here)   "
        "[bold magenta]╋[/bold magenta] peak in last agent round"
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
            f"  [{color}]█[/{color}] {label:<14} [dim]{desc:<44}[/dim]"
            f"{tok:>9,} {pct:>5.1f}%  {bar}"
        )
    c = theme.cmd_color
    lines.append(
        f"  [dim]scope:[/dim] session  "
        f"[dim]— last-turn view:[/dim] [{c}]/output last[/{c}]"
    )
    return "\n".join(lines)


# ── Textual UI ──────────────────────────────────────────────────────────────


def _build_textual_app(agent: "Agent", session=None):
    t = agent.config.ui.theme  # shorthand used throughout

    from textual.app import App, ComposeResult
    from textual.widgets import (
        Footer,
        RichLog,
        Static,
        TextArea,
        LoadingIndicator,
        TabbedContent,
        TabPane,
    )
    from textual.containers import Horizontal
    from textual.binding import Binding
    from textual.message import Message
    from textual.worker import Worker, WorkerState
    from textual import work
    from rich.markup import escape as _escape
    from rich.markdown import Markdown

    class TokenBar(Static):
        """Full-width context-usage bar.

        Renders current usage as filled blocks. Overlays two markers:
          * compaction threshold (where auto-compaction fires)
          * peak usage observed in the most recent agent round
        """

        def __init__(self, ctx_window: int, compact_frac: float = 0.75, **kwargs):
            super().__init__("", **kwargs)
            self._ctx = ctx_window
            self._compact_frac = compact_frac
            self._used = 0
            self._peak = 0

        def update_tokens(self, used: int, peak: int = 0, compact_frac: float | None = None) -> None:
            self._used = used
            self._peak = peak
            if compact_frac is not None:
                self._compact_frac = compact_frac
            self._redraw()

        def on_resize(self, _event) -> None:
            self._redraw()

        def _redraw(self) -> None:
            ctx = max(1, self._ctx)
            used = max(0, self._used)
            peak = max(0, self._peak)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            label = f"tokens: {used:,}/{ctx:,}"
            bar_len = max(10, width - len(label) - 2)
            used_frac = min(1.0, used / ctx)
            peak_frac = min(1.0, peak / ctx) if peak > 0 else 0.0
            compact_frac = max(0.0, min(1.0, self._compact_frac))
            used_cells = int(round(used_frac * bar_len))
            peak_cell = int(round(peak_frac * bar_len)) if peak_frac > 0 else -1
            compact_cell = int(round(compact_frac * bar_len))
            if peak_cell >= bar_len:
                peak_cell = bar_len - 1
            if compact_cell >= bar_len:
                compact_cell = bar_len - 1
            fill_color = "red" if used_frac > 0.85 else ("yellow" if used_frac > 0.65 else "green")
            parts = []
            for i in range(bar_len):
                if i == peak_cell and peak_cell >= 0:
                    parts.append("[bold magenta]╋[/bold magenta]")
                elif i == compact_cell:
                    parts.append("[bold red]│[/bold red]")
                elif i < used_cells:
                    parts.append(f"[{fill_color}]█[/{fill_color}]")
                else:
                    parts.append("[dim]░[/dim]")
            self.update(f"{label} {''.join(parts)}")

    class ContextBreakdownBar(Static):
        """One-line segmented bar showing how the context window is filled.

        Each registered content-type occupies a slice of the bar proportional
        to its token share of the configured ctx_window. The unused tail is
        drawn dim.
        """

        def __init__(self, ctx_window: int, **kwargs):
            super().__init__("", **kwargs)
            self._ctx = ctx_window
            self._segments: list[dict] = []

        def set_segments(self, segments: list[dict]) -> None:
            self._segments = list(segments)
            self._redraw()

        def on_resize(self, _event) -> None:
            self._redraw()

        def _redraw(self) -> None:
            ctx = max(1, self._ctx)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            # Leave room for a compact legend on the right.
            total_used = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            label = f"ctx: {total_used:,}/{ctx:,}"
            bar_len = max(10, width - len(label) - 2)

            # Compute integer cell counts per segment; ensure non-zero segments
            # get at least one cell when there's room, so tiny slices remain
            # visible.
            cells = []
            remaining = bar_len
            for seg in self._segments:
                tok = max(0, seg.get("tokens", 0))
                raw = tok / ctx * bar_len
                n = int(raw)
                if tok > 0 and n == 0 and remaining > 0:
                    n = 1
                if n > remaining:
                    n = remaining
                remaining -= n
                cells.append(n)

            parts = []
            for seg, n in zip(self._segments, cells):
                if n <= 0:
                    continue
                color = _CTX_SEGMENT_COLORS.get(seg["label"], "white")
                parts.append(f"[{color}]{'█' * n}[/{color}]")
            if remaining > 0:
                parts.append(f"[dim]{'░' * remaining}[/dim]")
            self.update(f"{label} {''.join(parts)}")

    class OutputBreakdownBar(Static):
        """One-line segmented bar showing how model *output* tokens were spent.

        Segments sum to 100% of the bar (scale = total output). Shows where
        generation budget goes: thinking vs tool args vs user reply vs other.
        """

        def __init__(self, **kwargs):
            super().__init__("", **kwargs)
            self._segments: list[dict] = []
            self._scope_label = "out"

        def set_segments(self, segments: list[dict], scope_label: str = "out") -> None:
            self._segments = list(segments)
            self._scope_label = scope_label
            self._redraw()

        def on_resize(self, _event) -> None:
            self._redraw()

        def _redraw(self) -> None:
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            total = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            label = f"{self._scope_label}: {total:,}"
            bar_len = max(10, width - len(label) - 2)
            if total <= 0:
                self.update(f"{label} [dim]{'░' * bar_len}[/dim]")
                return

            cells = []
            remaining = bar_len
            for seg in self._segments:
                tok = max(0, seg.get("tokens", 0))
                raw = tok / total * bar_len
                n = int(raw)
                if tok > 0 and n == 0 and remaining > 0:
                    n = 1
                if n > remaining:
                    n = remaining
                remaining -= n
                cells.append(n)

            parts = []
            for seg, n in zip(self._segments, cells):
                if n <= 0:
                    continue
                color = _OUT_SEGMENT_COLORS.get(seg["label"], "white")
                parts.append(f"[{color}]{'█' * n}[/{color}]")
            if remaining > 0:
                parts.append(f"[dim]{'░' * remaining}[/dim]")
            self.update(f"{label} {''.join(parts)}")

    class ConversationView(RichLog):
        """Live chat log — user ↔ agent turns."""

    class SysView(RichLog):
        """System log — commands, session info, help output."""

    def _one_line(text: str, limit: int = 120, wrap: bool = False) -> str:
        text = (text or "").strip()
        if not wrap:
            text = text.replace("\n", " ")
            if len(text) > limit:
                text = text[: limit - 1] + "…"
        return text

    class JumpToTurn(Message):
        """Posted when a user clicks an entry in Q/A/Sparse views."""

        def __init__(self, ordinal: int) -> None:
            super().__init__()
            self.ordinal = ordinal

    class _QALineTrackingMixin:
        """Mixin providing click-to-jump behavior shared by Q/A/Sparse views."""

        def _reset_line_map(self) -> None:
            self._line_ordinals: list[int] = []
            self._entry_count: int = 0

        def _track_line(self, ordinal: int) -> None:
            if not hasattr(self, "_line_ordinals"):
                self._line_ordinals = []
            self._line_ordinals.append(ordinal)

        def on_click(self, event) -> None:
            ordinals = getattr(self, "_line_ordinals", [])
            if not ordinals:
                return
            line_idx = int(self.scroll_offset.y) + int(event.y)
            if 0 <= line_idx < len(ordinals):
                self.post_message(JumpToTurn(ordinals[line_idx]))

    class QView(_QALineTrackingMixin, RichLog):
        """View for user questions: one line per turn."""

        def load_history(self, entries: "list[tuple[int, dict, dict]]") -> None:
            self.clear()
            self._reset_line_map()
            for ordinal, (tid, q, _a) in enumerate(entries):
                self._write_entry(ordinal, tid, q)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, q_data: dict, _a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            self._write_entry(ordinal, turn_id, q_data)

        def _write_entry(self, ordinal: int, turn_id: int, q: dict) -> None:
            if not q:
                return
            text = q.get("summary_q") or q.get("content") or ""
            self._track_line(ordinal)
            self.write(
                f"[{t.text_dim}]{turn_id:>3}[/{t.text_dim}] [bold {t.user_color}]Q:[/bold {t.user_color}] {_escape(_one_line(text, wrap=self.app._wrap_enabled))}"
            )

    class AView(_QALineTrackingMixin, RichLog):
        """View for agent answers: one line per turn."""

        def load_history(self, entries: "list[tuple[int, dict, dict]]") -> None:
            self.clear()
            self._reset_line_map()
            for ordinal, (tid, _q, a) in enumerate(entries):
                self._write_entry(ordinal, tid, a)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, _q_data: dict, a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            self._write_entry(ordinal, turn_id, a_data)

        def _write_entry(self, ordinal: int, turn_id: int, a: dict) -> None:
            if not a:
                return
            text = a.get("summary_a") or a.get("content") or ""
            self._track_line(ordinal)
            self.write(
                f"[{t.text_dim}]{turn_id:>3}[/{t.text_dim}] [bold {t.agent_color}]A:[/bold {t.agent_color}] {_escape(_one_line(text, wrap=self.app._wrap_enabled))}"
            )

    class SparseView(_QALineTrackingMixin, RichLog):
        """Condensed dialogue: Q and A interleaved by turn_id."""

        def load_history(self, entries: "list[tuple[int, dict, dict]]") -> None:
            self.clear()
            self._reset_line_map()
            for ordinal, (tid, q, a) in enumerate(entries):
                self._write_entry(ordinal, tid, q, a)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, q_data: dict, a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            self._write_entry(ordinal, turn_id, q_data, a_data)

        def _write_entry(self, ordinal: int, turn_id: int, q: dict, a: dict) -> None:
            tag = f"[{t.text_dim}]{turn_id:>3}[/{t.text_dim}]"
            q_text = (q or {}).get("summary_q") or (q or {}).get("content") or ""
            a_text = (a or {}).get("summary_a") or (a or {}).get("content") or ""
            if q_text:
                self._track_line(ordinal)
                self.write(
                    f"{tag} [bold {t.user_color}][User][/bold {t.user_color}] {_escape(_one_line(q_text, wrap=self.app._wrap_enabled))}"
                )
            if a_text:
                self._track_line(ordinal)
                self.write(
                    f"    [bold {t.agent_color}][Agent][/bold {t.agent_color}] {_escape(_one_line(a_text, wrap=self.app._wrap_enabled))}"
                )

    class ContextPanel(Static):
        def set_context(self, text: str) -> None:
            self.update(text)

    class GitStatusBar(Static):
        def set_status(self, text: str) -> None:
            self.update(text)

    class HintBar(Static):
        """Contextual hints shown during history navigation."""

    class CompletionBar(Static):
        """Inline completion list shown while the user types a /command."""

        MAX_VISIBLE = 6

        def set_completions(
            self,
            matches: "list[tuple[str, str, bool]]",
            selected_idx: int,
        ) -> None:
            if not matches:
                self.update("")
                self.remove_class("visible")
                return
            lines = []
            for i, (cmd, desc, _) in enumerate(matches[: self.MAX_VISIBLE]):
                marker = "▸" if i == selected_idx else " "
                if i == selected_idx:
                    cmd_part = f"[bold {t.cmd_color}]{cmd}[/bold {t.cmd_color}]"
                else:
                    cmd_part = f"[{t.cmd_color}]{cmd}[/{t.cmd_color}]"
                lines.append(
                    f" {marker} {cmd_part:<20} [{t.text_dim}]{desc}[/{t.text_dim}]"
                )
            if len(matches) > self.MAX_VISIBLE:
                lines.append(
                    f"[{t.text_dim}]   … {len(matches) - self.MAX_VISIBLE} more[/{t.text_dim}]"
                )
            self.update("\n".join(lines))
            self.add_class("visible")

    class PromptInput(TextArea):
        """Multi-line input.

        Keys:
          Enter                                     → submit
          Shift+Enter / Alt+Enter / Ctrl+J          → insert newline
          Up/Down on empty buffer                   → browse history
          ESC                                       → cancel browse / clear completions
        """

        class Submitted(Message):
            def __init__(self, area: "PromptInput", value: str) -> None:
                super().__init__()
                self.area = area
                self.value = value

        class HistorySubmitted(Message):
            """User confirmed editing a past message. remove_count interactions will be rolled back."""

            def __init__(
                self, area: "PromptInput", value: str, remove_count: int
            ) -> None:
                super().__init__()
                self.area = area
                self.value = value
                self.remove_count = remove_count

        class HintChanged(Message):
            def __init__(self, text: str) -> None:
                super().__init__()
                self.text = text

        class CompletionChanged(Message):
            def __init__(
                self,
                matches: "list[tuple[str, str, bool]]",
                selected_idx: int,
            ) -> None:
                super().__init__()
                self.matches = matches
                self.selected_idx = selected_idx

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._history: list[str] = []
            self._mode: str = "normal"  # "normal" | "browsing" | "editing"
            self._history_idx: int | None = None
            self._edit_source_idx: int | None = None
            self._saved_text: str = ""
            self._comp_matches: list[tuple[str, str, bool]] = []
            self._comp_idx: int = -1  # -1 = no item highlighted
            self._comp_suppress_update: bool = False

        def add_to_history(self, text: str) -> None:
            if not self._history or self._history[-1] != text:
                self._history.append(text)

        def _remove_count_for(self, idx: int) -> int:
            """How many interactions are removed when editing history[idx]."""
            return len(self._history) - idx

        def _enter_browsing(self, idx: int) -> None:
            self._mode = "browsing"
            self._history_idx = idx
            self.load_text(self._history[idx])
            self.move_cursor(self.document.end)
            rc = self._remove_count_for(idx)
            rc_str = f"  [dim](removes {rc} interaction{'s' if rc != 1 else ''})[/dim]"
            self.post_message(
                self.HintChanged(
                    f"[bold]↑↓[/bold] navigate  [bold]ENTER[/bold]=edit&retry  [bold]ESC[/bold]=cancel{rc_str}"
                )
            )

        def _exit_browsing(self) -> None:
            self._mode = "normal"
            self._history_idx = None
            self.load_text(self._saved_text)
            self._saved_text = ""
            self.move_cursor(self.document.end)
            self.post_message(self.HintChanged(""))

        def _enter_editing(self) -> None:
            self._mode = "editing"
            self._edit_source_idx = self._history_idx
            rc = self._remove_count_for(self._edit_source_idx)
            rc_str = f"  [dim](removes {rc} interaction{'s' if rc != 1 else ''})[/dim]"
            self.post_message(
                self.HintChanged(
                    f"[bold]ENTER[/bold]=retry  [bold]ESC[/bold]=cancel{rc_str}"
                )
            )

        # ── completion helpers ────────────────────────────────────────────────

        def _post_completion(self) -> None:
            self.post_message(
                PromptInput.CompletionChanged(self._comp_matches, self._comp_idx)
            )

        def _clear_completions(self) -> None:
            self._comp_matches = []
            self._comp_idx = -1
            self._post_completion()

        def _fill_completion(self) -> None:
            """Replace input text with the currently selected (or first) completion."""
            if not self._comp_matches:
                return
            idx = max(self._comp_idx, 0)
            cmd, _, takes_arg = self._comp_matches[idx]
            filled = cmd + (" " if takes_arg else "")
            self._comp_suppress_update = True
            self.load_text(filled)
            self.move_cursor(self.document.end)

        def on_text_area_changed(self, _event) -> None:
            if self._comp_suppress_update:
                self._comp_suppress_update = False
                return
            if self._mode != "normal":
                return
            text = self.text
            if text.startswith("/"):
                cmd_part = text.split()[0] if text.split() else text
                self._comp_matches = _match_commands(cmd_part)
            else:
                self._comp_matches = []
            self._comp_idx = -1
            self._post_completion()

        # ── key handling ─────────────────────────────────────────────────────

        def _on_key(self, event) -> None:
            if self._mode == "normal":
                # ── completion navigation (takes priority) ─────────────────
                if self._comp_matches:
                    if event.key == "tab":
                        event.prevent_default()
                        self._comp_idx = (self._comp_idx + 1) % len(self._comp_matches)
                        self._fill_completion()
                        self._post_completion()
                        return
                    if event.key == "down":
                        event.prevent_default()
                        self._comp_idx = (self._comp_idx + 1) % len(self._comp_matches)
                        self._post_completion()
                        return
                    if event.key == "up":
                        event.prevent_default()
                        self._comp_idx = (self._comp_idx - 1) % len(self._comp_matches)
                        self._post_completion()
                        return
                    if event.key == "escape":
                        event.prevent_default()
                        self._clear_completions()
                        return

                if event.key == "up" and not self.text.strip():
                    event.prevent_default()
                    if self._history:
                        self._saved_text = self.text
                        self._enter_browsing(len(self._history) - 1)
                    return
                if event.key in ("shift+enter", "alt+enter", "ctrl+j", "ctrl+enter"):
                    event.prevent_default()
                    self._clear_completions()
                    self.insert("\n")
                    return
                if event.key == "enter":
                    event.prevent_default()
                    text = self.text.strip()
                    if text:
                        self._clear_completions()
                        self.post_message(PromptInput.Submitted(self, text))
                        self.clear()

            elif self._mode == "browsing":
                event.prevent_default()
                if event.key == "up":
                    if self._history_idx is not None and self._history_idx > 0:
                        self._enter_browsing(self._history_idx - 1)
                elif event.key == "down":
                    if self._history_idx is not None:
                        if self._history_idx < len(self._history) - 1:
                            self._enter_browsing(self._history_idx + 1)
                        else:
                            self._exit_browsing()
                elif event.key == "escape":
                    self._exit_browsing()
                elif event.key == "enter":
                    self._enter_editing()

            elif self._mode == "editing":
                if event.key == "escape":
                    event.prevent_default()
                    self._mode = "browsing"
                    if self._edit_source_idx is not None:
                        self._enter_browsing(self._edit_source_idx)
                elif event.key in ("shift+enter", "alt+enter", "ctrl+j", "ctrl+enter"):
                    event.prevent_default()
                    self.insert("\n")
                elif event.key == "enter":
                    event.prevent_default()
                    text = self.text.strip()
                    if text:
                        rc = self._remove_count_for(self._edit_source_idx)
                        self.post_message(PromptInput.HistorySubmitted(self, text, rc))
                        self._mode = "normal"
                        self._history_idx = None
                        self._edit_source_idx = None
                        self.clear()
                        self.post_message(self.HintChanged(""))

    class ToolCallEvent(Message):
        def __init__(self, name: str, args: str = "") -> None:
            super().__init__()
            self.name = name
            self.args = args

    class ToolResultEvent(Message):
        def __init__(self, name: str, ok: bool) -> None:
            super().__init__()
            self.name = name
            self.ok = ok

    class TokenStreamEvent(Message):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    class IterationProgressEvent(Message):
        def __init__(self, done: int, limit: int) -> None:
            super().__init__()
            self.done = done
            self.limit = limit

    class PhaseEvent(Message):
        def __init__(self, label: str, detail: str = "") -> None:
            super().__init__()
            self.label = label
            self.detail = detail

    class ReasoningTokenEvent(Message):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    class ContextSizeEvent(Message):
        def __init__(self, tokens: int) -> None:
            super().__init__()
            self.tokens = tokens

    _PLACEHOLDER_Q = (
        "[bold]Q — User questions[/bold]\n\n"
        "[dim]Placeholder.[/dim] Will show the conversation rephrased as a single\n"
        "concise statement representing the user's intent or question per turn.\n\n"
        "[dim]Useful for reviewing what was actually asked without re-reading full turns.[/dim]"
    )
    _PLACEHOLDER_A = (
        "[bold]A — Agent answers[/bold]\n\n"
        "[dim]Placeholder.[/dim] Will show agent responses summarized into a single\n"
        "actionable statement or conclusion per turn.\n\n"
        "[dim]Useful for extracting decisions or outcomes from long agentic runs.[/dim]"
    )
    _PLACEHOLDER_SPARSE = (
        "[bold]sparse — Condensed dialogue[/bold]\n\n"
        "[dim]Placeholder.[/dim] Will show the full conversation with each entry\n"
        "shortened to its essential content, preserving the dialogue structure.\n\n"
        "[dim]Useful for skimming long sessions without losing the back-and-forth shape.[/dim]"
    )

    class CodeAgentApp(App):
        CSS = f"""
        Screen {{
            background: {t.bg};
            layout: vertical;
        }}
        #header-bar {{
            height: 1;
            background: {t.panel_bg};
            color: {t.text_dim};
            padding: 0 1;
        }}
        TabbedContent {{
            height: 1fr;
        }}
        ContentSwitcher {{
            height: 1fr;
        }}
        TabPane {{
            padding: 0;
            height: 1fr;
        }}
        #chat-log {{
            height: 1fr;
            border: solid {t.border};
            padding: 0 1;
        }}
        #chat-log:focus {{
            border: solid {t.active};
        }}
        #sys-log {{
            height: 1fr;
            border: solid {t.border};
            padding: 0 1;
        }}
        #sys-log:focus {{
            border: solid {t.active};
        }}
        #q-log, #a-log, #sparse-log {{
            height: 1fr;
            border: solid {t.border};
            padding: 0 1;
        }}
        #q-log:focus, #a-log:focus, #sparse-log:focus {{
            border: solid {t.active};
        }}
        .placeholder-pane {{
            height: 1fr;
            border: solid {t.border};
            padding: 2 4;
            color: {t.text_dim};
        }}
        #context-panel {{
            height: 3;
            background: {t.panel_bg};
            color: {t.text_dim};
            padding: 0 1;
        }}
        #git-status {{
            height: 1;
            background: {t.panel_bg_dark};
            color: {t.text_dim};
            padding: 0 1;
        }}
        #input-bar {{
            height: auto;
            max-height: 8;
            min-height: 3;
            border: solid {t.border};
        }}
        #input-bar:focus {{
            border: solid {t.active};
        }}
        CompletionBar {{
            height: auto;
            max-height: 8;
            display: none;
            background: {t.panel_bg_dark};
            color: {t.text_dim};
            padding: 0 1;
        }}
        CompletionBar.visible {{
            display: block;
        }}
        HintBar {{
            height: 0;
            background: {t.panel_bg};
            color: {t.text_dim};
            padding: 0 1;
        }}
        HintBar.visible {{
            height: 1;
        }}
        TokenBar {{
            height: 1;
            color: {t.text_dim};
        }}
        ContextBreakdownBar {{
            height: 1;
            color: {t.text_dim};
        }}
        OutputBreakdownBar {{
            height: 1;
            color: {t.text_dim};
        }}
        #loading-row {{
            display: none;
            height: 1;
        }}
        #loading-row.active {{
            display: block;
        }}
        LoadingIndicator {{
            width: auto;
            height: 1;
            background: {t.active};
        }}
        #loading-tokens {{
            height: 1;
            width: 1fr;
            background: {t.active};
            color: white;
            padding: 0 1;
        }}
        #stream-view {{
            height: auto;
            max-height: 10;
            display: none;
            background: {t.bg};
            padding: 0 1;
            color: {t.text};
        }}
        #stream-view.active {{
            display: block;
        }}
        """

        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+d", "quit", "Quit", show=False),
            Binding("f1", "show_help", "Help"),
            Binding("ctrl+r", "continue_turn", "Continue"),
            Binding("ctrl+tab", "focus_next", "Switch focus", show=False),
        ]

        def __init__(self, agent: "Agent", session=None, **kwargs):
            super().__init__(**kwargs)
            self._agent = agent
            self._session = session
            self._last_tool_calls: list[str] = []
            self._current_tool: str | None = None
            self._tokens_before: int = 0
            self._streaming_active: bool = False
            self._stream_buffer: str = ""
            self._modified_files: list[str] = []
            self._loading_timer = None
            self._iter_done: int = 0
            self._iter_limit: int = 0
            self._quit_requested: bool = False
            self._chat_user_lines: list[int] = []
            self._sys_messages: list[str] = []

            chat_wrap_cfg = self._agent.config.ui.chat_wrap
            if chat_wrap_cfg == "wrap":
                self._wrap_enabled = True
            elif chat_wrap_cfg == "nowrap":
                self._wrap_enabled = False
            elif chat_wrap_cfg == "last used":
                from agent.ui.prefs import load_prefs

                prefs = load_prefs()
                self._wrap_enabled = prefs.get("chat_wrap") == "wrap"
            else:
                self._wrap_enabled = False

            if session is not None:
                agent.set_session_id(session.id)

        def compose(self) -> ComposeResult:
            cfg = self._agent.config
            session_label = ""
            if self._session:
                label = self._session.short_name or self._session.id
                session_label = f"  [{t.text_dim}]{label}[/{t.text_dim}]"
            yield Static(
                f"[bold]local-code-agent[/bold]  [{t.text_dim}]{cfg.llm.model}[/{t.text_dim}]{session_label}",
                id="header-bar",
            )
            yield TokenBar(
                cfg.llm.ctx_window,
                compact_frac=getattr(cfg.llm, "compaction_threshold", 0.75),
                id="token-bar",
            )
            yield ContextBreakdownBar(cfg.llm.ctx_window, id="context-breakdown")
            yield OutputBreakdownBar(id="output-breakdown")
            with TabbedContent(initial="tab-chat", id="view-tabs"):
                with TabPane("chat", id="tab-chat"):
                    yield ConversationView(id="chat-log", markup=True, highlight=True)
                    yield Static("", id="stream-view", markup=True)
                with TabPane("Q", id="tab-q"):
                    yield QView(id="q-log", markup=True, highlight=False)
                with TabPane("A", id="tab-a"):
                    yield AView(id="a-log", markup=True, highlight=False)
                with TabPane("sparse", id="tab-sparse"):
                    yield SparseView(id="sparse-log", markup=True, highlight=False)
                with TabPane("sys", id="tab-sys"):
                    yield SysView(id="sys-log", markup=True, highlight=True)
            with Horizontal(id="loading-row"):
                yield LoadingIndicator(id="loading-indicator")
                yield Static("", id="loading-tokens", markup=True)
            yield ContextPanel("", id="context-panel")
            yield GitStatusBar("git: loading...", id="git-status")
            yield PromptInput(id="input-bar")
            yield CompletionBar("", id="completion-bar", markup=True)
            yield HintBar("", id="hint-bar", markup=True)
            yield Footer()

        def _refresh_token_bar(self) -> None:
            try:
                bar = self.query_one("#token-bar", TokenBar)
            except Exception:
                return
            bar.update_tokens(
                self._agent.token_estimate(),
                peak=getattr(self._agent, "round_peak_tokens", 0),
                compact_frac=getattr(
                    self._agent.config.llm, "compaction_threshold", 0.75
                ),
            )
            try:
                breakdown = self.query_one("#context-breakdown", ContextBreakdownBar)
                breakdown.set_segments(self._agent.context_breakdown())
            except Exception:
                pass
            try:
                out_bar = self.query_one("#output-breakdown", OutputBreakdownBar)
                out_bar.set_segments(
                    self._agent.output_breakdown("session"),
                    scope_label="out",
                )
            except Exception:
                pass

        def on_mount(self) -> None:
            self.query_one("#input-bar", PromptInput).focus()
            self.call_later(self._refresh_git)
            self.call_later(self._refresh_token_bar)
            # Register a thread-safe progress callback for analyze_asm
            try:
                from agent.tools.analyze_asm import set_ui_progress_cb

                app_ref = self

                def _asm_ui_cb(msg: str) -> None:
                    app_ref.call_from_thread(
                        app_ref.query_one("#context-panel", ContextPanel).set_context,
                        f"[{t.tool_color}]{msg}[/{t.tool_color}]",
                    )

                set_ui_progress_cb(_asm_ui_cb)
            except Exception:
                pass
            # Seed the sys log with session info
            sys_log = self.query_one("#sys-log", SysView)
            if self._session:
                sys_log.write(
                    f"[{t.text_dim}]session  {self._session.id}[/{t.text_dim}]"
                    + (
                        f"  [{t.cmd_color}]{self._session.short_name}[/{t.cmd_color}]"
                        if self._session.short_name
                        else ""
                    )
                )
            sys_log.write(
                f"[{t.text_dim}]Type /help for commands  ·  F1 opens this tab[/{t.text_dim}]"
            )
            # Restore prior dialogue if session was loaded before the UI started
            if self._agent.messages:
                self._restore_chat_history(self._agent.messages)
            self._reload_qa_views()

        # ── helpers ──────────────────────────────────────────────────────────

        def _write_sys(self, text: str) -> None:
            """Write to the sys log and switch to that tab."""
            self._sys_messages.append(text)
            self.query_one("#sys-log", SysView).write(text)
            self.query_one(TabbedContent).active = "tab-sys"

        def _write_chat(self, text: str) -> None:
            self.query_one("#chat-log", ConversationView).write(text)

        def _switch_to_chat(self) -> None:
            self.query_one(TabbedContent).active = "tab-chat"

        def _restore_chat_history(self, messages: list) -> None:
            """Replay session messages into the chat log (clears first)."""
            import json as _json

            chat_log = self.query_one("#chat-log", ConversationView)
            chat_log.clear()
            self._chat_user_lines = []
            for m in messages:
                role = m.get("role", "")
                if role == "system":
                    continue
                content = m.get("content") or ""
                if isinstance(content, list):
                    content = _json.dumps(content)
                tool_calls = m.get("tool_calls") or []
                if role == "user":
                    self._chat_user_lines.append(len(chat_log.lines))
                    chat_log.write(
                        f"[bold {t.user_color}]You:[/bold {t.user_color}] {_escape(_one_line(content, wrap=self._wrap_enabled))}"
                    )
                elif role == "assistant":
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            name = tc.get("function", {}).get("name", "?")
                            chat_log.write(
                                f"[{t.tool_color}]  ⚙ {name}[/{t.tool_color}]"
                            )
                    if content:
                        if self._wrap_enabled:
                            chat_log.write(
                                f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]"
                            )
                            chat_log.write(Markdown(content.strip()))
                        else:
                            chat_log.write(
                                f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {_escape(_one_line(content, wrap=False))}"
                            )

        def _reload_sys_view(self) -> None:
            sys_log = self.query_one("#sys-log", SysView)
            sys_log.clear()
            for msg in self._sys_messages:
                sys_log.write(msg)

        def _reload_qa_views(self) -> None:
            """Populate Q/A/sparse views from on-disk history for the current session."""
            if not self._session:
                return
            try:
                from agent.memory.qa_log import read_history_sync

                entries = read_history_sync(self._session.id)
            except Exception:
                logger.exception("_reload_qa_views: read_history_sync failed (ignored)")
                return
            try:
                self.query_one("#q-log", QView).load_history(entries)
                self.query_one("#a-log", AView).load_history(entries)
                self.query_one("#sparse-log", SparseView).load_history(entries)
            except Exception:
                logger.exception("_reload_qa_views: view update failed (ignored)")

        def _append_qa_turn(self, user_text: str, response: str) -> None:
            """Append the just-completed turn to the Q/A/sparse views."""
            try:
                turn_id = getattr(self._agent, "_turn_id", 0)
                q_data = {"turn_id": turn_id, "content": user_text}
                a_data = {
                    "turn_id": turn_id,
                    "content": response or "",
                    "tool_calls": list(self._last_tool_calls),
                    "modified_files": list(self._modified_files),
                }
                self.query_one("#q-log", QView).add_turn(turn_id, q_data, a_data)
                self.query_one("#a-log", AView).add_turn(turn_id, q_data, a_data)
                self.query_one("#sparse-log", SparseView).add_turn(
                    turn_id, q_data, a_data
                )
            except Exception:
                logger.exception("_append_qa_turn: failed (ignored)")

        # ── actions ──────────────────────────────────────────────────────────

        def action_quit(self) -> None:
            try:
                from agent.tools.analyze_asm import get_interrupt_flag

                get_interrupt_flag().set()
            except Exception:
                pass
            # Second press (while graceful shutdown is already in flight) →
            # cancel pending summary tasks and exit immediately.
            if self._quit_requested:
                try:
                    self._agent.cancel_background()
                except Exception:
                    pass
                self.exit()
                return
            # First press: if nothing is running in the background, exit now;
            # otherwise wait for the post-turn summary before tearing down.
            pending = 0
            try:
                pending = self._agent.pending_background_count()
            except Exception:
                pending = 0
            if pending == 0:
                self.exit()
                return
            self._quit_requested = True
            try:
                self.query_one("#context-panel", ContextPanel).set_context(
                    f"[{t.warning}]Finishing {pending} summary task(s)… "
                    f"press Ctrl+Q again to force exit.[/{t.warning}]"
                )
                self.query_one("#input-bar", PromptInput).disabled = True
            except Exception:
                pass
            self._graceful_exit_worker()

        @work(exclusive=True, name="graceful_exit")
        async def _graceful_exit_worker(self) -> None:
            try:
                await self._agent.wait_background(timeout=30.0)
            except Exception:
                logger.exception("graceful_exit: wait_background error (ignored)")
            self.exit()

        def action_show_help(self) -> None:
            self._write_sys(_make_help_text(t))

        def action_continue_turn(self) -> None:
            input_widget = self.query_one("#input-bar", PromptInput)
            if input_widget.disabled:
                return
            self._begin_chat("continue")

        # ── git refresh ──────────────────────────────────────────────────────

        async def _refresh_git(self) -> None:
            from agent.tools.git import git_status

            try:
                loop = asyncio.get_event_loop()
                s = await loop.run_in_executor(None, git_status)
                branch = s.get("branch", "?")
                staged = len(s.get("staged", []))
                self.query_one("#git-status", GitStatusBar).set_status(
                    f"git: {staged} staged  branch: {branch}"
                )
            except Exception:
                pass

        # ── slash commands → sys tab ─────────────────────────────────────────

        async def _run_slash(self, cmd: str, arg: str) -> None:
            from openai import AsyncOpenAI

            if cmd == "/help":
                self._write_sys(_make_help_text(t))

            elif cmd == "/q":
                self.query_one(TabbedContent).active = "tab-q"

            elif cmd == "/a":
                self.query_one(TabbedContent).active = "tab-a"

            elif cmd == "/sparse":
                self.query_one(TabbedContent).active = "tab-sparse"

            elif cmd == "/compact":
                from agent.memory.compactor import compact

                cfg = self._agent.config
                client = AsyncOpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key)
                before = self._agent.token_estimate()
                self._write_sys(f"[{t.text_dim}]Compacting…[/{t.text_dim}]")
                try:
                    self._agent.messages = await compact(
                        self._agent.messages, cfg, client
                    )
                    after = self._agent.token_estimate()
                    self._write_sys(
                        f"[{t.success}]Compacted.[/{t.success}] {before} → {after} tokens"
                    )
                except Exception as e:
                    self._write_sys(f"[{t.error}]Compact failed: {e}[/{t.error}]")
                self._refresh_token_bar()

            elif cmd in ("/continue", "/c"):
                self._begin_chat("continue")

            elif cmd == "/clear":
                self.query_one("#chat-log", ConversationView).clear()
                self._switch_to_chat()

            elif cmd == "/tokens":
                used = self._agent.token_estimate()
                cfg = self._agent.config
                peak = getattr(self._agent, "round_peak_tokens", 0)
                last_peak = getattr(self._agent, "last_round_peak_tokens", 0)
                self._write_sys(
                    f"tokens: {used}/{cfg.llm.ctx_window}  "
                    f"({len(self._agent.messages)} messages)  "
                    f"peak: {peak}  prev-round peak: {last_peak}"
                )

            elif cmd in ("/context", "/ctx", "/legend"):
                self._write_sys(_render_context_report(self._agent, t))
                self._refresh_token_bar()

            elif cmd in ("/output", "/out"):
                scope = (arg.strip().lower() or "session")
                if scope not in ("session", "last"):
                    self._write_sys(
                        f"[{t.warning}]Usage: /output [session|last][/{t.warning}]"
                    )
                else:
                    breakdown = self._agent.output_breakdown(scope)
                    total = sum(s["tokens"] for s in breakdown)
                    header = "Output breakdown — " + (
                        "cumulative session" if scope == "session" else "last turn"
                    )
                    self._write_sys(f"[bold]{header}[/bold]")
                    for seg in breakdown:
                        tok = seg["tokens"]
                        pct = (tok / total * 100) if total else 0
                        color = _OUT_SEGMENT_COLORS.get(seg["label"], "white")
                        self._write_sys(
                            f"  [{color}]█[/{color}] {seg['label']:<10} "
                            f"{tok:>7,}  ({pct:5.1f}%)"
                        )
                    self._write_sys(f"  {'total':<12} {total:>7,}")
                    try:
                        out_bar = self.query_one("#output-breakdown", OutputBreakdownBar)
                        out_bar.set_segments(
                            breakdown,
                            scope_label="out" if scope == "session" else "turn",
                        )
                    except Exception:
                        pass

            elif cmd == "/reset":
                system = next(
                    (m for m in self._agent.messages if m.get("role") == "system"), None
                )
                self._agent.messages = [system] if system else []
                self._write_sys(
                    f"[{t.text_dim}]Conversation history cleared.[/{t.text_dim}]"
                )

            elif cmd == "/tools":
                from agent.tools import get_schemas

                names = [s["function"]["name"] for s in get_schemas()]
                self._write_sys("Tools: " + "  ".join(names))

            elif cmd == "/save":
                from agent.memory.session import save_session

                save_session(self._session, self._agent.messages)
                label = self._session.short_name or self._session.id
                self._write_sys(
                    f"[{t.text_dim}]Saved session '{label}'.[/{t.text_dim}]"
                )

            elif cmd == "/load":
                if not arg.strip():
                    self._write_sys(
                        f"[{t.warning}]Usage: /load <session-id-or-short-name>[/{t.warning}]"
                    )
                else:
                    from agent.memory.session import load_session

                    loaded_session, loaded_msgs = load_session(arg.strip())
                    if loaded_session is None:
                        self._write_sys(
                            f"[{t.warning}]Session '{arg.strip()}' not found.[/{t.warning}]"
                        )
                    else:
                        loaded_msgs = [
                            {k: v for k, v in m.items() if not k.startswith("_")}
                            for m in loaded_msgs
                        ]
                        self._agent.messages = loaded_msgs
                        self._session = loaded_session
                        label = loaded_session.short_name or loaded_session.id
                        self._write_sys(
                            f"[{t.text_dim}]Loaded session '{label}' "
                            f"({len(loaded_msgs)} messages).[/{t.text_dim}]"
                        )
                        self._refresh_token_bar()
                        self._restore_chat_history(loaded_msgs)
                        self._switch_to_chat()

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime

                sessions = list_sessions()
                if not sessions:
                    self._write_sys(f"[{t.text_dim}]No sessions found.[/{t.text_dim}]")
                for s in sessions:
                    ts_val = s.get("updated_at") or s.get("created_at")
                    ts = (
                        datetime.datetime.fromtimestamp(ts_val).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        if ts_val
                        else "?"
                    )
                    label = s.get("short_name") or s["id"]
                    name_extra = (
                        f"  [{t.text_dim}]{s['name']}[/{t.text_dim}]"
                        if s.get("name")
                        else ""
                    )
                    self._write_sys(
                        f"  [{t.cmd_color}]{label}[/{t.cmd_color}]{name_extra}"
                        f"  {s['message_count']} msgs  [{t.text_dim}]{ts}[/{t.text_dim}]"
                    )

            elif cmd == "/undo":
                from agent.tools.files import undo_file, undo_candidates

                target = arg.strip()
                if not target:
                    candidates = undo_candidates()
                    if not candidates:
                        self._write_sys(f"[{t.warning}]Nothing to undo.[/{t.warning}]")
                    else:
                        self._write_sys("Undo candidates: " + ", ".join(candidates))
                else:
                    r = undo_file(target)
                    if "error" in r:
                        self._write_sys(f"[{t.error}]{r['error']}[/{t.error}]")
                    else:
                        self._write_sys(f"[{t.success}]Restored {target}[/{t.success}]")

            elif cmd == "/exec":
                if not arg.strip():
                    self._write_sys(
                        f"[{t.warning}]Usage: /exec <command>[/{t.warning}]"
                    )
                else:
                    from agent.tools.shell import run_command

                    self._write_sys(f"[{t.text_dim}]$ {arg.strip()}[/{t.text_dim}]")
                    result = run_command(arg.strip())
                    if result.get("stdout"):
                        self._write_sys(result["stdout"].rstrip())
                    if result.get("stderr"):
                        self._write_sys(
                            f"[{t.warning}]{result['stderr'].rstrip()}[/{t.warning}]"
                        )
                    rc = result.get("returncode", 0)
                    if rc != 0:
                        self._write_sys(f"[{t.error}]exit code {rc}[/{t.error}]")
                    elif result.get("error"):
                        self._write_sys(f"[{t.error}]{result['error']}[/{t.error}]")

            elif cmd == "/export":
                import json as _json

                lines = []
                for m in self._agent.messages:
                    role = m.get("role", "?")
                    if role == "system":
                        continue
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = _json.dumps(content)
                    tool_calls = m.get("tool_calls", [])
                    if role == "user":
                        lines.append(f"**You:** {content}\n")
                    elif role == "assistant":
                        if tool_calls:
                            names = ", ".join(
                                tc["function"]["name"]
                                for tc in tool_calls
                                if isinstance(tc, dict)
                            )
                            lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                        else:
                            lines.append(f"**Agent:** {content}\n")
                md_text = "\n---\n".join(lines)
                label = (
                    (self._session.short_name or self._session.id)
                    if self._session
                    else "session"
                )
                target = arg.strip() or f"{label}.md"
                Path(target).write_text(md_text, encoding="utf-8")
                self._write_sys(
                    f"[{t.text_dim}]Exported to {target} ({len(lines)} turns).[/{t.text_dim}]"
                )

            elif cmd == "/analyze-asm":
                await self._run_analyze_asm(arg)

            elif cmd == "/think":
                ok, msg = _apply_think(self._agent, arg)
                color = t.success if ok else t.warning
                for line in msg.splitlines():
                    self._write_sys(f"[{color}]{line}[/{color}]")

            elif cmd in ("/temperature", "/temp"):
                ok, msg = _apply_temperature(self._agent, arg)
                color = t.success if ok else t.warning
                for line in msg.splitlines():
                    self._write_sys(f"[{color}]{line}[/{color}]")

            elif cmd == "/wrap":
                from agent.ui.prefs import save_prefs

                self._wrap_enabled = not self._wrap_enabled
                state = "enabled" if self._wrap_enabled else "disabled"
                save_prefs({"chat_wrap": "wrap" if self._wrap_enabled else "nowrap"})
                self._write_sys(f"[{t.success}]Line wrapping {state}.[/{t.success}]")
                # Refresh views to apply the new wrapping setting
                self.query_one("#chat-log", ConversationView).clear()
                if self._agent.messages:
                    self._restore_chat_history(self._agent.messages)
                self._reload_qa_views()

            else:
                self._write_sys(
                    f"[{t.warning}]Unknown command '{cmd}'. Type /help.[/{t.warning}]"
                )

        async def _run_analyze_asm(self, arg: str) -> None:
            from agent.tools.analyze_asm import analyze_asm, get_interrupt_flag

            parts = arg.split()
            if not parts:
                self._write_sys(
                    f"[{t.warning}]Usage: /analyze-asm <file> [--resume] [--force] [--levels N][/{t.warning}]"
                )
                return
            path = parts[0]
            resume = "--resume" in parts
            force = "--force" in parts
            max_levels = None
            if "--levels" in parts:
                idx = parts.index("--levels")
                if idx + 1 < len(parts):
                    try:
                        max_levels = int(parts[idx + 1])
                    except ValueError:
                        pass

            interrupt = get_interrupt_flag()
            interrupt.clear()
            self._write_sys(
                f"[{t.text_dim}]Analyzing {path}…  ESC to interrupt[/{t.text_dim}]"
            )

            def _do_analyze():
                kwargs = {"path": path, "resume": resume, "force": force}
                if max_levels is not None:
                    kwargs["max_levels"] = max_levels
                return analyze_asm(**kwargs)

            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, _do_analyze)
            except Exception as e:
                self._write_sys(f"[{t.error}]analyze-asm error: {e}[/{t.error}]")
                return

            if "error" in result:
                self._write_sys(f"[{t.error}]{result['error']}[/{t.error}]")
            else:
                self._write_sys(
                    f"[{t.success}]{result.get('message', str(result))}[/{t.success}]"
                )

        # ── input handling ───────────────────────────────────────────────────

        def on_prompt_input_completion_changed(
            self, event: PromptInput.CompletionChanged
        ) -> None:
            self.query_one("#completion-bar", CompletionBar).set_completions(
                event.matches, event.selected_idx
            )

        def on_prompt_input_hint_changed(self, event: PromptInput.HintChanged) -> None:
            hint_bar = self.query_one("#hint-bar", HintBar)
            if event.text:
                hint_bar.update(event.text)
                hint_bar.add_class("visible")
            else:
                hint_bar.update("")
                hint_bar.remove_class("visible")

        async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
            user_text = event.value.strip()
            if not user_text:
                return

            if user_text.startswith("/"):
                parts = user_text.split(None, 1)
                await self._run_slash(
                    parts[0].lower(), parts[1] if len(parts) > 1 else ""
                )
                return

            if user_text.lower() == "continue":
                await self._run_slash("/continue", "")
                return

            input_widget = self.query_one("#input-bar", PromptInput)
            input_widget.add_to_history(user_text)
            self._begin_chat(user_text)

        async def on_prompt_input_history_submitted(
            self, event: PromptInput.HistorySubmitted
        ) -> None:
            user_text = event.value.strip()
            if not user_text:
                return

            # Roll back agent messages: remove the last remove_count user turns
            if event.remove_count > 0:
                user_positions = [
                    i
                    for i, m in enumerate(self._agent.messages)
                    if m.get("role") == "user"
                ]
                if event.remove_count >= len(user_positions):
                    system = next(
                        (m for m in self._agent.messages if m.get("role") == "system"),
                        None,
                    )
                    self._agent.messages = [system] if system else []
                else:
                    cut = user_positions[-event.remove_count]
                    self._agent.messages = self._agent.messages[:cut]
                self._refresh_token_bar()

            # Truncate history to the edit point and add the new version
            input_widget = self.query_one("#input-bar", PromptInput)
            if event.area._edit_source_idx is not None:
                input_widget._history = input_widget._history[
                    : event.area._edit_source_idx
                ]
            input_widget.add_to_history(user_text)

            self._begin_chat(user_text)

        def _update_loading_tokens(self) -> None:
            """Periodic callback to refresh token counts on the loading bar."""
            try:
                est = self._agent.token_estimate()
                rcvd = max(0, est - self._tokens_before)
                text = f"in: [bold]{self._tokens_before:,}[/bold]  out: +{rcvd:,}"
                if self._iter_limit:
                    left = max(0, self._iter_limit - self._iter_done)
                    text += f"  iter {self._iter_done}/{self._iter_limit} ({left} left)"
                self.query_one("#loading-tokens", Static).update(text)
            except Exception:
                pass

        def _begin_chat(self, user_text: str) -> None:
            # Switch to chat tab so the user sees the exchange.
            self._switch_to_chat()
            chat_log = self.query_one("#chat-log", ConversationView)
            self._chat_user_lines.append(len(chat_log.lines))
            self._write_chat(
                f"[bold {t.user_color}]You:[/bold {t.user_color}] {_escape(user_text)}"
            )
            self._last_tool_calls = []
            self._tool_stats: dict[str, dict[str, int]] = {}
            self._current_tool = None
            self._tokens_before = self._agent.token_estimate()
            self._streaming_active = False
            self._stream_buffer = ""
            self._modified_files = []
            self._iter_done = 0
            self._iter_limit = 0
            self._current_user_text = user_text

            self.query_one("#input-bar", PromptInput).disabled = True
            self.query_one("#loading-row").add_class("active")
            self.query_one("#loading-tokens", Static).update(
                f"in: [bold]{self._tokens_before:,}[/bold]"
            )
            if self._loading_timer is not None:
                self._loading_timer.stop()
            self._loading_timer = self.set_interval(0.3, self._update_loading_tokens)
            self.query_one("#context-panel", ContextPanel).set_context(
                "[dim]thinking…[/dim]"
            )

            self._start_chat(user_text)

        @work(exclusive=True, exit_on_error=False, name="chat")
        async def _start_chat(self, user_text: str) -> str:
            from agent.memory.session import save_session

            def on_tool(name: str, args: str) -> None:
                self._last_tool_calls.append(name)
                self._current_tool = name
                self._tool_stats.setdefault(name, {"ok": 0, "err": 0})
                self.post_message(ToolCallEvent(name, args))

            def on_tool_result(name: str, ok: bool) -> None:
                self.post_message(ToolResultEvent(name, ok))

            def on_user_message() -> None:
                if self._session is not None:
                    save_session(self._session, self._agent.messages)

            def on_token(token: str) -> None:
                self.post_message(TokenStreamEvent(token))

            def on_progress(done: int, limit: int) -> None:
                self.post_message(IterationProgressEvent(done, limit))

            def on_phase(label: str, detail: str = "") -> None:
                self.post_message(PhaseEvent(label, detail))

            def on_reasoning(tok: str) -> None:
                self.post_message(ReasoningTokenEvent(tok))

            def on_context_size(n: int) -> None:
                self.post_message(ContextSizeEvent(n))

            result = await self._agent.chat(
                user_text,
                on_tool_call=on_tool,
                on_tool_result=on_tool_result,
                on_user_message=on_user_message,
                on_token=on_token,
                on_progress=on_progress,
                on_phase=on_phase,
                on_reasoning=on_reasoning,
                on_context_size=on_context_size,
            )
            if self._session is not None:
                save_session(self._session, self._agent.messages)
            return result

        def on_jump_to_turn(self, event: JumpToTurn) -> None:
            anchors = self._chat_user_lines
            if not (0 <= event.ordinal < len(anchors)):
                return
            self._switch_to_chat()
            chat_log = self.query_one("#chat-log", ConversationView)
            y = anchors[event.ordinal]
            try:
                chat_log.scroll_to(y=y, animate=True)
            except Exception:
                logger.exception("jump_to_turn: scroll failed (ignored)")

        def on_tool_call_event(self, event: ToolCallEvent) -> None:
            import json

            # Track modified files for write_file / patch_file / edit_file
            if event.name in ("write_file", "patch_file", "edit_file"):
                try:
                    args = (
                        json.loads(event.args)
                        if isinstance(event.args, str)
                        else event.args
                    )
                    if event.name == "edit_file":
                        for ch in args.get("chunks") or []:
                            p = ch.get("path", "") if isinstance(ch, dict) else ""
                            if p and p not in self._modified_files:
                                self._modified_files.append(p)
                    else:
                        path = args.get("path", "")
                        if path and path not in self._modified_files:
                            self._modified_files.append(path)
                except Exception:
                    pass
            # Update context panel with current tool (visible while working)
            preview = ""
            try:
                args = (
                    json.loads(event.args)
                    if isinstance(event.args, str) and event.args
                    else (event.args or {})
                )
                if isinstance(args, dict):
                    def _pval(v: object) -> str:
                        if isinstance(v, str):
                            return repr(v[:35])
                        if isinstance(v, (int, float, bool)):
                            return repr(v)
                        if isinstance(v, (list, tuple)):
                            return f"({len(v)} items)"
                        if isinstance(v, dict):
                            return f"({len(v)} keys)"
                        return type(v).__name__
                    preview = ", ".join(
                        f"{k}={_pval(v)}" for k, v in list(args.items())[:2]
                    )
            except Exception:
                pass
            label = f"[{t.tool_color}]⚙ {_escape(event.name)}[/{t.tool_color}]"
            if preview:
                label += f" [dim]({_escape(preview)})[/dim]"
            self.query_one("#context-panel", ContextPanel).set_context(label)

        def on_tool_result_event(self, event: ToolResultEvent) -> None:
            stats = self._tool_stats.setdefault(event.name, {"ok": 0, "err": 0})
            if event.ok:
                stats["ok"] += 1
            else:
                stats["err"] += 1

        def _render_tool_summary(self) -> str:
            if not self._last_tool_calls:
                return ""
            seen = list(dict.fromkeys(self._last_tool_calls))
            parts = []
            for name in seen:
                s = self._tool_stats.get(name, {"ok": 0, "err": 0})
                counts = []
                if s["ok"]:
                    counts.append(f"[{t.success}]{s['ok']}[/{t.success}]")
                if s["err"]:
                    counts.append(f"[{t.error}]{s['err']}[/{t.error}]")
                suffix = f" {' '.join(counts)}" if counts else ""
                parts.append(f"{_escape(name)}{suffix}")
            return f"[{t.tool_color}]⚙[/{t.tool_color}] " + ", ".join(parts)

        def on_iteration_progress_event(self, event: IterationProgressEvent) -> None:
            self._iter_done = event.done
            self._iter_limit = event.limit
            self._update_loading_tokens()

        def on_context_size_event(self, event: ContextSizeEvent) -> None:
            self._refresh_token_bar()

        def on_phase_event(self, event: PhaseEvent) -> None:
            detail = f": {_escape(event.detail)}" if event.detail else ""
            self.query_one("#context-panel", ContextPanel).set_context(
                f"[dim]• {_escape(event.label)}{detail}[/dim]"
            )

        def on_reasoning_token_event(self, event: ReasoningTokenEvent) -> None:
            from rich.text import Text

            self._stream_buffer += event.token
            stream_view = self.query_one("#stream-view", Static)
            if not self._streaming_active:
                self._streaming_active = True
                stream_view.add_class("active")
            tail = (
                self._stream_buffer[-800:]
                if len(self._stream_buffer) > 800
                else self._stream_buffer
            )
            content = Text.assemble(
                ("thinking:", "dim italic"),
                (f" {tail}▌", "dim italic"),
            )
            stream_view.update(content)

        def on_token_stream_event(self, event: TokenStreamEvent) -> None:
            from rich.text import Text

            self._stream_buffer += event.token
            stream_view = self.query_one("#stream-view", Static)
            if not self._streaming_active:
                self._streaming_active = True
                stream_view.add_class("active")
            # Show tail of accumulated text to avoid unbounded growth in the widget
            tail = (
                self._stream_buffer[-800:]
                if len(self._stream_buffer) > 800
                else self._stream_buffer
            )
            content = Text.assemble(
                ("Agent:", f"bold {t.agent_color}"),
                (f" {tail}▌",),
            )
            stream_view.update(content)

        def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
            if event.worker.name != "chat":
                return
            if event.state not in (
                WorkerState.SUCCESS,
                WorkerState.ERROR,
                WorkerState.CANCELLED,
            ):
                return

            # Stop timer and hide loading row
            if self._loading_timer is not None:
                self._loading_timer.stop()
                self._loading_timer = None
            self.query_one("#loading-row").remove_class("active")

            input_widget = self.query_one("#input-bar", PromptInput)
            input_widget.disabled = False
            input_widget.focus()

            empty_response = False
            if event.state == WorkerState.SUCCESS:
                response = event.worker.result
                if not response:
                    logger.warning("chat worker returned empty response")
                    response = "(done)"
                    empty_response = True
            elif event.state == WorkerState.ERROR:
                err = event.worker.error
                tb = (
                    "".join(
                        traceback.format_exception(type(err), err, err.__traceback__)
                    )
                    if err
                    else ""
                )
                logger.error("chat worker error: %s\n%s", err, tb)
                response = f"[{t.error}]Error: {err}[/{t.error}]"
            else:
                logger.warning("chat worker cancelled")
                response = None

            tokens_after = self._agent.token_estimate()
            delta = tokens_after - self._tokens_before
            tools_line = self._render_tool_summary()
            token_line = (
                f"[{t.text_dim}]sent ≈{self._tokens_before:,}  "
                f"[{t.active}]+{delta:,}[/{t.active}] new  "
                f"total {tokens_after:,}[/{t.text_dim}]"
            )
            s = getattr(self._agent, "stats", None)
            if s and s.get("calls", 0) > 0:
                extras = [f"↑{s['input_tokens']:,}", f"↓{s['output_tokens']:,}"]
                if s.get("in_tps"):
                    extras.append(f"{s['in_tps']:.0f} in-tok/s")
                if s.get("out_tps"):
                    extras.append(f"{s['out_tps']:.1f} out-tok/s")
                if s.get("reasoning_tokens"):
                    extras.append(f"think {s['reasoning_tokens']:,}")
                if s.get("tool_tokens"):
                    extras.append(f"tool {s['tool_tokens']:,}")
                token_line += f"\n[{t.text_dim}]{'  '.join(extras)}[/{t.text_dim}]"
            self.query_one("#context-panel", ContextPanel).set_context(
                f"{tools_line}\n{token_line}" if tools_line else token_line
            )

            # Clear streaming widget before writing final response
            if self._streaming_active:
                stream_view = self.query_one("#stream-view", Static)
                stream_view.remove_class("active")
                stream_view.update("")
                self._streaming_active = False
                self._stream_buffer = ""

            # Write folded tool-call summary (collapsed single line)
            if self._last_tool_calls:
                tool_part = self._render_tool_summary()
                if self._modified_files:
                    files_part = f"[{t.success}]{_escape('  '.join(self._modified_files))}[/{t.success}]"
                    self._write_chat(f"  {tool_part}  ·  {files_part}")
                else:
                    self._write_chat(f"  {tool_part}")
            elif self._modified_files:
                files_part = f"[{t.success}]{_escape('  '.join(self._modified_files))}[/{t.success}]"
                self._write_chat(f"  {files_part}")

            if response:
                if empty_response:
                    body = _escape(response)
                    self._write_chat(
                        f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] [{t.text_dim}]{body}[/{t.text_dim}]"
                    )
                else:
                    self._write_chat(
                        f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]"
                    )
                    self._write_chat(Markdown(response))

            if event.state == WorkerState.SUCCESS:
                self._append_qa_turn(
                    getattr(self, "_current_user_text", ""), response or ""
                )

            self._refresh_token_bar()
            self.call_later(self._refresh_git)

    return CodeAgentApp(agent, session=session)


# ── Simple (Claude Code-style) ───────────────────────────────────────────────


def _make_help_text(theme: "ThemeConfig") -> str:  # type: ignore[name-defined]
    c = theme.cmd_color
    return f"""
[bold]Slash commands[/bold]

  [{c}]/help[/{c}]               show this message
  [{c}]/compact[/{c}]            summarise old messages to free context space
  [{c}]/continue[/{c}] (or [{c}]continue[/{c}], Ctrl+R)  resume after iteration cap / truncation
  [{c}]/tokens[/{c}]             show token usage breakdown
  [{c}]/clear[/{c}]              clear the screen
  [{c}]/reset[/{c}]              drop conversation history (keep system prompt)
  [{c}]/save [name][/{c}]        save session under a name (default: current)
  [{c}]/load <name>[/{c}]        load a saved session into the current conversation
  [{c}]/sessions[/{c}]           list saved sessions
  [{c}]/tools[/{c}]              list available tools
  [{c}]/exec <command>[/{c}]      run an OS command and show output
  [{c}]/apply [file][/{c}]       write last code block to file (bypass tool calling)
  [{c}]/undo [file][/{c}]        restore last pre-write snapshot of a file
  [{c}]/export [file][/{c}]      export conversation as markdown
  [{c}]/q[/{c}] · [{c}]/a[/{c}] · [{c}]/sparse[/{c}]    switch to Q / A / sparse tab  (click a line → jump to turn)
  [{c}]/analyze-asm <file>[/{c}]  LLM-driven assembly analysis and summarization
                       options: --resume  --force  --levels N
  [{c}]/think [level][/{c}]       thinking effort: off|low|normal|high|max ('-' resets)
  [{c}]/temperature [v][/{c}]     sampling temperature 0.0–2.0 (alias [{c}]/temp[/{c}]; '-' resets)
  [{c}]/max_tokens [args][/{c}]   set tokens: <n> | out <n> | in <n> | default
  [{c}]/context[/{c}] ([{c}]/ctx[/{c}], [{c}]/legend[/{c}])  context breakdown grid + color/marker key

[dim]Ctrl+D or Ctrl+C to quit[/dim]
"""


def _token_bar(used: int, ctx: int, bar_len: int = 20) -> str:
    pct = used / ctx if ctx else 0
    filled = int(pct * bar_len)
    color = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
    bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"
    return f"[dim]tokens {used}/{ctx}[/dim] {bar}"


def _spinner_status_fields(agent, status: str, elapsed: float) -> list[str]:
    """Return status fields in priority order (most meaningful first).

    Priority order (can be reordered by user preference in future):
      1. ctx%    — context fill % — most actionable; warns when near limit
      2. tokens  — used/total tokens — detail behind ctx%
      3. msgs    — conversation depth (message count)
      4. status  — current operation text (thinking / tool name)
      5. files   — number of indexed files (if RAG store present)
      6. chunks  — number of indexed chunks (if RAG store present)
      7. model   — model name (useful when switching models)
      8. time    — elapsed seconds for current operation
    """
    fields: list[str] = []

    if agent is not None:
        cfg = agent.config
        ctx = cfg.llm.ctx_window or 0
        used = agent.token_estimate()

        if ctx:
            pct = int(used / ctx * 100)
            fields.append(f"ctx {pct}%")
            k_used = f"{used / 1000:.1f}k" if used >= 1000 else str(used)
            k_ctx = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
            fields.append(f"{k_used}/{k_ctx}")

        msg_count = max(0, len(agent.messages) - 1)  # exclude system prompt
        fields.append(f"{msg_count} msg")

        # Usage stats: cumulative I/O counts, per-second rates, and breakdown
        # of output tokens into think/tool. Populated via Agent._record_usage.
        s = getattr(agent, "stats", None)
        if s and s.get("calls", 0) > 0:

            def _k(n: int) -> str:
                return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

            fields.append(f"↑{_k(s['input_tokens'])}")
            fields.append(f"↓{_k(s['output_tokens'])}")
            if s.get("in_tps"):
                fields.append(f"{s['in_tps']:.0f}↑t/s")
            if s.get("out_tps"):
                fields.append(f"{s['out_tps']:.1f}↓t/s")
            if s.get("reasoning_tokens"):
                fields.append(f"think {_k(s['reasoning_tokens'])}")
            if s.get("tool_tokens"):
                fields.append(f"tool {_k(s['tool_tokens'])}")

    fields.append(status)

    if agent is not None and agent.store is not None:
        try:
            s = agent.store.stats()
            fields.append(f"{s['files']} files")
            fields.append(f"{s['chunks']} chunks")
        except Exception:
            pass

    if agent is not None:
        model = agent.config.llm.model or ""
        if model:
            fields.append(model)

    fields.append(f"{elapsed:.1f}s")

    return fields


async def _run_spinner(status_ref: list[str], stop: asyncio.Event, agent=None) -> None:
    import sys
    import shutil
    import time as _time

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    # Animation prefix is at most 2 chars ("⠋ "), leaving the rest for status fields.
    # Max animation width = 5 chars; we use 2, giving 3 chars of padding for future use.
    ANIM_WIDTH = 2  # frame char + space
    SEP = "  "  # separator between fields
    i = 0
    t0 = _time.monotonic()
    while not stop.is_set():
        frame = frames[i % len(frames)]
        elapsed = _time.monotonic() - t0
        term_width = shutil.get_terminal_size((80, 24)).columns
        available = term_width - ANIM_WIDTH

        fields = _spinner_status_fields(agent, status_ref[0], elapsed)

        # Greedily fit as many fields as possible from highest priority
        parts: list[str] = []
        remaining = available
        for field in fields:
            needed = len(field) + (len(SEP) if parts else 0)
            if needed <= remaining:
                parts.append(field)
                remaining -= needed
            # Always include at least the first field (status), even if truncated
            elif not parts:
                parts.append(field[:available])
                break

        info = SEP.join(parts)
        sys.stdout.write(f"\r\033[2m{frame} {info}\033[0m")
        sys.stdout.flush()
        i += 1
        await asyncio.sleep(0.08)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _hex_to_ansi(hex_color: str) -> str:
    """Convert #RRGGBB to an ANSI 24-bit foreground escape sequence."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


async def simple_loop(agent: "Agent", session=None):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.markup import escape as _escape
    import readline  # enables arrow keys / history on Linux

    cfg = agent.config
    t = cfg.ui.theme
    console = Console()

    if session is not None:
        agent.set_session_id(session.id)

    prompt_esc = _hex_to_ansi(t.prompt)
    console.print(
        f"[bold {t.agent_color}]local-code-agent[/bold {t.agent_color}]  [dim]{cfg.llm.model}  {cfg.llm.ctx_window} ctx[/dim]"
    )
    console.print(
        f"[{t.text_dim}]/help /compact /tokens /reset /tools /exec /apply /save /sessions  ·  Ctrl+D to quit[/{t.text_dim}]\n"
    )

    while True:
        try:
            user_input = input(f"{prompt_esc}>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            pending = agent.pending_background_count()
            if pending:
                console.print(
                    f"\n[{t.warning}]Finishing {pending} summary task(s)… "
                    f"Ctrl+C again to force exit.[/{t.warning}]"
                )
                try:
                    await agent.wait_background(timeout=30.0)
                except KeyboardInterrupt:
                    agent.cancel_background()
            console.print(f"[{t.text_dim}]Bye.[/{t.text_dim}]")
            break

        if not user_input:
            continue

        # ── Slash commands ──────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                console.print(_make_help_text(t))

            elif cmd == "/tokens":
                used = agent.token_estimate()
                console.print(_token_bar(used, cfg.llm.ctx_window))
                console.print(f"  [dim]{len(agent.messages)} messages in context[/dim]")

            elif cmd == "/compact":
                from openai import AsyncOpenAI
                from agent.memory.compactor import compact

                client = AsyncOpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key)
                before = agent.token_estimate()
                console.print("[dim]Compacting…[/dim]")
                try:
                    agent.messages = await compact(agent.messages, cfg, client)
                    after = agent.token_estimate()
                    console.print(
                        f"[green]Compacted.[/green] {before} → {after} tokens  "
                        f"({len(agent.messages)} messages)"
                    )
                except Exception as e:
                    console.print(f"[red]Compact failed: {e}[/red]")

            elif cmd == "/clear":
                console.clear()

            elif cmd == "/reset":
                system = next(
                    (m for m in agent.messages if m.get("role") == "system"), None
                )
                agent.messages = [system] if system else []
                console.print("[dim]Conversation history cleared.[/dim]")

            elif cmd == "/save":
                from agent.memory.session import save_session

                if session is not None:
                    save_session(session, agent.messages)
                    label = session.short_name or session.id
                    console.print(f"[dim]Saved session '{label}'.[/dim]")
                else:
                    console.print("[yellow]No active session.[/yellow]")

            elif cmd == "/load":
                if not arg.strip():
                    console.print(
                        "[yellow]Usage: /load <session-id-or-short-name>[/yellow]"
                    )
                else:
                    from agent.memory.session import load_session

                    loaded_session, loaded_msgs = load_session(arg.strip())
                    if loaded_session is None:
                        console.print(
                            f"[yellow]Session '{arg.strip()}' not found.[/yellow]"
                        )
                    else:
                        loaded_msgs = [
                            {k: v for k, v in m.items() if not k.startswith("_")}
                            for m in loaded_msgs
                        ]
                        agent.messages = loaded_msgs
                        session = loaded_session
                        label = session.short_name or session.id
                        console.print(
                            f"[dim]Loaded session '{label}' ({len(loaded_msgs)} messages).[/dim]"
                        )

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime

                for s in list_sessions():
                    ts_val = s.get("updated_at") or s.get("created_at")
                    ts = (
                        datetime.datetime.fromtimestamp(ts_val).strftime(
                            "%Y-%m-%d %H:%M"
                        )
                        if ts_val
                        else "?"
                    )
                    label = s.get("short_name") or s["id"]
                    name_part = (
                        f"  [dim]{s.get('name', '')}[/dim]" if s.get("name") else ""
                    )
                    console.print(
                        f"  [cyan]{label}[/cyan]{name_part}  {s['message_count']} msgs  [dim]{ts}[/dim]"
                    )

            elif cmd == "/tools":
                from agent.tools import get_schemas

                for schema in get_schemas():
                    fn = schema["function"]
                    console.print(
                        f"  [cyan]{fn['name']}[/cyan]  [dim]{fn.get('description', '')[:60]}[/dim]"
                    )

            elif cmd == "/apply":
                from agent.agent import extract_last_code_block
                from agent.tools.files import write_file

                result = extract_last_code_block(agent.messages)
                if not result:
                    console.print(
                        "[yellow]No code block found in recent messages.[/yellow]"
                    )
                else:
                    fname, code = result
                    target = arg.strip() or fname
                    console.print(f"[dim]Writing to {target}:[/dim]")
                    console.print(
                        f"[dim]{code[:200]}{'…' if len(code) > 200 else ''}[/dim]"
                    )
                    confirm = input("Apply? [Y/n]: ").strip().lower()
                    if confirm in ("", "y", "yes"):
                        r = write_file(target, code)
                        if "error" in r:
                            console.print(f"[red]{r['error']}[/red]")
                        else:
                            console.print(f"[green]Written to {target}[/green]")

            elif cmd == "/undo":
                from agent.tools.files import undo_file, undo_candidates

                target = arg.strip()
                if not target:
                    candidates = undo_candidates()
                    if not candidates:
                        console.print("[yellow]Nothing to undo.[/yellow]")
                    else:
                        console.print("Undo candidates: " + ", ".join(candidates))
                        console.print("[dim]Usage: /undo <file>[/dim]")
                else:
                    r = undo_file(target)
                    if "error" in r:
                        console.print(f"[red]{r['error']}[/red]")
                    else:
                        console.print(f"[green]Restored {target}[/green]")

            elif cmd == "/exec":
                if not arg.strip():
                    console.print("[yellow]Usage: /exec <command>[/yellow]")
                else:
                    from agent.tools.shell import run_command

                    console.print(f"[dim]$ {arg.strip()}[/dim]")
                    result = run_command(arg.strip())
                    if result.get("stdout"):
                        console.print(result["stdout"].rstrip())
                    if result.get("stderr"):
                        console.print(f"[yellow]{result['stderr'].rstrip()}[/yellow]")
                    rc = result.get("returncode", 0)
                    if rc != 0:
                        console.print(f"[red]exit code {rc}[/red]")
                    elif result.get("error"):
                        console.print(f"[red]{result['error']}[/red]")

            elif cmd == "/export":
                import json as _json

                lines = []
                for m in agent.messages:
                    role = m.get("role", "?")
                    if role == "system":
                        continue
                    content = m.get("content") or ""
                    if isinstance(content, list):
                        content = _json.dumps(content)
                    tool_calls = m.get("tool_calls", [])
                    if role == "user":
                        lines.append(f"**You:** {content}\n")
                    elif role == "assistant":
                        if tool_calls:
                            names = ", ".join(
                                tc["function"]["name"]
                                for tc in tool_calls
                                if isinstance(tc, dict)
                            )
                            lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                        else:
                            lines.append(f"**Agent:** {content}\n")
                    # skip tool result messages (role == "tool")
                md_text = "\n---\n".join(lines)
                _session_label = (
                    (session.short_name or session.id) if session else "session"
                )
                target = arg.strip() or f"{_session_label}.md"
                Path(target).write_text(md_text, encoding="utf-8")
                console.print(f"[dim]Exported to {target} ({len(lines)} turns).[/dim]")

            elif cmd == "/analyze-asm":
                from agent.tools.analyze_asm import analyze_asm, get_interrupt_flag

                parts = arg.split()
                if not parts:
                    console.print(
                        "[yellow]Usage: /analyze-asm <file> [--resume] [--force] [--levels N][/yellow]"
                    )
                else:
                    path_arg = parts[0]
                    resume = "--resume" in parts
                    force_flag = "--force" in parts
                    max_lvls = None
                    if "--levels" in parts:
                        idx = parts.index("--levels")
                        if idx + 1 < len(parts):
                            try:
                                max_lvls = int(parts[idx + 1])
                            except ValueError:
                                pass
                    interrupt = get_interrupt_flag()
                    interrupt.clear()
                    console.print(
                        f"[dim]Analyzing {path_arg}… (Ctrl+C to interrupt)[/dim]"
                    )
                    try:
                        kwargs = {
                            "path": path_arg,
                            "resume": resume,
                            "force": force_flag,
                        }
                        if max_lvls is not None:
                            kwargs["max_levels"] = max_lvls
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: analyze_asm(**kwargs)
                        )
                        if "error" in result:
                            console.print(f"[red]{result['error']}[/red]")
                        else:
                            console.print(
                                f"[green]{result.get('message', str(result))}[/green]"
                            )
                    except KeyboardInterrupt:
                        interrupt.set()
                        console.print("[yellow]Interrupted.[/yellow]")
                    except Exception as e:
                        console.print(f"[red]analyze-asm error: {e}[/red]")

            elif cmd == "/think":
                ok, msg = _apply_think(agent, arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd in ("/temperature", "/temp"):
                ok, msg = _apply_temperature(agent, arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd == "/max_tokens":
                ok, msg = _apply_max_tokens(agent, arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd in ("/context", "/ctx", "/legend"):
                console.print(_render_context_report(agent, t))

            else:
                console.print(
                    f"[yellow]Unknown command '{cmd}'. Type /help for a list.[/yellow]"
                )

            continue

        # ── Normal message ──────────────────────────────────────────────────
        import sys

        import os as _os
        import json as _json

        verbose = _os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")

        tool_results: list[str] = []
        streaming_tokens: list[str] = []
        reasoning_active: list[bool] = [False]
        _spinner_status: list[str] = ["thinking…"]
        _spinner_stop = asyncio.Event()
        _spinner_task = asyncio.create_task(
            _run_spinner(_spinner_status, _spinner_stop, agent=agent)
        )

        def _clear_spinner() -> None:
            _spinner_stop.set()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        def _args_preview(args_str: str) -> str:
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                if not isinstance(args, dict):
                    return ""
                return ", ".join(
                    f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2]
                )
            except Exception:
                return ""

        def on_tool(name: str, args_str: str) -> None:
            _clear_spinner()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            if reasoning_active[0]:
                console.print()
                reasoning_active[0] = False
            preview = _args_preview(args_str)
            suffix = f"[{t.text_dim}]({_escape(preview)})[/{t.text_dim}]" if preview else ""
            console.print(f"  [{t.tool_color}]⚙ {_escape(name)}[/{t.tool_color}] {suffix}")
            tool_results.append(name)

        def on_tool_result(name: str, ok: bool) -> None:
            mark = f"[{t.success}]✓[/{t.success}]" if ok else f"[{t.error}]✗[/{t.error}]"
            console.print(f"    {mark} [{t.text_dim}]{name}[/{t.text_dim}]")

        def on_progress(done: int, limit: int) -> None:
            _spinner_status[0] = f"iter {done}/{limit}…"

        def on_phase(label: str, detail: str = "") -> None:
            _clear_spinner()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            if reasoning_active[0]:
                console.print()
                reasoning_active[0] = False
            msg = f"  [{t.text_dim}]• {label}"
            if detail:
                msg += f": {detail}"
            msg += f"[/{t.text_dim}]"
            console.print(msg)
            _spinner_status[0] = f"{label}…"

        def on_reasoning(tok: str) -> None:
            if not verbose:
                return
            if not reasoning_active[0]:
                _clear_spinner()
                if streaming_tokens:
                    console.print()
                    streaming_tokens.clear()
                sys.stdout.write(f"\033[2m  ◦ ")
                reasoning_active[0] = True
            sys.stdout.write(tok)
            sys.stdout.flush()

        def on_token(token: str) -> None:
            if reasoning_active[0]:
                sys.stdout.write("\033[0m\n")
                sys.stdout.flush()
                reasoning_active[0] = False
            if not streaming_tokens:
                # First token — stop spinner and clear its line
                _spinner_stop.set()
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            streaming_tokens.append(token)
            console.print(token, end="", highlight=False)

        def _on_user_message() -> None:
            if session is not None:
                from agent.memory.session import save_session

                save_session(session, agent.messages)

        async def _on_loop_detected(summary: str, count: int) -> bool:
            # Pause spinner output, ask user, resume.
            _spinner_stop.set()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            console.print(
                f"[{t.warning}]⚠ loop guard: repeated tool calls ({summary}).[/{t.warning}]"
            )
            loop = asyncio.get_running_loop()
            try:
                answer = await loop.run_in_executor(
                    None, lambda: input("  Continue anyway? [y/N] ").strip().lower()
                )
            except (EOFError, KeyboardInterrupt):
                answer = ""
            return answer in ("y", "yes")

        try:
            response = await agent.chat(
                user_input,
                on_tool_call=on_tool,
                on_tool_result=on_tool_result,
                on_token=on_token,
                on_user_message=_on_user_message,
                on_loop_detected=_on_loop_detected,
                on_progress=on_progress,
                on_phase=on_phase,
                on_reasoning=on_reasoning,
            )
        except Exception as e:
            logger.error("chat error: %s\n%s", e, traceback.format_exc())
            console.print(f"[{t.error}]Error: {e}[/{t.error}]")
            continue
        finally:
            _spinner_stop.set()
            await _spinner_task

        if session is not None:
            from agent.memory.session import save_session

            save_session(session, agent.messages)

        # If streaming was active, the text is already printed; just add newline.
        # If no streaming occurred (tool-only turn), print the response normally.
        if streaming_tokens:
            console.print()  # end the streaming line
        elif response:
            console.print()
            console.print(Markdown(response))
        elif tool_results:
            console.print(
                f"[{t.text_dim}]Done. ({', '.join(tool_results)})[/{t.text_dim}]"
            )
        else:
            console.print(f"[{t.warning}]No response from model.[/{t.warning}]")

        # Post-turn usage summary (persists after the spinner clears).
        s = getattr(agent, "stats", None)
        if s and s.get("calls", 0) > 0:
            parts = [
                f"↑{s['input_tokens']}",
                f"↓{s['output_tokens']}",
            ]
            if s.get("in_tps"):
                parts.append(f"{s['in_tps']:.0f} in-tok/s")
            if s.get("out_tps"):
                parts.append(f"{s['out_tps']:.1f} out-tok/s")
            if s.get("reasoning_tokens"):
                parts.append(f"think {s['reasoning_tokens']}")
            if s.get("tool_tokens"):
                parts.append(f"tool {s['tool_tokens']}")
            console.print(f"[{t.text_dim}]{'  '.join(parts)}[/{t.text_dim}]")

        if cfg.ui.show_token_count:
            console.print(
                f"\n{_token_bar(agent.token_estimate(), cfg.llm.ctx_window)}\n"
            )

    return session


# ── Entry point ──────────────────────────────────────────────────────────────


def run_ui(agent: "Agent", session=None):
    cfg = agent.config
    if cfg.ui.mode == "textual":
        try:
            app = _build_textual_app(agent, session=session)
            app.run()
            return app._session
        except ImportError:
            print("Textual not available, falling back to simple mode.")
            return asyncio.run(simple_loop(agent, session=session))
    else:
        return asyncio.run(simple_loop(agent, session=session))
