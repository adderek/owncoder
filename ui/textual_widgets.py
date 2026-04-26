"""Widget and message class factory for the Textual TUI.

All classes capture ``t`` (theme) and ``_escape`` from the enclosing scope via
the builder function, so they cannot be module-level.  Call
``build_widget_classes(t, agent)`` once inside ``_build_textual_app`` and
destructure the returned namespace.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def build_widget_classes(t, agent) -> SimpleNamespace:  # noqa: ARG001
    """Return a SimpleNamespace of all widget/message classes.

    ``t`` is ``agent.config.ui.theme`` — a plain object whose attributes are
    Rich color/style strings.  Classes reference it by closure so the theme is
    baked in at construction time.
    """
    from textual.widgets import Static, RichLog, TextArea
    from textual.message import Message
    from rich.markup import escape as _escape

    from agent.ui.render import (
        _CTX_SEGMENT_COLORS, _CTX_SEGMENT_LABELS,
        _OUT_SEGMENT_COLORS, _OUT_SEGMENT_LABELS,
        _labeled_bar_segment,
    )
    from agent.ui.slash import _match_commands

    # ── helpers ───────────────────────────────────────────────────────────────

    def _one_line(text: str, limit: int = 120, wrap: bool = False) -> str:
        text = (text or "").strip()
        if not wrap:
            text = text.replace("\n", " ")
            if len(text) > limit:
                text = text[: limit - 1] + "…"
        return text

    # ── bars ─────────────────────────────────────────────────────────────────

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
            if used_frac > 0.85:
                fill_rgb = "rgb(198,40,40)"
            elif used_frac > 0.65:
                fill_rgb = "rgb(249,168,37)"
            else:
                fill_rgb = "rgb(56,142,60)"
            empty_rgb = "rgb(30,30,30)"
            peak_rgb  = "rgb(186,85,211)"
            thresh_rgb = "rgb(198,40,40)"
            parts = []
            for i in range(bar_len):
                is_filled = i < used_cells
                bg = fill_rgb if is_filled else empty_rgb
                if i == peak_cell and peak_cell >= 0:
                    parts.append(f"[bold {peak_rgb} on {bg}]▕[/]")
                elif i == compact_cell:
                    parts.append(f"[bold {thresh_rgb} on {bg}]🞀[/]")
                elif is_filled:
                    parts.append(f"[{fill_rgb} on {fill_rgb}]█[/]")
                else:
                    parts.append(f"[rgb(70,70,70) on {empty_rgb}] [/]")
            self.update(f"{label} {''.join(parts)}")

    class ContextBreakdownBar(Static):
        """One-line segmented bar showing how the context window is filled."""

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
            total_used = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            label = f"ctx: {total_used:,}/{ctx:,}"
            bar_len = max(10, width - len(label) - 2)

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
                color = _CTX_SEGMENT_COLORS.get(seg["label"], "rgb(128,128,128)")
                short = _CTX_SEGMENT_LABELS.get(seg["label"], seg["label"])
                parts.append(_labeled_bar_segment(short, n, color))
            if remaining > 0:
                parts.append(f"[dim]{'░' * remaining}[/dim]")
            self.update(f"{label} {''.join(parts)}")

    class OutputBreakdownBar(Static):
        """One-line segmented bar showing how model output tokens were spent."""

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
                color = _OUT_SEGMENT_COLORS.get(seg["label"], "rgb(128,128,128)")
                short = _OUT_SEGMENT_LABELS.get(seg["label"], seg["label"])
                parts.append(_labeled_bar_segment(short, n, color))
            if remaining > 0:
                parts.append(f"[dim]{'░' * remaining}[/dim]")
            self.update(f"{label} {''.join(parts)}")

    # ── log views ─────────────────────────────────────────────────────────────

    class ConversationView(RichLog):
        """Live chat log — user ↔ agent turns."""

    class SysView(RichLog):
        """System log — commands, session info, help output."""

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

    # ── utility panels ────────────────────────────────────────────────────────

    class ContextPanel(Static):
        def set_context(self, text: str) -> None:
            self.update(text)

    class GitStatusBar(Static):
        def set_status(self, text: str) -> None:
            self.update(text)

    _MODEL_STATUS_ROLES = [("llm", "main"), ("emb", "emb"), ("sum", "sum")]

    class ModelStatusBar(Static):
        """Compact inline indicator of model request states (idle/running)."""

        def on_mount(self) -> None:
            self.set_interval(0.15, self._refresh)

        def _refresh(self) -> None:
            from agent.core.model_status import get_states
            states = get_states()
            parts = []
            for label, role in _MODEL_STATUS_ROLES:
                if states.get(role, "idle") == "running":
                    parts.append(f"[rgb(56,142,60)]{label}:●[/]")
                else:
                    parts.append(f"[dim]{label}:○[/dim]")
            self.update("  ".join(parts))

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
                    f" {marker} {cmd_part:<20} [{t.text_dim}]{desc.replace('[', '\\[')}[/{t.text_dim}]"
                )
            if len(matches) > self.MAX_VISIBLE:
                lines.append(
                    f"[{t.text_dim}]   … {len(matches) - self.MAX_VISIBLE} more[/{t.text_dim}]"
                )
            self.update("\n".join(lines))
            self.add_class("visible")

    # ── prompt input ──────────────────────────────────────────────────────────

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

        def _on_key(self, event) -> None:
            if self._mode == "normal":
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

    # ── async event messages ──────────────────────────────────────────────────

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

    # ── placeholder text ──────────────────────────────────────────────────────

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

    return SimpleNamespace(
        _one_line=_one_line,
        TokenBar=TokenBar,
        ContextBreakdownBar=ContextBreakdownBar,
        OutputBreakdownBar=OutputBreakdownBar,
        ConversationView=ConversationView,
        SysView=SysView,
        JumpToTurn=JumpToTurn,
        _QALineTrackingMixin=_QALineTrackingMixin,
        QView=QView,
        AView=AView,
        SparseView=SparseView,
        ContextPanel=ContextPanel,
        GitStatusBar=GitStatusBar,
        ModelStatusBar=ModelStatusBar,
        HintBar=HintBar,
        CompletionBar=CompletionBar,
        PromptInput=PromptInput,
        ToolCallEvent=ToolCallEvent,
        ToolResultEvent=ToolResultEvent,
        TokenStreamEvent=TokenStreamEvent,
        IterationProgressEvent=IterationProgressEvent,
        PhaseEvent=PhaseEvent,
        ReasoningTokenEvent=ReasoningTokenEvent,
        ContextSizeEvent=ContextSizeEvent,
        _PLACEHOLDER_Q=_PLACEHOLDER_Q,
        _PLACEHOLDER_A=_PLACEHOLDER_A,
        _PLACEHOLDER_SPARSE=_PLACEHOLDER_SPARSE,
    )
