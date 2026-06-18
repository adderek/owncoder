"""Widget and message class factory for the Textual TUI.

All classes capture ``t`` (theme) and ``_escape`` from the enclosing scope via
the builder function, so they cannot be module-level.  Call
``build_widget_classes(t)`` once inside ``_build_textual_app`` and
destructure the returned namespace.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ── module-level widgets (no theme dependency) ────────────────────────────────

def _build_spinner_widget():
    from textual.widgets import Static

    class SpinnerWidget(Static):
        """Single-character animated spinner. Replaces Textual's LoadingIndicator.

        Characters may be 1 or 2 columns wide; always rendered with a trailing
        space so the next character is not partially overwritten.
        """

        def __init__(self, frames: list[str], **kwargs):
            super().__init__("", **kwargs)
            self._frames = frames if frames else ["⠋"]
            self._idx = 0

        def on_mount(self) -> None:
            self.set_interval(0.1, self._tick)

        def _tick(self) -> None:
            frame = self._frames[self._idx % len(self._frames)]
            self._idx += 1
            self.update(f"{frame} ")

    return SpinnerWidget


try:
    SpinnerWidget = _build_spinner_widget()
except Exception:
    SpinnerWidget = None  # type: ignore[assignment,misc]


def build_widget_classes(t) -> SimpleNamespace:
    """Return a SimpleNamespace of all widget/message classes.

    ``t`` is the theme object whose attributes are Rich color/style strings.
    Classes reference it by closure so the theme is baked in at construction time.
    """
    from textual.widgets import Static, RichLog, TextArea
    from textual.message import Message
    from rich.markup import escape as _escape
    from rich.markdown import Markdown

    from agent.ui.textual_events import build_event_classes
    from agent.ui.render import (
        _CTX_SEGMENT_COLORS, _CTX_SEGMENT_LABELS,
        _OUT_SEGMENT_COLORS, _OUT_SEGMENT_LABELS,
        _labeled_bar_segment,
        _delatex,
    )
    from agent.ui.slash import _match_commands

    # Build event classes first so JumpToTurn is available to view mixins below.
    _events = build_event_classes()
    JumpToTurn = _events.JumpToTurn
    ExpandTurn = _events.ExpandTurn

    # ── helpers ───────────────────────────────────────────────────────────────

    def _one_line(text: str, limit: int = 160, wrap: bool = False) -> str:
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

        def update_tokens(self, used: int, peak: int = 0, compact_frac: float | None = None, ctx_window: int | None = None) -> None:
            self._used = used
            self._peak = peak
            if compact_frac is not None:
                self._compact_frac = compact_frac
            if ctx_window is not None:
                self._ctx = ctx_window
            self._redraw()

        def on_resize(self, _event) -> None:
            self._redraw()

        def on_mouse_move(self, event) -> None:
            self.tooltip = self._tooltip_at(event.x)

        def _bar_geometry(self) -> tuple[int, int, int]:
            """Return (label_len, bar_start, bar_len) for current width."""
            ctx = max(1, self._ctx)
            used = max(0, self._used)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            label = f"tokens: {used:,}/{ctx:,}"
            bar_len = max(10, width - len(label) - 2)
            return len(label), len(label) + 1, bar_len

        def hover_data_at(self, x_fraction: float) -> dict:
            """Return structured hover data for x_fraction in [0,1]. HTML-UI ready."""
            ctx = max(1, self._ctx)
            used = max(0, self._used)
            peak = max(0, self._peak)
            compact_frac = max(0.0, min(1.0, self._compact_frac))
            compact_tokens = int(compact_frac * ctx)
            _, bar_start_col, bar_len = self._bar_geometry()
            # x_fraction maps to bar position after label
            bar_frac = x_fraction  # caller maps raw x to fraction
            peak_frac = min(1.0, peak / ctx) if peak > 0 else -1.0
            compact_bar_frac = compact_frac
            peak_tol = 1.5 / max(1, bar_len)
            compact_tol = 1.5 / max(1, bar_len)
            if peak_frac >= 0 and abs(bar_frac - peak_frac) <= peak_tol:
                return {"type": "marker", "name": "peak", "tokens": peak,
                        "pct": peak / ctx * 100, "label": f"Peak this round: {peak:,} ({peak/ctx*100:.1f}%)"}
            if abs(bar_frac - compact_bar_frac) <= compact_tol:
                return {"type": "marker", "name": "compaction_threshold",
                        "tokens": compact_tokens, "pct": compact_frac * 100,
                        "label": f"Compaction threshold: {compact_frac*100:.0f}% ({compact_tokens:,} tokens)"}
            if bar_frac <= used / ctx:
                free = ctx - used
                return {"type": "fill", "tokens": used, "pct": used / ctx * 100,
                        "free": free, "label": f"Used: {used:,} / {ctx:,}  ({used/ctx*100:.1f}%)  free: {free:,}"}
            free = ctx - used
            return {"type": "empty", "tokens": free, "pct": free / ctx * 100,
                    "label": f"Free: {free:,} ({free/ctx*100:.1f}%)  used: {used:,}/{ctx:,}"}

        def _tooltip_at(self, x: int) -> str:
            _, bar_start, bar_len = self._bar_geometry()
            bar_x = x - bar_start
            bar_frac = max(0.0, min(1.0, bar_x / max(1, bar_len)))
            return self.hover_data_at(bar_frac)["label"]

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

        def set_segments(self, segments: list[dict], ctx_window: int | None = None) -> None:
            self._segments = list(segments)
            if ctx_window is not None:
                self._ctx = ctx_window
            self._redraw()

        def on_resize(self, _event) -> None:
            self._redraw()

        def on_mouse_move(self, event) -> None:
            self.tooltip = self._tooltip_at(event.x)

        def hover_data_at(self, x_fraction: float) -> dict:
            """Return structured hover data for x_fraction in [0,1]. HTML-UI ready."""
            ctx = max(1, self._ctx)
            total_used = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            if total_used <= 0:
                return {"type": "empty", "label": f"Context empty  (window: {ctx:,})"}
            cumulative = 0.0
            for seg in self._segments:
                tok = max(0, seg.get("tokens", 0))
                seg_frac = tok / total_used
                cumulative += seg_frac
                if x_fraction <= cumulative:
                    pct = tok / ctx * 100
                    return {"type": "segment", "label_key": seg["label"],
                            "tokens": tok, "pct": pct,
                            "label": f"{seg['label']}: {tok:,} ({pct:.1f}% of ctx)"}
            free = ctx - total_used
            return {"type": "free", "tokens": free, "pct": free / ctx * 100,
                    "label": f"Free: {free:,} ({free/ctx*100:.1f}%)"}

        def _tooltip_at(self, x: int) -> str:
            ctx = max(1, self._ctx)
            total_used = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            label = f"ctx: {total_used:,}/{ctx:,}"
            bar_start = len(label) + 1
            bar_len = max(10, width - len(label) - 2)
            bar_x = x - bar_start
            bar_frac = max(0.0, min(1.0, bar_x / max(1, bar_len)))
            return self.hover_data_at(bar_frac)["label"]

        def _redraw(self) -> None:
            ctx = max(1, self._ctx)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            total_used = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            label = f"ctx: {total_used:,}/{ctx:,}"
            bar_len = max(10, width - len(label) - 2)

            if total_used <= 0:
                self.update(f"{label} [dim]{'░' * bar_len}[/dim]")
                return

            cells = []
            remaining = bar_len
            for seg in self._segments:
                tok = max(0, seg.get("tokens", 0))
                raw = tok / total_used * bar_len
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

        def on_mouse_move(self, event) -> None:
            self.tooltip = self._tooltip_at(event.x)

        def hover_data_at(self, x_fraction: float) -> dict:
            """Return structured hover data for x_fraction in [0,1]. HTML-UI ready."""
            total = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            if total <= 0:
                return {"type": "empty", "label": "No output yet"}
            cumulative = 0.0
            for seg in self._segments:
                tok = max(0, seg.get("tokens", 0))
                cumulative += tok / total
                if x_fraction <= cumulative:
                    pct = tok / total * 100
                    return {"type": "segment", "label_key": seg["label"],
                            "tokens": tok, "pct": pct,
                            "label": f"{seg['label']}: {tok:,} ({pct:.1f}%)"}
            return {"type": "empty", "label": "No output"}

        def _tooltip_at(self, x: int) -> str:
            total = sum(max(0, s.get("tokens", 0)) for s in self._segments)
            width = int(getattr(self.size, "width", 0) or 0)
            if width < 20:
                width = 80
            label = f"{self._scope_label}: {total:,}"
            bar_start = len(label) + 1
            bar_len = max(10, width - len(label) - 2)
            bar_x = x - bar_start
            bar_frac = max(0.0, min(1.0, bar_x / max(1, bar_len)))
            return self.hover_data_at(bar_frac)["label"]

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
        """Live chat log — user ↔ agent turns. Click any line to expand that turn."""

        def on_click(self, event) -> None:
            line_idx = int(self.scroll_offset.y) + int(event.y)

            # Check if this line is a clickable file diff entry.
            file_lines = getattr(self.app, "_chat_file_lines", {})
            if line_idx in file_lines:
                entry = file_lines[line_idx]
                path = entry.get("path", "") if isinstance(entry, dict) else entry
                added = entry.get("added", 0) if isinstance(entry, dict) else 0
                removed = entry.get("removed", 0) if isinstance(entry, dict) else 0
                self.app.push_screen(
                    self.app._wt.FileDiffScreen(path, added, removed)
                )
                return

            qa_data = getattr(self.app, "_chat_qa_data", [])
            if not qa_data:
                return

            # Primary: per-visual-line ordinal map built during _restore_chat_history.
            line_to_ordinal = getattr(self.app, "_chat_line_to_ordinal", [])
            if line_to_ordinal:
                if 0 <= line_idx < len(line_to_ordinal):
                    ordinal = line_to_ordinal[line_idx]
                else:
                    ordinal = len(qa_data) - 1  # past end → last turn
                if ordinal < 0 or ordinal >= len(qa_data):
                    return
                q_d, a_d = qa_data[ordinal]
                self.post_message(ExpandTurn(ordinal, q_d, a_d))
                return

            # Fallback: anchor search for live sessions without line_to_ordinal.
            anchors = getattr(self.app, "_chat_user_lines", [])
            if not anchors:
                return
            ordinal = 0
            for i, line_no in enumerate(anchors):
                if line_no <= line_idx:
                    ordinal = i
                else:
                    break
            if ordinal < len(qa_data):
                q_d, a_d = qa_data[ordinal]
                self.post_message(ExpandTurn(ordinal, q_d, a_d))

    class SysView(RichLog):
        """System log — commands, session info, help output."""

    class _QALineTrackingMixin:
        """Mixin providing click-to-expand behavior shared by Q/A/Sparse views."""

        def _reset_line_map(self) -> None:
            self._line_ordinals: list[int] = []
            self._entry_count: int = 0
            self._qa_entries: list[tuple] = []

        def _track_line(self, ordinal: int) -> None:
            if not hasattr(self, "_line_ordinals"):
                self._line_ordinals = []
            self._line_ordinals.append(ordinal)

        def _store_entries(self, entries: list) -> None:
            self._qa_entries = list(entries)

        def on_click(self, event) -> None:
            ordinals = getattr(self, "_line_ordinals", [])
            if not ordinals:
                return
            line_idx = int(self.scroll_offset.y) + int(event.y)
            if 0 <= line_idx < len(ordinals):
                ordinal = ordinals[line_idx]
                entries = getattr(self, "_qa_entries", [])
                if 0 <= ordinal < len(entries):
                    _, q_data, a_data = entries[ordinal]
                    self.post_message(ExpandTurn(ordinal, q_data or {}, a_data or {}))
                else:
                    self.post_message(JumpToTurn(ordinal))

    class QView(_QALineTrackingMixin, RichLog):
        """View for user questions: one line per turn."""

        def load_history(self, entries: "list[tuple[int, dict, dict]]") -> None:
            self.clear()
            self._reset_line_map()
            self._store_entries(entries)
            for ordinal, (tid, q, _a) in enumerate(entries):
                self._write_entry(ordinal, tid, q)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, q_data: dict, a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            entries = getattr(self, "_qa_entries", [])
            entries.append((turn_id, q_data, a_data))
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
            self._store_entries(entries)
            for ordinal, (tid, _q, a) in enumerate(entries):
                self._write_entry(ordinal, tid, a)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, q_data: dict, a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            entries = getattr(self, "_qa_entries", [])
            entries.append((turn_id, q_data, a_data))
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
            self._store_entries(entries)
            for ordinal, (tid, q, a) in enumerate(entries):
                self._write_entry(ordinal, tid, q, a)
            self._entry_count = len(entries)

        def add_turn(self, turn_id: int, q_data: dict, a_data: dict) -> None:
            ordinal = getattr(self, "_entry_count", 0)
            self._entry_count = ordinal + 1
            entries = getattr(self, "_qa_entries", [])
            entries.append((turn_id, q_data, a_data))
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

    class QSummaryView(RichLog):
        """AI-generated session-level summary of all user questions."""

        def set_loading(self) -> None:
            self.clear()
            self.write(f"[{t.text_dim}]Summarizing questions…[/{t.text_dim}]")

        def set_summary(self, text: str) -> None:
            self.clear()
            if text:
                self.write(Markdown(text))
            else:
                self.write(f"[{t.text_dim}]No questions to summarize yet.[/{t.text_dim}]")

    class ASummaryView(RichLog):
        """AI-generated session-level summary of all agent responses."""

        def set_loading(self) -> None:
            self.clear()
            self.write(f"[{t.text_dim}]Summarizing responses…[/{t.text_dim}]")

        def set_summary(self, text: str) -> None:
            self.clear()
            if text:
                self.write(Markdown(text))
            else:
                self.write(f"[{t.text_dim}]No responses to summarize yet.[/{t.text_dim}]")

    _MODEL_STATUS_ROLES = [("llm", "main"), ("emb", "emb"), ("sum", "sum"), ("sec", "sec")]

    class ModelConfigScreen:
        pass  # defined below after ModalScreen import

    from textual.screen import ModalScreen
    from textual.widgets import Button
    from textual.containers import Vertical

    class ModelConfigScreen(ModalScreen):
        """Modal popup showing config for all model roles."""

        CSS = """
        ModelConfigScreen {
            align: center middle;
        }
        #model-config-dialog {
            width: 60;
            height: auto;
            max-height: 90%;
            border: solid $primary;
            background: $surface;
            padding: 1 2;
        }
        #model-config-body {
            height: auto;
            max-height: 22;
            overflow-y: auto;
            margin-bottom: 1;
        }
        #model-config-close {
            margin-top: 1;
            width: 100%;
        }
        #model-config-refresh {
            margin-top: 1;
            width: 100%;
        }
        """

        def __init__(self, configs: dict, server=None) -> None:
            super().__init__()
            self._configs = configs
            self._server = server

        def _build_body(self) -> str:
            lines = []
            role_labels = [("llm", "LLM  (main)"), ("emb", "Embed"), ("sum", "Summarizer")]
            for role, heading in role_labels:
                cfg = self._configs.get(role, {})
                avail = cfg.get("available")
                if avail is True:
                    badge = f"  [rgb(56,142,60)]● online[/]"
                elif avail is False:
                    badge = f"  [rgb(198,40,40)]● OFFLINE — model not on endpoint[/]"
                else:
                    badge = ""  # not probed / unknown
                lines.append(f"[bold]{heading}[/bold]{badge}")
                for k, v in cfg.items():
                    if k == "available":
                        continue
                    lines.append(f"  [{t.text_dim}]{k}[/{t.text_dim}]  {_escape(str(v))}")
                lines.append("")
            return "\n".join(lines).rstrip()

        def compose(self):
            from textual.containers import Vertical, ScrollableContainer
            from textual.widgets import Button, Static
            with Vertical(id="model-config-dialog"):
                with ScrollableContainer(id="model-config-body"):
                    yield Static(self._build_body(), id="model-config-body-text", markup=True)
                if self._server is not None:
                    yield Button("Refresh ctx sizes", id="model-config-refresh")
                yield Button("Close  [ESC]", id="model-config-close")

        async def on_button_pressed(self, event) -> None:
            if event.button.id == "model-config-refresh" and self._server is not None:
                import asyncio
                event.button.disabled = True
                event.button.label = "Refreshing…"
                try:
                    await asyncio.to_thread(self._server.refresh_model_info)
                    self._configs = self._server.get_model_configs()
                    self.query_one("#model-config-body-text").update(self._build_body())
                    event.button.label = "Refresh ctx sizes"
                except Exception:
                    event.button.label = "Refresh failed"
                finally:
                    event.button.disabled = False
                return
            self.dismiss()

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss()

    class WorkersScreen(ModalScreen):
        """Modal showing parallel worker status. Auto-refreshes while workers run."""

        CSS = """
        WorkersScreen {
            align: center middle;
        }
        #workers-dialog {
            width: 72;
            height: auto;
            max-height: 36;
            border: solid $primary;
            background: $surface;
            padding: 1 2;
        }
        #workers-close {
            margin-top: 1;
            width: 100%;
        }
        """

        def compose(self):
            from textual.containers import Vertical
            from textual.widgets import Button, Static
            with Vertical(id="workers-dialog"):
                yield Static("", id="workers-body", markup=True)
                yield Button("Close  [ESC]", id="workers-close")

        def on_mount(self) -> None:
            self.set_interval(0.25, self._refresh)
            self._refresh()

        def _refresh(self) -> None:
            from agent.core.model_status import get_workers
            workers = get_workers()
            if not workers:
                body = f"[{t.text_dim}]No parallel workers recorded.[/{t.text_dim}]"
            else:
                lines = [f"[bold]Parallel workers[/bold]  [{t.text_dim}](most recent first)[/{t.text_dim}]\n"]
                status_color = {
                    "running": "rgb(232,128,26)",
                    "done": "rgb(56,142,60)",
                    "error": "rgb(198,40,40)",
                }
                status_icon = {"running": "●", "done": "✓", "error": "✗"}
                for w in workers:
                    sc = status_color.get(w["status"], t.text_dim)
                    ic = status_icon.get(w["status"], "?")
                    elapsed = f"{w['elapsed']}s"
                    lines.append(
                        f"[{sc}]{ic}[/{sc}] [{t.text_dim}]#{w['id']}[/{t.text_dim}]"
                        f"  [bold]{_escape(w['model'])}[/bold]"
                        f"  [{t.text_dim}]{elapsed}[/{t.text_dim}]"
                    )
                    lines.append(f"   [{t.text_dim}]{_escape(w['task'])}[/{t.text_dim}]")
                    if w["error"]:
                        lines.append(f"   [rgb(198,40,40)]{_escape(w['error'])}[/]")
                    lines.append("")
                body = "\n".join(lines).rstrip()
            try:
                self.query_one("#workers-body").update(body)
            except Exception:
                pass

        def on_button_pressed(self, event) -> None:
            self.dismiss()

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss()

    class TurnDetailScreen(ModalScreen):
        """Modal showing full Q+A content for a clicked turn."""

        CSS = """
        TurnDetailScreen {
            align: center middle;
        }
        #turn-detail-dialog {
            width: 90%;
            max-width: 120;
            height: auto;
            max-height: 80%;
            border: solid $primary;
            background: $surface;
            padding: 1 2;
        }
        #turn-detail-q {
            margin-bottom: 1;
            max-height: 15;
            overflow-y: auto;
        }
        #turn-detail-a {
            margin-bottom: 1;
            max-height: 25;
            overflow-y: auto;
        }
        #turn-detail-close {
            margin-top: 1;
            width: 100%;
        }
        .tool-call-btn {
            width: 100%;
            margin-bottom: 0;
        }
        """

        def __init__(self, ordinal: int, q_data: dict, a_data: dict, session_dir=None) -> None:
            super().__init__()
            self._ordinal = ordinal
            self._q_data = q_data
            self._a_data = a_data
            self._session_dir = session_dir

        def compose(self):
            from textual.containers import Vertical, ScrollableContainer
            from textual.widgets import Button, Static
            from rich.markdown import Markdown as _Md
            from agent.ui.render import _delatex

            q_content = self._q_data.get("content", "") or ""
            a_content = self._a_data.get("content", "") or ""
            tid = self._q_data.get("turn_id") or self._a_data.get("turn_id") or (self._ordinal + 1)

            tools = self._a_data.get("tool_calls") or []
            files = self._a_data.get("modified_files") or []
            # Normalize: stored as strings (names) or dicts; extract name only for dedup.
            tool_names = list(dict.fromkeys(
                (n if isinstance(n, str) else n.get("name", "?")) for n in tools
            ))
            files = list(dict.fromkeys(files))

            with Vertical(id="turn-detail-dialog"):
                yield Static(
                    f"[bold {t.user_color}]Turn {tid} — Q[/bold {t.user_color}]",
                    markup=True,
                )
                with ScrollableContainer(id="turn-detail-q"):
                    yield Static(_Md(q_content.strip()) if q_content else f"[{t.text_dim}](empty)[/{t.text_dim}]", id="turn-q-body", markup=not bool(q_content))
                yield Static(
                    f"[bold {t.agent_color}]A[/bold {t.agent_color}]",
                    markup=True,
                )
                with ScrollableContainer(id="turn-detail-a"):
                    a_display = _delatex(a_content.strip()) if a_content else ""
                    yield Static(_Md(a_display) if a_display else f"[{t.text_dim}](empty)[/{t.text_dim}]", id="turn-a-body", markup=not bool(a_display))
                if tool_names or files:
                    from agent.ui.render import tool_icon as _ti
                    if tool_names:
                        yield Static(f"[{t.text_dim}]Tools (click for details):[/{t.text_dim}]", markup=True)
                        for idx, name in enumerate(tool_names):
                            yield Button(
                                f"{_ti(name)} {name}",
                                id=f"tool-btn-{idx}",
                                classes="tool-call-btn",
                            )
                    if files:
                        yield Static(f"[{t.text_dim}]Files (click for diff):[/{t.text_dim}]", markup=True)
                        for fidx, fentry in enumerate(files):
                            if isinstance(fentry, dict):
                                fp = fentry.get("path", "")
                                fa = fentry.get("added", 0)
                                fr = fentry.get("removed", 0)
                                stat = f" +{fa}/-{fr}" if (fa or fr) else ""
                            else:
                                fp = str(fentry)
                                stat = ""
                            yield Button(
                                f"📄 {_escape(fp)}{stat}",
                                id=f"file-btn-{fidx}",
                                classes="tool-call-btn",
                            )
                yield Button("Close  [ESC]", id="turn-detail-close")

        def on_button_pressed(self, event) -> None:
            btn_id = event.button.id or ""
            if btn_id == "turn-detail-close":
                self.dismiss()
                return
            if btn_id.startswith("tool-btn-"):
                idx = int(btn_id[len("tool-btn-"):])
                tools = self._a_data.get("tool_calls") or []
                tool_names = list(dict.fromkeys(
                    (n if isinstance(n, str) else n.get("name", "?")) for n in tools
                ))
                if idx < len(tool_names):
                    tool_name = tool_names[idx]
                    tid = self._q_data.get("turn_id") or self._a_data.get("turn_id") or (self._ordinal + 1)
                    self.app.push_screen(ToolCallDetailScreen(tool_name, tid, self._session_dir))
            if btn_id.startswith("file-btn-"):
                idx = int(btn_id[len("file-btn-"):])
                files = self._a_data.get("modified_files") or []
                files = list(dict.fromkeys(files))
                if idx < len(files):
                    fentry = files[idx]
                    if isinstance(fentry, dict):
                        fp = fentry.get("path", "")
                        fa = fentry.get("added", 0)
                        fr = fentry.get("removed", 0)
                    else:
                        fp = str(fentry)
                        fa = fr = 0
                    self.app.push_screen(FileDiffScreen(fp, fa, fr))

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss()

    class ToolCallDetailScreen(ModalScreen):
        """Modal showing arguments and result for a single tool call."""

        CSS = """
        ToolCallDetailScreen {
            align: center middle;
        }
        #tc-detail-dialog {
            width: 90%;
            max-width: 120;
            height: 85%;
            border: solid $accent;
            background: $surface;
            padding: 1 2;
        }
        #tc-detail-body {
            height: 1fr;
            overflow-y: auto;
        }
        .tc-detail-block {
            height: auto;
            margin-bottom: 1;
        }
        #tc-detail-close {
            width: 100%;
            dock: bottom;
        }
        """

        def __init__(self, tool_name: str, turn_id: int, session_dir=None) -> None:
            super().__init__()
            self._tool_name = tool_name
            self._turn_id = turn_id
            self._session_dir = session_dir

        def _load_records(self) -> list[dict]:
            if self._session_dir is None:
                return []
            try:
                import json
                from pathlib import Path
                p = Path(self._session_dir) / "tool_calls.jsonl"
                if not p.exists():
                    return []
                records = []
                seen_ids: set = set()
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        if rec.get("tool") == self._tool_name and rec.get("turn") == self._turn_id:
                            # Same call may be logged twice (live exec + history
                            # compaction). Keep the first row per tool_call_id.
                            cid = rec.get("tool_call_id")
                            if cid is not None and cid in seen_ids:
                                continue
                            if cid is not None:
                                seen_ids.add(cid)
                            records.append(rec)
                return records
            except Exception:
                return []

        def compose(self):
            import json as _json
            from textual.containers import Vertical, ScrollableContainer
            from textual.widgets import Button, Static
            from rich.markup import escape as _esc

            records = self._load_records()

            with Vertical(id="tc-detail-dialog"):
                from agent.ui.render import tool_icon as _ti
                yield Static(
                    f"[bold]{_ti(self._tool_name)} {_esc(self._tool_name)}[/bold]  [dim](turn {self._turn_id})[/dim]",
                    markup=True,
                )
                # Single outer scroller holds every call; the Back button is docked
                # to the dialog bottom so it stays reachable no matter how many
                # calls there are. (markup=False on tool I/O: it is arbitrary text
                # that may contain [..] / <tags> — parsing it as Rich markup raises
                # MarkupError and crashes the app.)
                with ScrollableContainer(id="tc-detail-body"):
                    if not records:
                        yield Static("[dim]No side-log records found for this tool call.[/dim]", markup=True)
                    else:
                        for i, rec in enumerate(records):
                            if len(records) > 1:
                                yield Static(f"[dim]— call {i+1}/{len(records)} —[/dim]", markup=True)
                            args = rec.get("arguments", {})
                            args_str = _json.dumps(args, indent=2, ensure_ascii=False) if args else "(none)"
                            yield Static("[bold]Arguments:[/bold]", markup=True)
                            yield Static(args_str, markup=False, classes="tc-detail-block")
                            result_raw = rec.get("result", "")
                            try:
                                result_parsed = _json.loads(result_raw)
                                result_str = _json.dumps(result_parsed, indent=2, ensure_ascii=False)
                            except Exception:
                                result_str = result_raw or "(empty)"
                            yield Static("[bold]Result:[/bold]", markup=True)
                            yield Static(result_str[:4000], markup=False, classes="tc-detail-block")
                yield Button("Back  [ESC]", id="tc-detail-close")

        def on_button_pressed(self, event) -> None:
            self.dismiss()

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss()

    class FileDiffScreen(ModalScreen):
        """Modal showing git diff for a modified file."""

        CSS = """
        FileDiffScreen {
            align: center middle;
        }
        #diff-dialog {
            width: 90%;
            max-width: 120;
            height: auto;
            max-height: 80%;
            border: solid $accent;
            background: $surface;
            padding: 1 2;
        }
        #diff-body {
            max-height: 35;
            overflow-y: auto;
            margin-bottom: 1;
        }
        #diff-close {
            width: 100%;
        }
        """

        def __init__(self, file_path: str, added: int = 0, removed: int = 0) -> None:
            super().__init__()
            self._file_path = file_path
            self._added = added
            self._removed = removed

        def _get_diff(self) -> str:
            import subprocess
            try:
                result = subprocess.run(
                    ["git", "diff", "--", self._file_path],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout
                # Try staged diff too
                result = subprocess.run(
                    ["git", "diff", "--cached", "--", self._file_path],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout
            except Exception:
                pass
            return f"[dim]No diff available for {_escape(self._file_path)}[/dim]"

        def compose(self):
            from textual.containers import Vertical, ScrollableContainer
            from textual.widgets import Button, Static
            from rich.markup import escape as _esc
            from rich.diff import Diff

            diff_text = self._get_diff()
            stat = ""
            if self._added or self._removed:
                stat = f"  [{t.success}]+{self._added}[/{t.success}] [{t.error}]-{self._removed}[/{t.error}]"

            with Vertical(id="diff-dialog"):
                yield Static(
                    f"[bold]📄 {_esc(self._file_path)}[/bold]{stat}",
                    markup=True,
                )
                with ScrollableContainer(id="diff-body"):
                    yield Static(diff_text, markup=False)
                yield Button("Close  [ESC]", id="diff-close")

        def on_button_pressed(self, event) -> None:
            self.dismiss()

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss()

    class ModelStatusBar(Static):
        """Compact inline indicator of model request states (idle/running). Click to view config."""

        def on_mount(self) -> None:
            self.set_interval(0.5, self._refresh)
            # Re-probe availability periodically (endpoints may come/go).
            self.set_interval(30.0, self._probe_availability)
            self.tooltip = "Click to view model config / worker status"
            self._probe_availability()

        def _probe_availability(self) -> None:
            server = getattr(self.app, "_server", None)
            if server is None or not hasattr(server, "probe_model_availability"):
                return
            # Off the UI thread — a /models GET can block on a dead endpoint.
            try:
                self.run_worker(
                    lambda: server.probe_model_availability(),
                    thread=True,
                    exclusive=True,
                    group="model-availability",
                )
            except Exception:
                pass

        def _refresh(self) -> None:
            from agent.core.model_status import get_states, get_counts, get_availability
            states = get_states()
            avail = get_availability()
            parts = []
            for label, role in _MODEL_STATUS_ROLES:
                # Offline (configured model missing on its endpoint) → red, takes
                # priority over idle/running so the user can spot it at a glance.
                if avail.get(label) is False:
                    parts.append(f"[rgb(198,40,40)]{label}:✗[/]")
                elif states.get(role, "idle") == "running":
                    parts.append(f"[rgb(56,142,60)]{label}:●[/]")
                else:
                    parts.append(f"[dim]{label}:○[/dim]")
            worker_count = get_counts().get("workers", 0)
            if worker_count > 0:
                parts.append(f"[rgb(232,128,26)]agents:{worker_count}●[/]")
            self.update("  ".join(parts))

        async def on_click(self) -> None:
            import asyncio
            from agent.core.model_status import get_workers
            if get_workers():
                self.app.push_screen(WorkersScreen())
                return
            server = getattr(self.app, "_server", None)
            if server is not None:
                try:
                    await asyncio.to_thread(server.refresh_model_info)
                except Exception:
                    pass
            try:
                configs = server.get_model_configs() if server else {}
            except Exception:
                configs = {}
            self.app.push_screen(ModelConfigScreen(configs, server=server))

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

        _HISTORY_PREFS_KEY = "input_history"
        _MAX_SAVED_HISTORY = 200

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

        def on_mount(self) -> None:
            try:
                from agent.ui.prefs import load_prefs
                saved = load_prefs().get(self._HISTORY_PREFS_KEY, [])
                if isinstance(saved, list):
                    self._history = [str(h) for h in saved if h]
            except Exception:
                pass

        def on_unmount(self) -> None:
            try:
                from agent.ui.prefs import load_prefs, save_prefs
                prefs = load_prefs()
                prefs[self._HISTORY_PREFS_KEY] = self._history[-self._MAX_SAVED_HISTORY:]
                save_prefs(prefs)
            except Exception:
                pass

        def add_to_history(self, text: str) -> None:
            self._history = [h for h in self._history if h != text]
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
                text_parts = text.split()
                cmd_part = text_parts[0] if text_parts else text
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
                if event.key == "up":
                    event.prevent_default()
                    if self._history_idx is not None and self._history_idx > 0:
                        self._enter_browsing(self._history_idx - 1)
                elif event.key == "down":
                    event.prevent_default()
                    if self._history_idx is not None:
                        if self._history_idx < len(self._history) - 1:
                            self._enter_browsing(self._history_idx + 1)
                        else:
                            self._exit_browsing()
                elif event.key == "escape":
                    event.prevent_default()
                    self._exit_browsing()
                elif event.key == "enter":
                    event.prevent_default()
                    self._enter_editing()
                else:
                    self._exit_browsing()

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

    # ── paths view ────────────────────────────────────────────────────────────

    def _path_id(path) -> str:
        """Safe widget ID from a path string."""
        return "pg-" + str(path).replace("/", "_").replace(".", "_").replace(" ", "_")[-60:]

    class PathGrantRow(Static):
        """One row per PathGrant in the paths tab."""

        DEFAULT_CSS = "PathGrantRow { height: auto; }"

        class Accepted(Message):
            def __init__(self, path) -> None:
                super().__init__()
                self.path = path

        class Rejected(Message):
            def __init__(self, path) -> None:
                super().__init__()
                self.path = path

        def __init__(self, grant, **kwargs) -> None:
            super().__init__(**kwargs)
            self._grant = grant

        def compose(self):
            from textual.containers import Horizontal as _H
            from textual.widgets import Button as _Btn
            g = self._grant
            mode_color = t.success if g.mode == "rw" else t.warning
            origin_colors = {
                "default": t.text_dim,
                "user": t.success,
                "agent": t.warning,
            }
            oc = origin_colors.get(g.origin, t.text_dim)
            state_icon = "⚠ " if g.state == "pending" else "  "
            label = (
                f"{state_icon}[bold]{_escape(str(g.path))}[/bold]"
                f"  [{oc}]{g.origin}[/{oc}]"
                f"  [{mode_color}]{g.mode.upper()}[/{mode_color}]"
            )
            if g.state == "pending":
                label += f"  [{t.warning}][pending][/{t.warning}]"
            with _H(classes="grant-row"):
                yield Static(label, markup=True, classes="grant-path-label")
                if g.state == "pending":
                    yield _Btn("Accept", id=f"accept-{_path_id(g.path)}", variant="success", classes="grant-btn")
                    yield _Btn("Reject", id=f"reject-{_path_id(g.path)}", variant="error", classes="grant-btn")

        def on_button_pressed(self, event) -> None:
            bid = event.button.id or ""
            path = self._grant.path
            if bid.startswith("accept-"):
                self.post_message(PathGrantRow.Accepted(path))
                event.stop()
            elif bid.startswith("reject-"):
                self.post_message(PathGrantRow.Rejected(path))
                event.stop()

    class PathsView(Static):
        """Paths tab: list of accessible paths with Accept/Reject for pending."""

        DEFAULT_CSS = "PathsView { height: 1fr; }"

        class AddPathClicked(Message):
            pass

        def compose(self):
            from textual.containers import VerticalScroll as _VS, Horizontal as _H
            from textual.widgets import Button as _Btn
            yield Static(
                f"[bold]Accessible Paths[/bold]  [{t.text_dim}]— paths the agent can read/write[/{t.text_dim}]",
                markup=True,
                id="paths-header",
            )
            yield _VS(id="paths-rows")
            with _H(id="paths-actions-row"):
                yield _Btn("+ Add path", id="btn-add-path", classes="add-path-btn")

        def refresh_grants(self) -> None:
            from agent.security import path_grants as _pg
            try:
                rows = self.query_one("#paths-rows")
            except Exception:
                return
            for child in list(rows.children):
                child.remove()
            grants = _pg.get_all()
            if not grants:
                rows.mount(Static(f"[{t.text_dim}](no grants)[/{t.text_dim}]", markup=True))
            for g in grants:
                rows.mount(PathGrantRow(g))

        def on_button_pressed(self, event) -> None:
            if (event.button.id or "") == "btn-add-path":
                self.post_message(PathsView.AddPathClicked())
                event.stop()

    # ── async event messages ──────────────────────────────────────────────────

    ToolCallEvent = _events.ToolCallEvent
    ToolResultEvent = _events.ToolResultEvent
    TokenStreamEvent = _events.TokenStreamEvent
    IterationProgressEvent = _events.IterationProgressEvent
    PhaseEvent = _events.PhaseEvent
    ReasoningTokenEvent = _events.ReasoningTokenEvent
    ContextSizeEvent = _events.ContextSizeEvent

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
        ExpandTurn=ExpandTurn,
        TurnDetailScreen=TurnDetailScreen,
        ToolCallDetailScreen=ToolCallDetailScreen,
        _QALineTrackingMixin=_QALineTrackingMixin,
        QView=QView,
        AView=AView,
        QSummaryView=QSummaryView,
        ASummaryView=ASummaryView,
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
        PathGrantRow=PathGrantRow,
        PathsView=PathsView,
    )
