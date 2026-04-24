"""Event handler mixin for CodeAgentApp.

Handles all on_* Textual events and related helpers.
Accesses self._t, self._wt, and app state set in CodeAgentApp.__init__.
"""
from __future__ import annotations

import json
import logging
import traceback

from rich.markup import escape as _escape
from rich.markdown import Markdown

logger = logging.getLogger(__name__)


class EventHandlerMixin:
    """Textual event handlers for agent tool/stream/worker events."""

    def on_jump_to_turn(self, event) -> None:
        anchors = self._chat_user_lines
        if not (0 <= event.ordinal < len(anchors)):
            return
        self._switch_to_chat()
        chat_log = self.query_one("#chat-log", self._wt.ConversationView)
        y = anchors[event.ordinal]
        try:
            chat_log.scroll_to(y=y, animate=True)
        except Exception:
            logger.exception("jump_to_turn: scroll failed (ignored)")

    def on_tool_call_event(self, event) -> None:
        t = self._t
        # Track modified files
        if event.name in ("write_file", "patch_file", "edit_file"):
            try:
                args = json.loads(event.args) if isinstance(event.args, str) else event.args
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
        # Build context-panel preview
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
                preview = ", ".join(f"{k}={_pval(v)}" for k, v in list(args.items())[:2])
        except Exception:
            pass
        label = f"[{t.tool_color}]⚙ {_escape(event.name)}[/{t.tool_color}]"
        if preview:
            label += f" [dim]({_escape(preview)})[/dim]"
        self.query_one("#context-panel", self._wt.ContextPanel).set_context(label)

    def on_tool_result_event(self, event) -> None:
        stats = self._tool_stats.setdefault(event.name, {"ok": 0, "err": 0})
        if event.ok:
            stats["ok"] += 1
        else:
            stats["err"] += 1

    def _render_tool_summary(self) -> str:
        t = self._t
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

    def on_iteration_progress_event(self, event) -> None:
        self._iter_done = event.done
        self._iter_limit = event.limit
        self._update_loading_tokens()

    def on_context_size_event(self, event) -> None:
        self._refresh_token_bar()

    def on_phase_event(self, event) -> None:
        t = self._t
        detail = f": {_escape(event.detail)}" if event.detail else ""
        self.query_one("#context-panel", self._wt.ContextPanel).set_context(
            f"[dim]• {_escape(event.label)}{detail}[/dim]"
        )

    def on_reasoning_token_event(self, event) -> None:
        from rich.text import Text
        t = self._t
        self._stream_buffer += event.token
        stream_view = self.query_one("#stream-view")
        if not self._streaming_active:
            self._streaming_active = True
            stream_view.add_class("active")
        tail = self._stream_buffer[-800:] if len(self._stream_buffer) > 800 else self._stream_buffer
        content = Text.assemble(("thinking:", "dim italic"), (f" {tail}▌", "dim italic"))
        stream_view.update(content)

    def on_token_stream_event(self, event) -> None:
        from rich.text import Text
        t = self._t
        self._stream_buffer += event.token
        stream_view = self.query_one("#stream-view")
        if not self._streaming_active:
            self._streaming_active = True
            stream_view.add_class("active")
        tail = self._stream_buffer[-800:] if len(self._stream_buffer) > 800 else self._stream_buffer
        content = Text.assemble(("Agent:", f"bold {t.agent_color}"), (f" {tail}▌",))
        stream_view.update(content)

    def on_worker_state_changed(self, event) -> None:
        from textual.worker import WorkerState
        from rich.markdown import Markdown as _Markdown

        if event.worker.name != "chat":
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return

        t = self._t

        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None
        self.query_one("#loading-row").remove_class("active")

        input_widget = self.query_one("#input-bar", self._wt.PromptInput)
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
                "".join(traceback.format_exception(type(err), err, err.__traceback__))
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
        self.query_one("#context-panel", self._wt.ContextPanel).set_context(
            f"{tools_line}\n{token_line}" if tools_line else token_line
        )

        if self._streaming_active:
            stream_view = self.query_one("#stream-view")
            stream_view.remove_class("active")
            stream_view.update("")
            self._streaming_active = False
            self._stream_buffer = ""

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
                self._write_chat(f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]")
                self._write_chat(_Markdown(response))

        if event.state == WorkerState.SUCCESS:
            if self._round_summary_enabled:
                self._write_round_summary(getattr(self, "_current_user_text", ""), response or "")
            self._append_qa_turn(getattr(self, "_current_user_text", ""), response or "")

        self._refresh_token_bar()
        self.call_later(self._refresh_git)
