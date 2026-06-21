"""Event handler mixin for CodeAgentApp.

Handles all on_* Textual events and related helpers.
Accesses self._t, self._wt, and app state set in CodeAgentApp.__init__.
"""
from __future__ import annotations

import json
import logging
import time
import traceback

from rich.markup import escape as _escape
from rich.markdown import Markdown

logger = logging.getLogger(__name__)


def _fmt_tps(v: float) -> str:
    if v >= 10:
        return f"{v:.0f}"
    s = f"{v:.1f}"
    return s.lstrip("0") or "0"


def _compute_file_diffs(paths: "list[str]") -> "list[dict]":
    """Run git diff --numstat for the given paths and return [{path, added, removed}].

    Returns entries even if git is unavailable (added/removed = 0).
    """
    import subprocess
    if not paths:
        return []
    try:
        result = subprocess.run(
            ["git", "diff", "--numstat", "--"] + paths,
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            # git not available or no changes; return paths with zero stats
            return [{"path": p, "added": 0, "removed": 0} for p in paths]
        stats: dict[str, dict] = {p: {"path": p, "added": 0, "removed": 0} for p in paths}
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                a, r, fpath = parts
                a_int = int(a) if a != "-" else 0
                r_int = int(r) if r != "-" else 0
                # Match by basename or full path
                for p in paths:
                    if fpath == p or fpath.endswith(p) or p.endswith(fpath):
                        stats[p] = {"path": p, "added": a_int, "removed": r_int}
                        break
        return list(stats.values())
    except Exception:
        return [{"path": p, "added": 0, "removed": 0} for p in paths]


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

    def on_expand_turn(self, event) -> None:
        try:
            session_dir = None
            if getattr(self, "_session", None) is not None:
                try:
                    from agent.memory.session import get_session_full_dir
                    session_dir = get_session_full_dir(self._session.id)
                except Exception:
                    pass
            self.push_screen(self._wt.TurnDetailScreen(event.ordinal, event.q_data, event.a_data, session_dir=session_dir))
        except Exception:
            logger.exception("on_expand_turn: push_screen failed (ignored)")

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
        from agent.ui.render import tool_icon as _ti
        label = f"[{t.tool_color}]{_ti(event.name)} {_escape(event.name)}[/{t.tool_color}]"
        if preview:
            safe_preview = preview.replace("[", "\\[").replace("]", "\\]")
            label += f" [dim]({safe_preview})[/dim]"
        self.query_one("#context-panel", self._wt.ContextPanel).set_context(label)
        self._title_task_label = event.name

    def on_tool_result_event(self, event) -> None:
        stats = self._tool_stats.setdefault(event.name, {"ok": 0, "err": 0})
        if event.ok:
            stats["ok"] += 1
        else:
            stats["err"] += 1

    def _render_file_diffs(self) -> list[str]:
        """Return one line per modified file with +N/-M stats (clickable in chat)."""
        if not self._modified_files:
            return []
        t = self._t
        lines = []
        for entry in self._modified_files:
            if isinstance(entry, str):
                lines.append(f"📄 {_escape(entry)}")
                continue
            path = entry.get("path", "")
            added = entry.get("added", 0)
            removed = entry.get("removed", 0)
            stats = ""
            if added:
                stats += f"[{t.success}]+{added}[/{t.success}]"
            if removed:
                stats += f"[{t.error}]-{removed}[/{t.error}]"
            if not stats:
                stats = "[dim]0[/dim]"
            lines.append(f"📄 {_escape(path)} {stats}")
        return lines

    def _render_tool_summary(self) -> str:
        t = self._t
        if not self._last_tool_calls:
            return ""
        seen = list(dict.fromkeys(self._last_tool_calls))
        from agent.ui.render import tool_icon as _ti
        parts = []
        for name in seen:
            s = self._tool_stats.get(name, {"ok": 0, "err": 0})
            counts = []
            if s["ok"]:
                counts.append(f"[{t.success}]{s['ok']}[/{t.success}]")
            if s["err"]:
                counts.append(f"[{t.error}]{s['err']}[/{t.error}]")
            suffix = f" {' '.join(counts)}" if counts else ""
            parts.append(f"[{t.tool_color}]{_ti(name)}[/{t.tool_color}] {_escape(name)}{suffix}")
        return ", ".join(parts)

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
        self._title_task_label = event.label

    _STREAM_RENDER_INTERVAL = 0.05  # 50ms throttle

    def _stream_tail(self, buf: list) -> str:
        return "".join(buf)[-800:]

    def _flush_reasoning(self) -> None:
        """Write folded reasoning summary to chat log and clear stream-view."""
        if not self._reasoning_buffer:
            return
        total_chars = sum(len(s) for s in self._reasoning_buffer)
        t = self._t
        self._write_chat(
            f"[{t.thinking_color}]▶ thinking ({total_chars:,} chars)[/{t.thinking_color}]"
        )
        self._reasoning_buffer = []
        self._reasoning_active = False
        stream_view = self.query_one("#stream-view")
        stream_view.remove_class("active")
        stream_view.update("")
        self._streaming_active = False
        self._stream_last_render = 0.0

    def on_reasoning_token_event(self, event) -> None:
        from rich.text import Text
        t = self._t
        self._reasoning_buffer.append(event.token)
        self._reasoning_active = True
        now = time.monotonic()
        if now - self._stream_last_render < self._STREAM_RENDER_INTERVAL:
            return
        self._stream_last_render = now
        stream_view = self.query_one("#stream-view")
        if not self._streaming_active:
            self._streaming_active = True
            stream_view.add_class("active")
        tail = self._stream_tail(self._reasoning_buffer)
        content = Text.assemble(("thinking:", "dim italic"), (f" {tail}▌", "dim italic"))
        stream_view.update(content)

    def on_token_stream_event(self, event) -> None:
        from rich.text import Text
        t = self._t
        if self._reasoning_active:
            self._reasoning_active = False
            fold_mode = self._server.get_ui_config()["reasoning_fold"]
            if fold_mode == "immediate":
                self._flush_reasoning()
        self._stream_buffer.append(event.token)
        now = time.monotonic()
        if now - self._stream_last_render < self._STREAM_RENDER_INTERVAL:
            return
        self._stream_last_render = now
        stream_view = self.query_one("#stream-view")
        if not self._streaming_active:
            self._streaming_active = True
            stream_view.add_class("active")
        tail = self._stream_tail(self._stream_buffer)
        content = Text.assemble(("Agent:", f"bold {t.agent_color}"), (f" {tail}▌",))
        stream_view.update(content)

    def _turn_cleanup(self, event) -> None:
        from textual.worker import WorkerState
        t = self._t
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None
        if self._title_spinner_timer is not None:
            self._title_spinner_timer.stop()
            self._title_spinner_timer = None
        self.query_one("#loading-row").remove_class("active")
        self._agent_running = False
        self.query_one("#input-bar", self._wt.PromptInput).focus()
        if getattr(self, "_terminal_title", "auto") != "off":
            icon = getattr(self, "_title_icon", "🌟")
            suffix = self._session_title_suffix() if hasattr(self, "_session_title_suffix") else ""
            if event.state == WorkerState.ERROR:
                self._set_terminal_title(f"{icon} agent — error, waiting for input{suffix}")
            elif event.state == WorkerState.SUCCESS:
                self._set_terminal_title(f"{icon} agent — waiting for input{suffix}")
            else:
                self._set_terminal_title(f"{icon} agent — cancelled{suffix}")
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            if getattr(self, "_bell_on_input_request", True):
                self.bell()
        # Compute diff stats for modified files (convert list[str] → list[dict]).
        if self._modified_files:
            raw_paths = [f if isinstance(f, str) else f.get("path", "") for f in self._modified_files]
            raw_paths = [p for p in raw_paths if p]
            self._modified_files = _compute_file_diffs(raw_paths)
            # Write file change summary to sys view
            t = self._t
            file_lines = []
            for entry in self._modified_files:
                if isinstance(entry, str):
                    file_lines.append(f"  {_escape(entry)}")
                    continue
                p = entry.get("path", "")
                a = entry.get("added", 0)
                r = entry.get("removed", 0)
                stat = f"[{t.success}]+{a}[/{t.success}] [{t.error}]-{r}[/{t.error}]" if (a or r) else ""
                file_lines.append(f"  {_escape(p)} {stat}")
            self._write_sys(f"[{t.text_dim}]Files changed ({len(self._modified_files)}):[/{t.text_dim}]\n" + "\n".join(file_lines), switch_tab=False)

    def _turn_extract_response(self, event) -> "tuple[str | None, bool, bool]":
        from textual.worker import WorkerState
        t = self._t
        empty_response = False
        is_error = False
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
                if err else ""
            )
            logger.error("chat worker error: %s\n%s", err, tb)
            response = f"[{t.error}]Error: {_escape(str(err))}[/{t.error}]"
            is_error = True
        else:
            logger.info("chat worker cancelled")
            response = None
            if self._session is not None:
                try:
                    self._server.save_session(self._session)
                except Exception:
                    logger.exception("save_session on cancel failed")
        return response, empty_response, is_error

    def _turn_update_context_panel(self) -> None:
        t = self._t
        tokens_after = self._server.token_estimate()
        delta = tokens_after - self._tokens_before
        tools_line = self._render_tool_summary()
        token_line = (
            f"[{t.text_dim}]sent ≈{self._tokens_before:,}  "
            f"[{t.active}]+{delta:,}[/{t.active}] new  "
            f"total {tokens_after:,}[/{t.text_dim}]"
        )
        s = self._server.stats()
        if s and s.get("calls", 0) > 0:
            extras = [f"↑{s['input_tokens']:,}", f"↓{s['output_tokens']:,}"]
            if s.get("in_tps"):
                extras.append(f"{_fmt_tps(s['in_tps'])} in-tok/s")
            if s.get("out_tps"):
                extras.append(f"{_fmt_tps(s['out_tps'])} out-tok/s")
            if s.get("reasoning_tokens"):
                extras.append(f"think {s['reasoning_tokens']:,}")
            if s.get("tool_tokens"):
                extras.append(f"tool {s['tool_tokens']:,}")
            token_line += f"\n[{t.text_dim}]{'  '.join(extras)}[/{t.text_dim}]"
        self.query_one("#context-panel", self._wt.ContextPanel).set_context(
            f"{tools_line}\n{token_line}" if tools_line else token_line
        )

    def _turn_flush_streaming(self) -> None:
        fold_mode = self._server.get_ui_config()["reasoning_fold"]
        if self._reasoning_buffer and fold_mode != "never":
            self._flush_reasoning()
        if self._streaming_active:
            stream_view = self.query_one("#stream-view")
            stream_view.remove_class("active")
            stream_view.update("")
            self._streaming_active = False
            self._stream_buffer = []
            self._stream_last_render = 0.0

    def _turn_write_chat(self, response: "str | None", empty_response: bool, is_error: bool = False) -> None:
          from rich.markdown import Markdown as _Markdown
          from agent.ui.render import _delatex
          t = self._t
          chat_log = self.query_one("#chat-log", self._wt.ConversationView)
          file_lines = self._render_file_diffs()
          if self._last_tool_calls:
              tool_part = self._render_tool_summary()
              self._write_chat(f"  {tool_part}")
          if file_lines:
              for idx, line_text in enumerate(file_lines):
                  before = len(chat_log.lines)
                  self._write_chat(f"  {line_text}")
                  after = len(chat_log.lines)
                  entry = self._modified_files[idx] if idx < len(self._modified_files) else None
                  if entry is not None:
                      for li in range(before, after):
                          self._chat_file_lines[li] = entry
          if response:
              if empty_response:
                  self._write_chat(
                      f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] "
                      f"[{t.text_dim}]{_escape(response)}[/{t.text_dim}]"
                  )
              elif is_error:
                  # response carries Rich markup ([color]…[/color]); write it
                  # directly so the console renders the color. The Markdown path
                  # below would print the markup tags as literal text.
                  self._write_chat(
                      f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {response}"
                  )
              else:
                  self._write_chat(f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]")
                  self._write_chat(_Markdown(_delatex(response)))

    def on_worker_state_changed(self, event) -> None:
        from textual.worker import WorkerState
        if event.worker.name != "chat":
            return
        if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return
        self._turn_cleanup(event)
        response, empty_response, is_error = self._turn_extract_response(event)
        self._turn_update_context_panel()
        self._turn_flush_streaming()
        self._turn_write_chat(response, empty_response, is_error)
        if event.state == WorkerState.SUCCESS:
            if self._round_summary_enabled:
                self._write_round_summary(getattr(self, "_current_user_text", ""), response or "")
            self._append_qa_turn(getattr(self, "_current_user_text", ""), response or "")
        self._refresh_token_bar()
        self.call_later(self._refresh_git)

    def on_tabbed_content_tab_activated(self, event) -> None:
        mode = self._server.get_ui_config().get("qa_summary_mode", "lazy")
        if mode != "lazy":
            return
        # ContentTab IDs are prefixed: "--content-tab-<pane-id>"
        pane_id = (getattr(event.tab, "id", "") or "").removeprefix("--content-tab-")
        if pane_id not in ("tab-q-summary", "tab-a-summary"):
            return
        if getattr(self, "_qa_summary_dirty", False) or not self._any_cached_summary():
            self._start_qa_summary_worker()

    def _any_cached_summary(self) -> bool:
        try:
            from agent.memory.session import _get_session_dir, get_session_subpath
            from agent.memory.session_summarizer import load_stored
            if not self._session:
                return False
            sd = _get_session_dir() / get_session_subpath(self._session.id)
            return bool(load_stored(sd, "q").get("content") or load_stored(sd, "a").get("content"))
        except Exception:
            return False
