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


def _changeset_of(app):
    """The core.changeset.Changeset the agent produced for the round just
    finished, or None.

    Source of truth is core/checkpoint.py's edit journal (see core/changeset.py),
    not ``git diff`` — the journal gives exact per-round attribution, including
    newly created files and foreign edits, which git cannot. Not every server
    exposes ``last_changeset`` (a relay-backed server may not), so this looks
    at the server first and falls back to the wrapped agent, returning None
    rather than raising when neither has it.
    """
    server = getattr(app, "_server", None)
    if server is None:
        return None
    cs = getattr(server, "last_changeset", None)
    if cs is not None:
        return cs
    return getattr(getattr(server, "_agent", None), "last_changeset", None)


def changeset_file_dict(fc, turn_id: int = 0) -> dict:
    """Plain-dict view of one core.changeset.FileChange.

    Click-map storage (``_chat_file_lines``) and the modal screens in
    textual_widgets.py work off this shape rather than the dataclass, so a
    restored/legacy entry (already a dict) and a live one look the same to
    them. ``turn_id`` rides along so a spilled (oversized) diff can be found
    again later from the session's changesets directory.
    """
    from dataclasses import asdict
    d = asdict(fc)
    d["note"] = fc.note()
    d["turn_id"] = turn_id
    return d


def _file_row_markup(t, fc) -> str:
    mark = {"added": "+", "deleted": "-"}.get(fc.status, "~")
    if fc.binary:
        stat = "[dim]binary[/dim]"
    else:
        parts = []
        if fc.added:
            parts.append(f"[{t.success}]+{fc.added}[/{t.success}]")
        if fc.removed:
            parts.append(f"[{t.error}]-{fc.removed}[/{t.error}]")
        stat = " ".join(parts) if parts else "[dim]0[/dim]"
    trunc = "  [dim](diff too large)[/dim]" if fc.truncated else ""
    return f"📄 {mark} {_escape(fc.path)}  {stat}{trunc}"


def _file_note_markup(t, fc) -> "str | None":
    """A foreign_edit's warning, in the theme's warning colour — visible, not
    silently dropped, since it means someone other than the agent wrote the
    file too."""
    note = fc.note()
    if not note:
        return None
    return f"  [{t.warning}]⚠ {_escape(note)}[/{t.warning}]"


def _changeset_sys_lines(t, cs) -> "list[str]":
    """Plain colourised lines for the sys-view log, honouring the changeset's
    tier the same way the chat log does: count → headline only, list →
    headline + rows, inline → headline + rows + diffs."""
    if not cs:
        return []
    head = _escape(cs.headline()) + (" (diffs truncated)" if cs.truncated else "")
    lines = [f"[{t.text_dim}]{head}[/{t.text_dim}]"]
    if cs.tier == "count":
        return lines
    for fc in cs.files:
        lines.append(f"  {_file_row_markup(t, fc)}")
        note = _file_note_markup(t, fc)
        if note:
            lines.append(note)
        if cs.tier == "inline" and fc.diff:
            lines.extend(f"    {_escape(l)}" for l in fc.diff.rstrip("\n").splitlines())
    return lines


def session_rollup_line(app) -> str:
    """``session: 7 files changed, +120 -33`` for the whole session, or "".

    Pinned under the round's changeset block so the answer to "what has this
    session done so far" does not require scrolling back through every round.
    The rounds come from ``core.changeset.SessionRollup``, which reads the QA
    log — a session resumed from disk therefore rolls up the rounds it never
    watched live, not just the ones since the reload. Empty when the rollup is
    switched off (``ui.changeset.session_rollup``) or nothing changed yet.
    """
    from agent.core.changeset import SessionRollup, session_rollup_enabled
    try:
        cfg = getattr(getattr(getattr(app, "_server", None), "_agent", None), "config", None)
        if cfg is not None and not session_rollup_enabled(cfg):
            return ""
        sid = getattr(getattr(app, "_session", None), "id", "") or ""
        rollup = getattr(app, "_session_rollup", None)
        if rollup is None or rollup.session_id != sid:
            rollup = SessionRollup(sid)
            app._session_rollup = rollup
        rollup.add(getattr(app, "_last_changeset", None))
        return rollup.line()
    except Exception:
        logger.debug("textual ui: session rollup failed", exc_info=True)
        return ""


def write_changeset_rows(app, _cw, chat_log, cs, rollup: str = "") -> None:
    """Write *cs* to the chat log via *_cw*, tiered, registering click targets.

    Shared by the live end-of-turn write (event_mixin._turn_write_chat) and by
    history re-render (view_mixin._render_full_turn), so a restored turn looks
    identical to how it looked live. ``count`` writes only the headline and
    registers it in ``app._chat_changeset_lines`` (click → file list); ``list``
    adds one row per file registered in ``app._chat_file_lines`` (click →
    diff); ``inline`` also writes each file's diff, already fully visible.

    *rollup* is the session-total footer line (see session_rollup_line). The
    live path passes one; history re-render does not, since a rollup printed
    under an old round would state a total that was not true when it ended.
    """
    if not cs:
        return
    t = app._t
    if not hasattr(app, "_chat_changeset_lines"):
        app._chat_changeset_lines = {}
    header = cs.headline() + (" (diffs truncated)" if cs.truncated else "")
    before = len(chat_log.lines)
    _cw(f"  [{t.text_dim}]{header}[/{t.text_dim}]")
    if cs.tier == "count":
        for li in range(before, len(chat_log.lines)):
            app._chat_changeset_lines[li] = cs
        if rollup:
            _cw(f"  [{t.text_dim}]{_escape(rollup)}[/{t.text_dim}]")
        return
    for fc in cs.files:
        row_before = len(chat_log.lines)
        _cw(f"  {_file_row_markup(t, fc)}")
        note = _file_note_markup(t, fc)
        if note:
            _cw(note)
        entry = changeset_file_dict(fc, cs.turn_id)
        for li in range(row_before, len(chat_log.lines)):
            app._chat_file_lines[li] = entry
        if cs.tier == "inline" and fc.diff:
            for l in fc.diff.rstrip("\n").splitlines():
                _cw(f"    {_escape(l)}")
    if rollup:
        _cw(f"  [{t.text_dim}]{_escape(rollup)}[/{t.text_dim}]")


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

    def on_injected_message_event(self, event) -> None:
        """A turn's own note, written where the user is reading.

        These are stored as user messages, so a reloaded session has always
        shown them — but live the turn only flashed a phase label, and a round
        that ended on a failing verify read as finished.
        """
        t = self._t
        kind = _escape(getattr(event, "kind", "") or "agent")
        self._write_chat(f"[{t.text_dim}]⚙ {kind}[/{t.text_dim}]")
        for line in (event.text or "").splitlines() or [""]:
            self._write_chat(f"[{t.text_dim}]{_escape(line)}[/{t.text_dim}]")

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
        # Read the changeset the agent already produced for this round (source
        # of truth: core/checkpoint.py's edit journal, bounded to the round —
        # see core/changeset.py). Not every server exposes one (a relay-backed
        # server may not), so this degrades to showing nothing rather than
        # crashing or falling back to a `git diff` that would report every
        # uncommitted change instead of just this round's.
        cs = _changeset_of(self)
        self._last_changeset = cs
        if cs:
            self._write_sys("\n".join(_changeset_sys_lines(t, cs)), switch_tab=False)

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
            try:
                cfg = getattr(getattr(self._server, "_agent", None), "config", None)
                if cfg is not None:
                    from agent.metrics.model_calls import session_cost_usd
                    cost = session_cost_usd(cfg)
                    if cost > 0:
                        extras.append(f"${cost:.3f}" if cost < 1 else f"${cost:.2f}")
            except Exception:
                logger.debug("textual ui: cost estimate failed", exc_info=True)
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
          if self._last_tool_calls:
              tool_part = self._render_tool_summary()
              self._write_chat(f"  {tool_part}")
          write_changeset_rows(self, self._write_chat, chat_log,
                               getattr(self, "_last_changeset", None),
                               session_rollup_line(self))
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
            try:
                from agent.metrics import model_calls
                duration = model_calls.round_duration()
                _mc = model_calls.format_line(
                    model_calls.round_counts(), duration=duration)
                if _mc:
                    # Remember which chat-log lines hold this round line so a
                    # click can open the per-call role × model detail modal.
                    detail = model_calls.round_detail()
                    chat_log = self.query_one("#chat-log", self._wt.ConversationView)
                    before = len(chat_log.lines)
                    self._write_chat(
                        f"[{self._t.text_dim}]{_mc}"
                        f"  [i]▸ click for detail[/i][/{self._t.text_dim}]")
                    if detail:
                        try:
                            stats = self._server.stats()
                        except Exception:
                            stats = {}
                        payload = {"calls": detail, "duration": duration,
                                   "stats": stats}
                        for li in range(before, len(chat_log.lines)):
                            self._chat_model_lines[li] = payload
            except Exception:
                pass
            self._append_qa_turn(getattr(self, "_current_user_text", ""), response or "")
            self._continue_prompt_loop()
        else:
            # Aborted/failed turn kills an active /loop.
            p_loop = getattr(self, "_prompt_loop", None)
            if p_loop is not None and p_loop.active:
                p_loop.stop()
                self._write_sys("↻ loop stopped (turn aborted or failed).")
        self._refresh_token_bar()
        self.call_later(self._refresh_git)

    def _continue_prompt_loop(self) -> None:
        """After a successful /loop turn: count it and schedule the next one."""
        p_loop = getattr(self, "_prompt_loop", None)
        if p_loop is None or not p_loop.active:
            return
        if not p_loop.record_iteration():
            self._write_sys(f"↻ loop finished: {p_loop.done} iteration(s).")
            return
        note = (f"↻ loop iteration {p_loop.done} done — next in "
                f"{int(p_loop.interval)}s" if p_loop.interval
                else f"↻ loop iteration {p_loop.done} done — next immediately")
        if p_loop.limit:
            note += f" ({p_loop.done}/{p_loop.limit})"
        self._write_sys(note)

        def _fire() -> None:
            self._loop_timer = None
            if p_loop.active and not getattr(self, "_agent_running", False):
                self._begin_chat(p_loop.prompt)

        if p_loop.interval:
            self._loop_timer = self.set_timer(p_loop.interval, _fire)
        else:
            self.call_later(_fire)

    def on_tabbed_content_tab_activated(self, event) -> None:
        self._save_ui_state()
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
