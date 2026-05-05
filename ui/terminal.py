from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent.core.agent import Agent

from agent.ui.slash import (
    _SLASH_COMMANDS,
    _apply_think,
    _apply_temperature,
    _apply_max_tokens,
    _active_plan,
    _render_plan,
    _apply_plan,
    _match_commands,
)

# ── Textual UI ──────────────────────────────────────────────────────────────


def _build_textual_app(agent: "Agent", session=None, server=None):
    from agent.ui_server import LocalUIServer
    if server is None:
        server = LocalUIServer(agent)
    t = server.get_ui_config()["theme"]

    import os
    session_name = f'{session.id} {session.description}' if session else 'No Session'
    os.system(f'echo -ne "\033]0;🌟 {session_name}\007"')

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

    from agent.ui.textual_widgets import build_widget_classes
    from agent.ui_server import LocalUIServer
    if server is None:
        server = LocalUIServer(agent)
    t = server.get_ui_config()["theme"]

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

    from agent.ui.textual_widgets import build_widget_classes
    _w = build_widget_classes(t)

    # Import mixins inside factory so ImportError propagates if Textual absent
    from agent.ui.app_css import build_app_css
    from agent.ui.view_mixin import ViewMixin
    from agent.ui.slash_mixin import SlashHandlerMixin
    from agent.ui.event_mixin import EventHandlerMixin

    # Unpack widget-type aliases used directly in this file
    _one_line = _w._one_line
    TokenBar = _w.TokenBar
    ContextBreakdownBar = _w.ContextBreakdownBar
    OutputBreakdownBar = _w.OutputBreakdownBar
    ConversationView = _w.ConversationView
    SysView = _w.SysView
    JumpToTurn = _w.JumpToTurn
    _QALineTrackingMixin = _w._QALineTrackingMixin
    QView = _w.QView
    AView = _w.AView
    QSummaryView = _w.QSummaryView
    ASummaryView = _w.ASummaryView
    SparseView = _w.SparseView
    ContextPanel = _w.ContextPanel
    GitStatusBar = _w.GitStatusBar
    ModelStatusBar = _w.ModelStatusBar
    HintBar = _w.HintBar
    CompletionBar = _w.CompletionBar
    PromptInput = _w.PromptInput
    ToolCallEvent = _w.ToolCallEvent
    ToolResultEvent = _w.ToolResultEvent
    TokenStreamEvent = _w.TokenStreamEvent
    IterationProgressEvent = _w.IterationProgressEvent
    PhaseEvent = _w.PhaseEvent
    ReasoningTokenEvent = _w.ReasoningTokenEvent
    ContextSizeEvent = _w.ContextSizeEvent
    _PLACEHOLDER_Q = _w._PLACEHOLDER_Q
    _PLACEHOLDER_A = _w._PLACEHOLDER_A
    _PLACEHOLDER_SPARSE = _w._PLACEHOLDER_SPARSE

    class CodeAgentApp(SlashHandlerMixin, EventHandlerMixin, ViewMixin, App):
        CSS = build_app_css(t)

        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+d", "quit", "Quit", show=False),
            Binding("f1", "show_help", "Help"),
            Binding("ctrl+r", "continue_turn", "Continue"),
            Binding("ctrl+c", "interrupt_turn", "Stop", show=False),
            Binding("ctrl+tab", "focus_next", "Switch focus", show=False),
        ]

        def __init__(self, agent: "Agent", session=None, server=None, **kwargs):
            super().__init__(**kwargs)
            self._server = server
            self._session = session
            self._t = t
            self._wt = _w
            self._last_tool_calls: list[str] = []
            self._current_tool: str | None = None
            self._tokens_before: int = 0
            self._streaming_active: bool = False
            self._stream_buffer: list[str] = []
            self._stream_last_render: float = 0.0
            self._modified_files: list[str] = []
            self._loading_timer = None
            self._iter_done: int = 0
            self._iter_limit: int = 0
            self._stop_requested: bool = False
            self._chat_worker = None
            self._quit_requested: bool = False
            self._chat_user_lines: list[int] = []
            self._sys_messages: list[str] = []
            self._tool_stats: dict[str, dict[str, int]] = {}
            self._agent_running: bool = False
            self._reasoning_buffer: list[str] = []
            self._reasoning_active: bool = False
            self._title_task_label: str = ""
            self._title_spinner_idx: int = 0
            self._title_spinner_timer = None

            ui_cfg = self._server.get_ui_config()
            chat_wrap_cfg = ui_cfg["chat_wrap"]
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

            self._rating_prompted: bool = False
            self._bell_on_input_request: bool = ui_cfg.get("bell_on_input_request", True)
            self._terminal_title: str = ui_cfg.get("terminal_title", "auto")
            self._round_summary_enabled = ui_cfg["round_summary"]
            try:
                from agent.ui.prefs import load_prefs
                _p = load_prefs()
                if "round_summary" in _p:
                    self._round_summary_enabled = bool(_p.get("round_summary"))
            except Exception:
                pass

            if session is not None:
                self._server.set_session_id(session.id)

        def compose(self) -> ComposeResult:
            _info = self._server.get_llm_info()
            session_label = ""
            if self._session:
                label = self._session.short_name or self._session.id
                session_label = f"  [{t.text_dim}]{label}[/{t.text_dim}]"
            with Horizontal(id="header-bar"):
                yield Static(
                    f"[bold]local-code-agent[/bold]  [{t.text_dim}]{_info['model']}[/{t.text_dim}]{session_label}",
                    id="header-title",
                )
                yield ModelStatusBar("", id="model-status")
            yield TokenBar(
                _info["ctx_window"],
                compact_frac=_info["compaction_threshold"],
                id="token-bar",
            )
            yield ContextBreakdownBar(_info["ctx_window"], id="context-breakdown")
            yield OutputBreakdownBar(id="output-breakdown")
            with TabbedContent(initial="tab-chat", id="view-tabs"):
                with TabPane("chat", id="tab-chat"):
                    yield ConversationView(id="chat-log", markup=True, highlight=False)
                    yield Static("", id="stream-view", markup=True)
                with TabPane("Q", id="tab-q"):
                    yield QView(id="q-log", markup=True, highlight=False)
                with TabPane("Q Summary", id="tab-q-summary"):
                    yield QSummaryView(id="q-summary-log", markup=True, highlight=False)
                with TabPane("A", id="tab-a"):
                    yield AView(id="a-log", markup=True, highlight=False)
                with TabPane("A Summary", id="tab-a-summary"):
                    yield ASummaryView(id="a-summary-log", markup=True, highlight=False)
                with TabPane("sparse", id="tab-sparse"):
                    yield SparseView(id="sparse-log", markup=True, highlight=False)
                with TabPane("sys", id="tab-sys"):
                    yield SysView(id="sys-log", markup=True, highlight=True)
            with Horizontal(id="loading-row"):
                yield LoadingIndicator(id="loading-indicator")
                yield Static("", id="loading-tokens", markup=True)
            from textual.widgets import Button as _Button
            with Horizontal(id="rating-bar"):
                yield Static("Rate:", id="rating-label", markup=False)
                yield _Button("👍 Good", id="btn-rate-good")
                yield _Button("👎 Bad", id="btn-rate-bad")
                yield _Button("skip", id="btn-rate-skip")
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
            info = self._server.get_llm_info()
            ctx = info["ctx_window"]
            peak, _ = self._server.get_peak_tokens()
            bar.update_tokens(
                self._server.token_estimate(),
                peak=peak,
                compact_frac=info["compaction_threshold"],
                ctx_window=ctx,
            )
            try:
                breakdown = self.query_one("#context-breakdown", ContextBreakdownBar)
                breakdown.set_segments(self._server.context_breakdown(), ctx_window=ctx)
            except Exception:
                pass
            try:
                out_bar = self.query_one("#output-breakdown", OutputBreakdownBar)
                out_bar.set_segments(self._server.output_breakdown("session"), scope_label="out")
            except Exception:
                pass

        def on_mount(self) -> None:
            self.query_one("#input-bar", PromptInput).focus()
            self.call_later(self._refresh_git)
            self.call_later(self._refresh_token_bar)
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
            sys_log.write(f"[{t.text_dim}]Type /help for commands  ·  F1 opens this tab[/{t.text_dim}]")
            msgs = self._server.get_messages()
            if msgs:
                self._restore_chat_history(msgs)
            self._reload_qa_views()

        # ── actions ──────────────────────────────────────────────────────────

        def action_quit(self) -> None:
            try:
                from agent.tools.analyze_asm import get_interrupt_flag
                get_interrupt_flag().set()
            except Exception:
                pass
            if self._quit_requested:
                try:
                    self._server.cancel_background()
                except Exception:
                    pass
                self.exit()
                return
            # Prompt for session rating before quitting if unrated and has turns.
            if (
                not self._rating_prompted
                and self._session is not None
                and getattr(self._session, "user_outcome", None) is None
                and self._server.get_turn_id() > 0
                and not self._agent_running
            ):
                self._rating_prompted = True
                self._show_rating_bar("Rate session:")
                return
            pending = 0
            try:
                pending = self._server.pending_background_count()
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
                await self._server.wait_background(timeout=30.0)
            except Exception:
                logger.exception("graceful_exit: wait_background error (ignored)")
            self.exit()

        def action_show_help(self) -> None:
            from agent.ui.readline_loop import _make_help_text
            self._write_sys(_make_help_text(t))

        def action_continue_turn(self) -> None:
            if self._agent_running:
                return
            input_widget = self.query_one("#input-bar", PromptInput)
            if input_widget.disabled:
                return
            self._begin_chat("continue")

        def action_interrupt_turn(self) -> None:
            if not self._agent_running:
                return
            if not self._stop_requested:
                self._stop_requested = True
                self._server.stop_after_iteration()
                self._write_sys(
                    f"[{t.warning}]Stopping after current iteration… "
                    f"Press Ctrl+C again to cancel immediately.[/{t.warning}]"
                )
            else:
                if self._chat_worker is not None:
                    self._chat_worker.cancel()
                self._write_sys(f"[{t.error}]Cancelled.[/{t.error}]")

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
            if self._agent_running:
                if user_text.startswith("/") or user_text.lower() == "continue":
                    self._write_sys(
                        f"[{t.warning}]Agent running — slash commands and 'continue' "
                        f"not accepted mid-turn. Text messages are injected.[/{t.warning}]"
                    )
                    return
                self._server.inject(user_text)
                self._write_chat(
                    f"[bold {t.user_color}]↑ You (mid-turn):[/bold {t.user_color}] {_escape(user_text)}"
                )
                return
            if user_text.startswith("/"):
                parts = user_text.split(None, 1)
                await self._run_slash(parts[0].lower(), parts[1] if len(parts) > 1 else "")
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
            if event.remove_count > 0:
                _msgs = self._server.get_messages()
                user_positions = [
                    i for i, m in enumerate(_msgs) if m.get("role") == "user"
                ]
                if event.remove_count >= len(user_positions):
                    system = next(
                        (m for m in _msgs if m.get("role") == "system"), None
                    )
                    self._server.set_messages([system] if system else [])
                else:
                    cut = user_positions[-event.remove_count]
                    self._server.set_messages(_msgs[:cut])
                self._refresh_token_bar()
            input_widget = self.query_one("#input-bar", PromptInput)
            if event.area._edit_source_idx is not None:
                input_widget._history = input_widget._history[: event.area._edit_source_idx]
            input_widget.add_to_history(user_text)
            self._begin_chat(user_text)

        def _update_loading_tokens(self) -> None:
            try:
                est = self._server.token_estimate()
                rcvd = max(0, est - self._tokens_before)
                text = f"in: [bold]{self._tokens_before:,}[/bold]  out: +{rcvd:,}"
                if self._iter_limit == -1:
                    stop_hint = "  [bold yellow]⏹ stopping…[/bold yellow]" if self._stop_requested else ""
                    text += f"  iter {self._iter_done}/∞{stop_hint}"
                elif self._iter_limit > 0:
                    left = max(0, self._iter_limit - self._iter_done)
                    text += f"  iter {self._iter_done}/{self._iter_limit} ({left} left)"
                self.query_one("#loading-tokens", Static).update(text)
            except Exception:
                pass

        _TITLE_SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def _tick_title_spinner(self) -> None:
            s = self._TITLE_SPINNERS[self._title_spinner_idx % len(self._TITLE_SPINNERS)]
            self._title_spinner_idx += 1
            label = self._title_task_label or "working"
            self.title = f"{s} agent — {label}"

        def _begin_chat(self, user_text: str) -> None:
            self._hide_rating_bar()
            self._switch_to_chat()
            chat_log = self.query_one("#chat-log", ConversationView)
            self._chat_user_lines.append(len(chat_log.lines))
            self._write_chat(
                f"[bold {t.user_color}]You:[/bold {t.user_color}] {_escape(user_text)}"
            )
            self._last_tool_calls = []
            self._tool_stats = {}
            self._current_tool = None
            self._tokens_before = self._server.token_estimate()
            self._streaming_active = False
            self._stream_buffer = []
            self._stream_last_render = 0.0
            self._modified_files = []
            self._iter_done = 0
            self._iter_limit = 0
            self._stop_requested = False
            self._current_user_text = user_text
            self._reasoning_buffer = []
            self._reasoning_active = False

            self._agent_running = True
            self._title_task_label = "thinking"
            self._title_spinner_idx = 0
            if self._terminal_title != "off":
                if self._title_spinner_timer is not None:
                    self._title_spinner_timer.stop()
                self._title_spinner_timer = self.set_interval(0.1, self._tick_title_spinner)
            self.query_one("#loading-row").add_class("active")
            self.query_one("#loading-tokens", Static).update(
                f"in: [bold]{self._tokens_before:,}[/bold]"
            )
            if self._loading_timer is not None:
                self._loading_timer.stop()
            self._loading_timer = self.set_interval(0.3, self._update_loading_tokens)
            self.query_one("#context-panel", ContextPanel).set_context("[dim]thinking…[/dim]")
            self._chat_worker = self._start_chat(user_text)

        @work(exclusive=True, exit_on_error=False, name="chat")
        async def _start_chat(self, user_text: str) -> str:
            def on_tool(name: str, args: str) -> None:
                self._last_tool_calls.append(name)
                self._current_tool = name
                self._tool_stats.setdefault(name, {"ok": 0, "err": 0})
                self.post_message(ToolCallEvent(name, args))

            def on_tool_result(name: str, ok: bool) -> None:
                self.post_message(ToolResultEvent(name, ok))

            def on_user_message() -> None:
                if self._session is not None:
                    self._server.save_session(self._session)

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

            async def on_loop_detected(summary: str, count: int) -> bool:
                self._write_chat(
                    f"[{t.warning}]⚠ loop guard: repeated tool calls ({summary})."
                    f" Rephrase to redirect.[/{t.warning}]"
                )
                return False

            result = await self._server.chat(
                user_text,
                session_id=self._session.id if self._session else "",
                on_tool_call=on_tool,
                on_tool_result=on_tool_result,
                on_user_message=on_user_message,
                on_token=on_token,
                on_progress=on_progress,
                on_phase=on_phase,
                on_reasoning=on_reasoning,
                on_context_size=on_context_size,
                on_loop_detected=on_loop_detected,
            )
            if self._session is not None:
                self._server.save_session(self._session)
            return result

        def _hide_rating_bar(self) -> None:
            try:
                self.query_one("#rating-bar").remove_class("active")
            except Exception:
                pass

        def _show_rating_bar(self, label: str = "Rate:") -> None:
            try:
                self.query_one("#rating-label").update(label)
                self.query_one("#rating-bar").add_class("active")
            except Exception:
                pass

        def _do_rate_session(self, outcome: str, voter: str = "user") -> None:
            try:
                sid = self._session.id if self._session else ""
                self._server.rate_session(outcome=outcome, voter=voter, session_id=sid)
                if self._session is not None:
                    self._session.user_outcome = outcome
                icon = "👍" if outcome == "good" else ("👎" if outcome == "bad" else "·")
                self._write_chat(
                    f"[{t.text_dim}]{icon} Session rated: {outcome}[/{t.text_dim}]"
                )
            except Exception as exc:
                logger.warning("rate_session failed: %s", exc)

        def on_button_pressed(self, event) -> None:
            bid = event.button.id
            if bid == "btn-rate-good":
                self._hide_rating_bar()
                self._do_rate_session("good")
                if getattr(self, "_rating_prompted", False):
                    self._rating_prompted = False
                    self.action_quit()
            elif bid == "btn-rate-bad":
                self._hide_rating_bar()
                self._do_rate_session("bad")
                if getattr(self, "_rating_prompted", False):
                    self._rating_prompted = False
                    self.action_quit()
            elif bid == "btn-rate-skip":
                self._hide_rating_bar()
                if getattr(self, "_rating_prompted", False):
                    self._rating_prompted = False
                    self.action_quit()

    return CodeAgentApp(agent, session=session, server=server)


from agent.ui.readline_loop import (
    simple_loop, _make_help_text, _token_bar,
)
from agent.ui.spinner import _spinner_status_fields, _run_spinner
from agent.ui.colors import _hex_to_ansi


# ── Entry point ──────────────────────────────────────────────────────────────


def run_ui(agent: "Agent", session=None):
    from agent.ui_server import LocalUIServer
    server = LocalUIServer(agent)
    if server.get_ui_config()["mode"] == "textual":
        try:
            app = _build_textual_app(agent, session=session, server=server)
            app.run()
            return app._session
        except ImportError:
            print("Textual not available, falling back to simple mode.")
            return asyncio.run(simple_loop(agent, session=session, server=server))
    else:
        return asyncio.run(simple_loop(agent, session=session, server=server))
