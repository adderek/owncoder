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
from agent.ui.render import (
    _CTX_SEGMENT_COLORS,
    _CTX_SEGMENT_DESCS,
    _OUT_SEGMENT_COLORS,
    _OUT_SEGMENT_DESCS,
    _mini_bar,
    _render_context_report,
)


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

    from agent.ui.textual_widgets import build_widget_classes
    _w = build_widget_classes(t, agent)
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
    SparseView = _w.SparseView
    ContextPanel = _w.ContextPanel
    GitStatusBar = _w.GitStatusBar
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
        }}
        ContextBreakdownBar {{
            height: 1;
        }}
        OutputBreakdownBar {{
            height: 1;
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

            self._round_summary_enabled = bool(
                getattr(self._agent.config.ui, "round_summary", True)
            )
            try:
                from agent.ui.prefs import load_prefs

                _p = load_prefs()
                if "round_summary" in _p:
                    self._round_summary_enabled = bool(_p.get("round_summary"))
            except Exception:
                pass

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

        def _write_round_summary(self, user_text: str, response: str) -> None:
            """Write a gray one-line Q/A summary of the just-completed round."""
            q = _one_line((user_text or "").strip(), limit=80, wrap=False)
            a_src = (response or "").strip()
            a = _one_line(a_src, limit=80, wrap=False) if a_src else ""
            tools = list(dict.fromkeys(self._last_tool_calls))
            action_bits: list[str] = []
            if tools:
                action_bits.append("⚙ " + ", ".join(tools[:4]))
            if self._modified_files:
                action_bits.append("✎ " + " ".join(self._modified_files[:4]))
            action = " · ".join(action_bits)
            if action and a:
                a_part = f"{action} — {a}"
            else:
                a_part = action or a or "(no reply)"
            a_part = _one_line(a_part, limit=140, wrap=False)
            line = (
                f"[{t.text_dim}]↳ Q:[/{t.text_dim}] [{t.text_dim}]{_escape(q)}[/{t.text_dim}]  "
                f"[{t.text_dim}]A:[/{t.text_dim}] [{t.text_dim}]{_escape(a_part)}[/{t.text_dim}]"
            )
            self._write_chat(line)

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
                from agent.ui.prefs import load_prefs, save_prefs

                self._wrap_enabled = not self._wrap_enabled
                state = "enabled" if self._wrap_enabled else "disabled"
                _p = load_prefs()
                _p["chat_wrap"] = "wrap" if self._wrap_enabled else "nowrap"
                save_prefs(_p)
                self._write_sys(f"[{t.success}]Line wrapping {state}.[/{t.success}]")
                # Refresh views to apply the new wrapping setting
                self.query_one("#chat-log", ConversationView).clear()
                if self._agent.messages:
                    self._restore_chat_history(self._agent.messages)
                self._reload_qa_views()

            elif cmd in ("/round-summary", "/summary"):
                from agent.ui.prefs import load_prefs, save_prefs

                self._round_summary_enabled = not self._round_summary_enabled
                state = "enabled" if self._round_summary_enabled else "disabled"
                p = load_prefs()
                p["round_summary"] = self._round_summary_enabled
                save_prefs(p)
                self._write_sys(
                    f"[{t.success}]Round summary {state}.[/{t.success}]"
                )

            elif cmd == "/plan":
                ok, msg = _apply_plan(self._agent, arg)
                color = t.success if ok else t.warning
                for line in msg.splitlines():
                    self._write_sys(f"[{color}]{line}[/{color}]")

            elif cmd == "/plans":
                from agent.planning import list_plans
                plans = list_plans()
                if not plans:
                    self._write_sys(f"[{t.text_dim}]No plans.[/{t.text_dim}]")
                else:
                    for p in plans:
                        done, total = p.progress()
                        self._write_sys(
                            f"[{t.success}]{p.id}[/{t.success}] "
                            f"[dim]({p.status}, {done}/{total})[/dim] {p.goal[:80]}"
                        )

            elif cmd in ("/abort-plan", "/pause-plan", "/stash-plan"):
                sub_map = {
                    "/abort-plan": "abort",
                    "/pause-plan": "pause",
                    "/stash-plan": "stash",
                }
                ok, msg = _apply_plan(self._agent, sub_map[cmd])
                color = t.success if ok else t.warning
                for line in msg.splitlines():
                    self._write_sys(f"[{color}]{line}[/{color}]")

            elif cmd in ("/quit", "/exit", "/q!"):
                self.action_quit()
                return

            elif cmd == "/recoveries":
                from agent.planning import recovery
                recs = recovery.scan_pending()
                if not recs:
                    self._write_sys(f"[{t.text_dim}]No pending crash recoveries.[/{t.text_dim}]")
                else:
                    for r in recs:
                        self._write_sys(
                            f"[{t.warning}]{r.session_id}[/{t.warning}] "
                            f"[dim]{r.exception}[/dim]"
                        )

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
                if self._round_summary_enabled:
                    self._write_round_summary(
                        getattr(self, "_current_user_text", ""), response or ""
                    )
                self._append_qa_turn(
                    getattr(self, "_current_user_text", ""), response or ""
                )

            self._refresh_token_bar()
            self.call_later(self._refresh_git)

    return CodeAgentApp(agent, session=session)


from agent.ui.readline_loop import (
    simple_loop, _make_help_text, _token_bar,
    _spinner_status_fields, _run_spinner, _hex_to_ansi,
)




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
