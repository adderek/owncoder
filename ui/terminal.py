from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.config import Config


# ── Textual UI ──────────────────────────────────────────────────────────────

def _build_textual_app(agent: "Agent", session=None):
    t = agent.config.ui.theme  # shorthand used throughout

    from textual.app import App, ComposeResult
    from textual.widgets import (
        Footer, RichLog, Static, TextArea, LoadingIndicator,
        TabbedContent, TabPane,
    )
    from textual.binding import Binding
    from textual.message import Message
    from textual.worker import Worker, WorkerState
    from textual import work
    from rich.markdown import Markdown

    class TokenBar(Static):
        def __init__(self, ctx_window: int, **kwargs):
            super().__init__(**kwargs)
            self._ctx = ctx_window

        def update_tokens(self, used: int) -> None:
            pct = used / self._ctx if self._ctx else 0
            bar_len = 20
            filled = int(pct * bar_len)
            color = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
            bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"
            self.update(f"tokens: {used}/{self._ctx} {bar}")

    class ConversationView(RichLog):
        """Live chat log — user ↔ agent turns."""

    class SysView(RichLog):
        """System log — commands, session info, help output."""

    class ContextPanel(Static):
        def set_context(self, text: str) -> None:
            self.update(text)

    class GitStatusBar(Static):
        def set_status(self, text: str) -> None:
            self.update(text)

    class PromptInput(TextArea):
        """Multi-line input; Enter submits, Shift+Enter inserts newline."""

        class Submitted(Message):
            def __init__(self, area: "PromptInput", value: str) -> None:
                super().__init__()
                self.area = area
                self.value = value

        def _on_key(self, event) -> None:
            if event.key == "enter":
                event.prevent_default()
                text = self.text.strip()
                if text:
                    self.post_message(PromptInput.Submitted(self, text))
                    self.clear()

    class ToolCallEvent(Message):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

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
        TokenBar {{
            height: 1;
            color: {t.text_dim};
        }}
        LoadingIndicator {{
            display: none;
            height: 1;
            background: {t.active};
        }}
        LoadingIndicator.active {{
            display: block;
        }}
        """

        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+d", "quit", "Quit", show=False),
            Binding("f1", "show_help", "Help"),
            Binding("ctrl+tab", "focus_next", "Switch focus", show=False),
        ]

        def __init__(self, agent: "Agent", session=None, **kwargs):
            super().__init__(**kwargs)
            self._agent = agent
            self._session = session
            self._last_tool_calls: list[str] = []
            self._current_tool: str | None = None
            self._tokens_before: int = 0

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
            yield TokenBar(cfg.llm.ctx_window, id="token-bar")
            with TabbedContent(initial="tab-chat", id="view-tabs"):
                with TabPane("chat", id="tab-chat"):
                    yield ConversationView(id="chat-log", markup=True, highlight=True)
                with TabPane("Q", id="tab-q"):
                    yield Static(_PLACEHOLDER_Q, id="placeholder-q", classes="placeholder-pane", markup=True)
                with TabPane("A", id="tab-a"):
                    yield Static(_PLACEHOLDER_A, id="placeholder-a", classes="placeholder-pane", markup=True)
                with TabPane("sparse", id="tab-sparse"):
                    yield Static(_PLACEHOLDER_SPARSE, id="placeholder-sparse", classes="placeholder-pane", markup=True)
                with TabPane("sys", id="tab-sys"):
                    yield SysView(id="sys-log", markup=True, highlight=True)
            yield LoadingIndicator(id="loading-indicator")
            yield ContextPanel("", id="context-panel")
            yield GitStatusBar("git: loading...", id="git-status")
            yield PromptInput(id="input-bar")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#input-bar", PromptInput).focus()
            self.call_later(self._refresh_git)
            # Seed the sys log with session info
            sys_log = self.query_one("#sys-log", SysView)
            if self._session:
                sys_log.write(
                    f"[{t.text_dim}]session  {self._session.id}[/{t.text_dim}]"
                    + (f"  [{t.cmd_color}]{self._session.short_name}[/{t.cmd_color}]"
                       if self._session.short_name else "")
                )
            sys_log.write(f"[{t.text_dim}]Type /help for commands  ·  F1 opens this tab[/{t.text_dim}]")

        # ── helpers ──────────────────────────────────────────────────────────

        def _write_sys(self, text: str) -> None:
            """Write to the sys log and switch to that tab."""
            self.query_one("#sys-log", SysView).write(text)
            self.query_one(TabbedContent).active = "tab-sys"

        def _write_chat(self, text: str) -> None:
            self.query_one("#chat-log", ConversationView).write(text)

        def _switch_to_chat(self) -> None:
            self.query_one(TabbedContent).active = "tab-chat"

        # ── actions ──────────────────────────────────────────────────────────

        def action_show_help(self) -> None:
            self._write_sys(_make_help_text(t))

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

            elif cmd == "/compact":
                from agent.memory.compactor import compact
                cfg = self._agent.config
                client = AsyncOpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key)
                before = self._agent.token_estimate()
                self._write_sys(f"[{t.text_dim}]Compacting…[/{t.text_dim}]")
                try:
                    self._agent.messages = await compact(self._agent.messages, cfg, client)
                    after = self._agent.token_estimate()
                    self._write_sys(f"[{t.success}]Compacted.[/{t.success}] {before} → {after} tokens")
                except Exception as e:
                    self._write_sys(f"[{t.error}]Compact failed: {e}[/{t.error}]")
                self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())

            elif cmd == "/clear":
                self.query_one("#chat-log", ConversationView).clear()
                self._switch_to_chat()

            elif cmd == "/tokens":
                used = self._agent.token_estimate()
                cfg = self._agent.config
                self._write_sys(
                    f"tokens: {used}/{cfg.llm.ctx_window}  "
                    f"({len(self._agent.messages)} messages)"
                )

            elif cmd == "/reset":
                system = next((m for m in self._agent.messages if m.get("role") == "system"), None)
                self._agent.messages = [system] if system else []
                self._write_sys(f"[{t.text_dim}]Conversation history cleared.[/{t.text_dim}]")

            elif cmd == "/tools":
                from agent.tools import get_schemas
                names = [s["function"]["name"] for s in get_schemas()]
                self._write_sys("Tools: " + "  ".join(names))

            elif cmd == "/save":
                from agent.memory.session import save_session
                save_session(self._session, self._agent.messages)
                label = self._session.short_name or self._session.id
                self._write_sys(f"[{t.text_dim}]Saved session '{label}'.[/{t.text_dim}]")

            elif cmd == "/load":
                if not arg.strip():
                    self._write_sys(f"[{t.warning}]Usage: /load <session-id-or-short-name>[/{t.warning}]")
                else:
                    from agent.memory.session import load_session
                    loaded_session, loaded_msgs = load_session(arg.strip())
                    if loaded_session is None:
                        self._write_sys(f"[{t.warning}]Session '{arg.strip()}' not found.[/{t.warning}]")
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
                        self.query_one("#token-bar", TokenBar).update_tokens(
                            self._agent.token_estimate()
                        )

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime
                sessions = list_sessions()
                if not sessions:
                    self._write_sys(f"[{t.text_dim}]No sessions found.[/{t.text_dim}]")
                for s in sessions:
                    ts_val = s.get("updated_at") or s.get("created_at")
                    ts = datetime.datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M") if ts_val else "?"
                    label = s.get("short_name") or s["id"]
                    name_extra = f"  [{t.text_dim}]{s['name']}[/{t.text_dim}]" if s.get("name") else ""
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
                                tc["function"]["name"] for tc in tool_calls if isinstance(tc, dict)
                            )
                            lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                        else:
                            lines.append(f"**Agent:** {content}\n")
                md_text = "\n---\n".join(lines)
                label = (self._session.short_name or self._session.id) if self._session else "session"
                target = arg.strip() or f"{label}.md"
                Path(target).write_text(md_text, encoding="utf-8")
                self._write_sys(
                    f"[{t.text_dim}]Exported to {target} ({len(lines)} turns).[/{t.text_dim}]"
                )

            else:
                self._write_sys(f"[{t.warning}]Unknown command '{cmd}'. Type /help.[/{t.warning}]")

        # ── input handling ───────────────────────────────────────────────────

        async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
            user_text = event.value.strip()
            if not user_text:
                return

            if user_text.startswith("/"):
                parts = user_text.split(None, 1)
                await self._run_slash(parts[0].lower(), parts[1] if len(parts) > 1 else "")
                return

            # Switch to chat tab so the user sees the exchange.
            self._switch_to_chat()
            self._write_chat(f"[bold {t.user_color}]You:[/bold {t.user_color}] {user_text}")
            self._last_tool_calls = []
            self._current_tool = None
            self._tokens_before = self._agent.token_estimate()

            self.query_one("#input-bar", PromptInput).disabled = True
            self.query_one("#loading-indicator", LoadingIndicator).add_class("active")
            self.query_one("#context-panel", ContextPanel).set_context("[dim]thinking…[/dim]")

            self._start_chat(user_text)

        @work(exclusive=True, exit_on_error=False, name="chat")
        async def _start_chat(self, user_text: str) -> str:
            from agent.memory.session import save_session

            def on_tool(name: str, args: str) -> None:
                self._last_tool_calls.append(name)
                self._current_tool = name
                self.post_message(ToolCallEvent(name))

            def on_user_message() -> None:
                if self._session is not None:
                    save_session(self._session, self._agent.messages)

            result = await self._agent.chat(
                user_text, on_tool_call=on_tool, on_user_message=on_user_message
            )
            if self._session is not None:
                save_session(self._session, self._agent.messages)
            return result

        def on_tool_call_event(self, event: ToolCallEvent) -> None:
            self._write_chat(f"[{t.tool_color}]  ⚙ {event.name}[/{t.tool_color}]")
            self.query_one("#context-panel", ContextPanel).set_context(
                f"[{t.tool_color}]⚙ {event.name}[/{t.tool_color}]"
            )

        def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
            if event.worker.name != "chat":
                return
            if event.state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
                return

            self.query_one("#loading-indicator", LoadingIndicator).remove_class("active")
            input_widget = self.query_one("#input-bar", PromptInput)
            input_widget.disabled = False
            input_widget.focus()

            if event.state == WorkerState.SUCCESS:
                response = event.worker.result
            elif event.state == WorkerState.ERROR:
                response = f"[{t.error}]Error: {event.worker.error}[/{t.error}]"
            else:
                response = None

            tokens_after = self._agent.token_estimate()
            delta = tokens_after - self._tokens_before
            tools_line = (
                f"[{t.tool_color}]⚙ {', '.join(self._last_tool_calls[-3:])}[/{t.tool_color}]"
                if self._last_tool_calls else ""
            )
            token_line = (
                f"[{t.text_dim}]sent ≈{self._tokens_before:,}  "
                f"[{t.active}]+{delta:,}[/{t.active}] new  "
                f"total {tokens_after:,}[/{t.text_dim}]"
            )
            self.query_one("#context-panel", ContextPanel).set_context(
                f"{tools_line}\n{token_line}" if tools_line else token_line
            )

            if response:
                self._write_chat(
                    f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {response}"
                )

            self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())
            self.call_later(self._refresh_git)

    return CodeAgentApp(agent, session=session)


# ── Simple (Claude Code-style) ───────────────────────────────────────────────

def _make_help_text(theme: "ThemeConfig") -> str:  # type: ignore[name-defined]
    c = theme.cmd_color
    return f"""
[bold]Slash commands[/bold]

  [{c}]/help[/{c}]               show this message
  [{c}]/compact[/{c}]            summarise old messages to free context space
  [{c}]/tokens[/{c}]             show token usage breakdown
  [{c}]/clear[/{c}]              clear the screen
  [{c}]/reset[/{c}]              drop conversation history (keep system prompt)
  [{c}]/save [name][/{c}]        save session under a name (default: current)
  [{c}]/load <name>[/{c}]        load a saved session into the current conversation
  [{c}]/sessions[/{c}]           list saved sessions
  [{c}]/tools[/{c}]              list available tools
  [{c}]/apply [file][/{c}]       write last code block to file (bypass tool calling)
  [{c}]/undo [file][/{c}]        restore last pre-write snapshot of a file
  [{c}]/export [file][/{c}]      export conversation as markdown

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
            k_used = f"{used/1000:.1f}k" if used >= 1000 else str(used)
            k_ctx = f"{ctx//1000}k" if ctx >= 1000 else str(ctx)
            fields.append(f"{k_used}/{k_ctx}")

        msg_count = max(0, len(agent.messages) - 1)  # exclude system prompt
        fields.append(f"{msg_count} msg")

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
    SEP = "  "      # separator between fields
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


async def simple_loop(agent: "Agent", session=None) -> None:
    from rich.console import Console
    from rich.markdown import Markdown
    import readline  # enables arrow keys / history on Linux

    cfg = agent.config
    t = cfg.ui.theme
    console = Console()

    prompt_esc = _hex_to_ansi(t.prompt)
    console.print(f"[bold {t.agent_color}]local-code-agent[/bold {t.agent_color}]  [dim]{cfg.llm.model}  {cfg.llm.ctx_window} ctx[/dim]")
    console.print(f"[{t.text_dim}]/help /compact /tokens /reset /tools /apply /save /sessions  ·  Ctrl+D to quit[/{t.text_dim}]\n")

    while True:
        try:
            user_input = input(f"{prompt_esc}>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{t.text_dim}]Bye.[/{t.text_dim}]")
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
                    console.print(f"[green]Compacted.[/green] {before} → {after} tokens  "
                                  f"({len(agent.messages)} messages)")
                except Exception as e:
                    console.print(f"[red]Compact failed: {e}[/red]")

            elif cmd == "/clear":
                console.clear()

            elif cmd == "/reset":
                system = next((m for m in agent.messages if m.get("role") == "system"), None)
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
                    console.print("[yellow]Usage: /load <session-id-or-short-name>[/yellow]")
                else:
                    from agent.memory.session import load_session
                    loaded_session, loaded_msgs = load_session(arg.strip())
                    if loaded_session is None:
                        console.print(f"[yellow]Session '{arg.strip()}' not found.[/yellow]")
                    else:
                        loaded_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in loaded_msgs]
                        agent.messages = loaded_msgs
                        session = loaded_session
                        label = session.short_name or session.id
                        console.print(f"[dim]Loaded session '{label}' ({len(loaded_msgs)} messages).[/dim]")

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime
                for s in list_sessions():
                    ts_val = s.get("updated_at") or s.get("created_at")
                    ts = datetime.datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M") if ts_val else "?"
                    label = s.get("short_name") or s["id"]
                    name_part = f"  [dim]{s.get('name', '')}[/dim]" if s.get("name") else ""
                    console.print(f"  [cyan]{label}[/cyan]{name_part}  {s['message_count']} msgs  [dim]{ts}[/dim]")

            elif cmd == "/tools":
                from agent.tools import get_schemas
                for t in get_schemas():
                    fn = t["function"]
                    console.print(f"  [cyan]{fn['name']}[/cyan]  [dim]{fn.get('description','')[:60]}[/dim]")

            elif cmd == "/apply":
                from agent.agent import extract_last_code_block
                from agent.tools.files import write_file
                result = extract_last_code_block(agent.messages)
                if not result:
                    console.print("[yellow]No code block found in recent messages.[/yellow]")
                else:
                    fname, code = result
                    target = arg.strip() or fname
                    console.print(f"[dim]Writing to {target}:[/dim]")
                    console.print(f"[dim]{code[:200]}{'…' if len(code) > 200 else ''}[/dim]")
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
                            names = ", ".join(tc["function"]["name"] for tc in tool_calls if isinstance(tc, dict))
                            lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                        else:
                            lines.append(f"**Agent:** {content}\n")
                    # skip tool result messages (role == "tool")
                md_text = "\n---\n".join(lines)
                _session_label = (session.short_name or session.id) if session else "session"
                target = arg.strip() or f"{_session_label}.md"
                Path(target).write_text(md_text, encoding="utf-8")
                console.print(f"[dim]Exported to {target} ({len(lines)} turns).[/dim]")

            else:
                console.print(f"[yellow]Unknown command '{cmd}'. Type /help for a list.[/yellow]")

            continue

        # ── Normal message ──────────────────────────────────────────────────
        import sys
        tool_results: list[str] = []
        streaming_tokens: list[str] = []
        _spinner_status: list[str] = ["thinking…"]
        _spinner_stop = asyncio.Event()
        _spinner_task = asyncio.create_task(_run_spinner(_spinner_status, _spinner_stop, agent=agent))

        def on_tool(name: str, args_str: str) -> None:
            # Stop spinner and clear its line before printing
            _spinner_stop.set()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            console.print(f"  [{t.tool_color}]⚙ {name}[/{t.tool_color}]")
            tool_results.append(name)

        def on_token(token: str) -> None:
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

        try:
            response = await agent.chat(
                user_input,
                on_tool_call=on_tool,
                on_token=on_token,
                on_user_message=_on_user_message,
            )
        except Exception as e:
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
            console.print(f"[{t.text_dim}]Done. ({', '.join(tool_results)})[/{t.text_dim}]")
        else:
            console.print(f"[{t.warning}]No response from model.[/{t.warning}]")

        if cfg.ui.show_token_count:
            console.print(f"\n{_token_bar(agent.token_estimate(), cfg.llm.ctx_window)}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def run_ui(agent: "Agent", session=None) -> None:
    cfg = agent.config
    if cfg.ui.mode == "textual":
        try:
            app = _build_textual_app(agent, session=session)
            app.run()
        except ImportError:
            print("Textual not available, falling back to simple mode.")
            asyncio.run(simple_loop(agent, session=session))
    else:
        asyncio.run(simple_loop(agent, session=session))
