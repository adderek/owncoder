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
    ("/analyze-asm", ["/asm"],  "analyze assembly file  --resume --force --levels N", True),
    ("/apply",       [],        "write last code block to file", False),
    ("/clear",       [],        "clear the chat screen", False),
    ("/compact",     [],        "summarize old messages to free context", False),
    ("/exec",        [],        "run a shell command", True),
    ("/export",      [],        "export conversation as markdown", False),
    ("/help",        ["/?"],    "show this help", False),
    ("/load",        [],        "load a saved session", True),
    ("/reset",       [],        "drop conversation history", False),
    ("/save",        [],        "save session under a name", False),
    ("/sessions",    [],        "list saved sessions", False),
    ("/tokens",      [],        "show token usage", False),
    ("/tools",       [],        "list available tools", False),
    ("/undo",        [],        "restore last file snapshot", False),
]


def _match_commands(prefix: str) -> list[tuple[str, str, bool]]:
    """Return (primary_name, description, takes_arg) for commands whose primary
    name or any alias starts with *prefix* (case-insensitive)."""
    pl = prefix.lower()
    out = []
    for primary, aliases, desc, takes_arg in _SLASH_COMMANDS:
        if primary.startswith(pl) or any(a.startswith(pl) for a in aliases):
            out.append((primary, desc, takes_arg))
    return out


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
            for i, (cmd, desc, _) in enumerate(matches[:self.MAX_VISIBLE]):
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
        """Multi-line input; Enter submits, Shift+Enter inserts newline.
        Up/Down when empty browses history; ESC cancels; Enter edits & retries."""

        class Submitted(Message):
            def __init__(self, area: "PromptInput", value: str) -> None:
                super().__init__()
                self.area = area
                self.value = value

        class HistorySubmitted(Message):
            """User confirmed editing a past message. remove_count interactions will be rolled back."""
            def __init__(self, area: "PromptInput", value: str, remove_count: int) -> None:
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
            self._mode: str = "normal"   # "normal" | "browsing" | "editing"
            self._history_idx: int | None = None
            self._edit_source_idx: int | None = None
            self._saved_text: str = ""
            self._comp_matches: list[tuple[str, str, bool]] = []
            self._comp_idx: int = -1          # -1 = no item highlighted
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
            self.post_message(self.HintChanged(
                f"[bold]↑↓[/bold] navigate  [bold]ENTER[/bold]=edit&retry  [bold]ESC[/bold]=cancel{rc_str}"
            ))

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
            self.post_message(self.HintChanged(
                f"[bold]ENTER[/bold]=retry  [bold]ESC[/bold]=cancel{rc_str}"
            ))

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
            yield CompletionBar("", id="completion-bar", markup=True)
            yield HintBar("", id="hint-bar", markup=True)
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#input-bar", PromptInput).focus()
            self.call_later(self._refresh_git)
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
                    + (f"  [{t.cmd_color}]{self._session.short_name}[/{t.cmd_color}]"
                       if self._session.short_name else "")
                )
            sys_log.write(f"[{t.text_dim}]Type /help for commands  ·  F1 opens this tab[/{t.text_dim}]")
            # Restore prior dialogue if session was loaded before the UI started
            if self._agent.messages:
                self._restore_chat_history(self._agent.messages)

        # ── helpers ──────────────────────────────────────────────────────────

        def _write_sys(self, text: str) -> None:
            """Write to the sys log and switch to that tab."""
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
            for m in messages:
                role = m.get("role", "")
                if role == "system":
                    continue
                content = m.get("content") or ""
                if isinstance(content, list):
                    content = _json.dumps(content)
                tool_calls = m.get("tool_calls") or []
                if role == "user":
                    chat_log.write(
                        f"[bold {t.user_color}]You:[/bold {t.user_color}] {content}"
                    )
                elif role == "assistant":
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            name = tc.get("function", {}).get("name", "?")
                            chat_log.write(
                                f"[{t.tool_color}]  ⚙ {name}[/{t.tool_color}]"
                            )
                    if content:
                        chat_log.write(
                            f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {content}"
                        )

        # ── actions ──────────────────────────────────────────────────────────

        def action_quit(self) -> None:
            try:
                from agent.tools.analyze_asm import get_interrupt_flag
                get_interrupt_flag().set()
            except Exception:
                pass
            self.exit()

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

            elif cmd == "/exec":
                if not arg.strip():
                    self._write_sys(f"[{t.warning}]Usage: /exec <command>[/{t.warning}]")
                else:
                    from agent.tools.shell import run_command
                    self._write_sys(f"[{t.text_dim}]$ {arg.strip()}[/{t.text_dim}]")
                    result = run_command(arg.strip())
                    if result.get("stdout"):
                        self._write_sys(result["stdout"].rstrip())
                    if result.get("stderr"):
                        self._write_sys(f"[{t.warning}]{result['stderr'].rstrip()}[/{t.warning}]")
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

            elif cmd == "/analyze-asm":
                await self._run_analyze_asm(arg)

            else:
                self._write_sys(f"[{t.warning}]Unknown command '{cmd}'. Type /help.[/{t.warning}]")

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
            self._write_sys(f"[{t.text_dim}]Analyzing {path}…  ESC to interrupt[/{t.text_dim}]")

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
                self._write_sys(f"[{t.success}]{result.get('message', str(result))}[/{t.success}]")

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
                await self._run_slash(parts[0].lower(), parts[1] if len(parts) > 1 else "")
                return

            input_widget = self.query_one("#input-bar", PromptInput)
            input_widget.add_to_history(user_text)
            self._begin_chat(user_text)

        async def on_prompt_input_history_submitted(self, event: PromptInput.HistorySubmitted) -> None:
            user_text = event.value.strip()
            if not user_text:
                return

            # Roll back agent messages: remove the last remove_count user turns
            if event.remove_count > 0:
                user_positions = [
                    i for i, m in enumerate(self._agent.messages)
                    if m.get("role") == "user"
                ]
                if event.remove_count >= len(user_positions):
                    system = next(
                        (m for m in self._agent.messages if m.get("role") == "system"), None
                    )
                    self._agent.messages = [system] if system else []
                else:
                    cut = user_positions[-event.remove_count]
                    self._agent.messages = self._agent.messages[:cut]
                self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())

            # Truncate history to the edit point and add the new version
            input_widget = self.query_one("#input-bar", PromptInput)
            if event.area._edit_source_idx is not None:
                input_widget._history = input_widget._history[:event.area._edit_source_idx]
            input_widget.add_to_history(user_text)

            self._begin_chat(user_text)

        def _begin_chat(self, user_text: str) -> None:
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
                err = event.worker.error
                tb = "".join(traceback.format_exception(type(err), err, err.__traceback__)) if err else ""
                logger.error("chat worker error: %s\n%s", err, tb)
                response = f"[{t.error}]Error: {err}[/{t.error}]"
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
  [{c}]/exec <command>[/{c}]      run an OS command and show output
  [{c}]/apply [file][/{c}]       write last code block to file (bypass tool calling)
  [{c}]/undo [file][/{c}]        restore last pre-write snapshot of a file
  [{c}]/export [file][/{c}]      export conversation as markdown
  [{c}]/analyze-asm <file>[/{c}]  LLM-driven assembly analysis and summarization
                       options: --resume  --force  --levels N

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


async def simple_loop(agent: "Agent", session=None):
    from rich.console import Console
    from rich.markdown import Markdown
    import readline  # enables arrow keys / history on Linux

    cfg = agent.config
    t = cfg.ui.theme
    console = Console()

    prompt_esc = _hex_to_ansi(t.prompt)
    console.print(f"[bold {t.agent_color}]local-code-agent[/bold {t.agent_color}]  [dim]{cfg.llm.model}  {cfg.llm.ctx_window} ctx[/dim]")
    console.print(f"[{t.text_dim}]/help /compact /tokens /reset /tools /exec /apply /save /sessions  ·  Ctrl+D to quit[/{t.text_dim}]\n")

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
                for schema in get_schemas():
                    fn = schema["function"]
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

            elif cmd == "/analyze-asm":
                from agent.tools.analyze_asm import analyze_asm, get_interrupt_flag
                parts = arg.split()
                if not parts:
                    console.print("[yellow]Usage: /analyze-asm <file> [--resume] [--force] [--levels N][/yellow]")
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
                    console.print(f"[dim]Analyzing {path_arg}… (Ctrl+C to interrupt)[/dim]")
                    try:
                        kwargs = {"path": path_arg, "resume": resume, "force": force_flag}
                        if max_lvls is not None:
                            kwargs["max_levels"] = max_lvls
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: analyze_asm(**kwargs))
                        if "error" in result:
                            console.print(f"[red]{result['error']}[/red]")
                        else:
                            console.print(f"[green]{result.get('message', str(result))}[/green]")
                    except KeyboardInterrupt:
                        interrupt.set()
                        console.print("[yellow]Interrupted.[/yellow]")
                    except Exception as e:
                        console.print(f"[red]analyze-asm error: {e}[/red]")

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
            console.print(f"[{t.text_dim}]Done. ({', '.join(tool_results)})[/{t.text_dim}]")
        else:
            console.print(f"[{t.warning}]No response from model.[/{t.warning}]")

        if cfg.ui.show_token_count:
            console.print(f"\n{_token_bar(agent.token_estimate(), cfg.llm.ctx_window)}\n")

    return session


# ── Entry point ──────────────────────────────────────────────────────────────

def run_ui(agent: "Agent", session=None):
    cfg = agent.config
    if cfg.ui.mode == "textual":
        try:
            app = _build_textual_app(agent, session=session)
            app.run()
            return session
        except ImportError:
            print("Textual not available, falling back to simple mode.")
            return asyncio.run(simple_loop(agent, session=session))
    else:
        return asyncio.run(simple_loop(agent, session=session))
