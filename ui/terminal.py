from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent
    from agent.config import Config


# ── Textual UI ──────────────────────────────────────────────────────────────

def _build_textual_app(agent: "Agent"):
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Input, RichLog, Static, ProgressBar, TextArea, LoadingIndicator
    from textual.containers import Vertical, Horizontal
    from textual.binding import Binding
    from textual.message import Message
    from textual.worker import Worker, WorkerState
    from textual import work
    from rich.text import Text
    from rich.markdown import Markdown

    class TokenBar(Static):
        def __init__(self, ctx_window: int, **kwargs):
            super().__init__(**kwargs)
            self._ctx = ctx_window
            self._used = 0

        def update_tokens(self, used: int) -> None:
            self._used = used
            pct = used / self._ctx if self._ctx else 0
            bar_len = 20
            filled = int(pct * bar_len)
            color = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
            bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"
            self.update(f"tokens: {used}/{self._ctx} {bar}")

    class ConversationView(RichLog):
        pass

    class ContextPanel(Static):
        def set_context(self, text: str) -> None:
            self.update(text)

    class GitStatusBar(Static):
        def set_status(self, text: str) -> None:
            self.update(text)

    class PromptInput(TextArea):
        """Wrapping multi-line input that submits on Enter (Shift+Enter for newline)."""

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

    class CodeAgentApp(App):
        CSS = """
        Screen {
            layout: vertical;
        }
        #header-bar {
            height: 1;
            background: $panel;
            padding: 0 1;
        }
        #conversation {
            height: 1fr;
            border: solid $primary;
            padding: 0 1;
        }
        #context-panel {
            height: 3;
            background: $panel;
            padding: 0 1;
        }
        #git-status {
            height: 1;
            background: $panel-darken-1;
            padding: 0 1;
        }
        #input-bar {
            height: auto;
            max-height: 8;
            min-height: 3;
            border: solid $accent;
        }
        TokenBar {
            height: 1;
        }
        LoadingIndicator {
            display: none;
            height: 1;
            background: $accent;
        }
        LoadingIndicator.active {
            display: block;
        }
        """

        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+d", "quit", "Quit", show=False),
            Binding("f1", "show_help", "Help"),
        ]

        def __init__(self, agent: "Agent", **kwargs):
            super().__init__(**kwargs)
            self._agent = agent
            self._last_tool_calls: list[str] = []
            self._current_tool: str | None = None

        def compose(self) -> ComposeResult:
            cfg = self._agent.config
            yield Static(
                f"[bold]local-code-agent[/bold]  model: {cfg.llm.model}  "
                f"  /help for commands",
                id="header-bar",
            )
            yield TokenBar(cfg.llm.ctx_window, id="token-bar")
            yield ConversationView(id="conversation", markup=True, highlight=True)
            yield LoadingIndicator(id="loading-indicator")
            yield ContextPanel("context: (none)", id="context-panel")
            yield GitStatusBar("git: loading...", id="git-status")
            yield PromptInput(id="input-bar")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#input-bar", PromptInput).focus()
            self.call_later(self._refresh_git)

        def action_show_help(self) -> None:
            conv = self.query_one("#conversation", ConversationView)
            conv.write("[bold]Commands:[/bold]  /help  /compact  /tokens  /reset  /tools  /save  /sessions")

        async def _refresh_git(self) -> None:
            from agent.tools.git import git_status
            try:
                loop = asyncio.get_event_loop()
                s = await loop.run_in_executor(None, git_status)
                branch = s.get("branch", "?")
                staged = len(s.get("staged", []))
                bar = self.query_one("#git-status", GitStatusBar)
                bar.set_status(f"git: {staged} staged  branch: {branch}")
            except Exception:
                pass

        async def _run_slash(self, cmd: str, arg: str, conv: "ConversationView") -> None:
            from openai import AsyncOpenAI
            if cmd == "/help":
                conv.write("[bold]Commands:[/bold]  /help  /compact  /tokens  /reset  /tools  /save [name]  /load <name>  /sessions  /export [file]")
            elif cmd == "/compact":
                from agent.memory.compactor import compact
                cfg = self._agent.config
                client = AsyncOpenAI(base_url=cfg.llm.base_url, api_key=cfg.llm.api_key)
                before = self._agent.token_estimate()
                conv.write("[dim]Compacting…[/dim]")
                try:
                    self._agent.messages = await compact(self._agent.messages, cfg, client)
                    after = self._agent.token_estimate()
                    conv.write(f"[green]Compacted.[/green] {before} → {after} tokens")
                except Exception as e:
                    conv.write(f"[red]Compact failed: {e}[/red]")
                self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())
            elif cmd == "/tokens":
                used = self._agent.token_estimate()
                cfg = self._agent.config
                conv.write(f"tokens: {used}/{cfg.llm.ctx_window}  ({len(self._agent.messages)} messages)")
            elif cmd == "/reset":
                system = next((m for m in self._agent.messages if m.get("role") == "system"), None)
                self._agent.messages = [system] if system else []
                conv.write("[dim]History cleared.[/dim]")
            elif cmd == "/tools":
                from agent.tools import get_schemas
                names = [t["function"]["name"] for t in get_schemas()]
                conv.write("Tools: " + "  ".join(names))
            elif cmd == "/save":
                from agent.memory.session import save_session
                name = arg.strip() or "default"
                save_session(name, self._agent.messages)
                conv.write(f"[dim]Saved as '{name}'.[/dim]")
            elif cmd == "/load":
                if not arg.strip():
                    conv.write("[yellow]Usage: /load <session-name>[/yellow]")
                else:
                    from agent.memory.session import load_session
                    loaded_msgs, _ = load_session(arg.strip())
                    if not loaded_msgs:
                        conv.write(f"[yellow]Session '{arg.strip()}' not found.[/yellow]")
                    else:
                        loaded_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in loaded_msgs]
                        self._agent.messages = loaded_msgs
                        conv.write(f"[dim]Loaded session '{arg.strip()}' ({len(loaded_msgs)} messages).[/dim]")
                        self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())
            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime
                for s in list_sessions():
                    ts = datetime.datetime.fromtimestamp(s["saved_at"]).strftime("%Y-%m-%d %H:%M") if s["saved_at"] else "?"
                    conv.write(f"  {s['name']}  {s['message_count']} msgs  {ts}")
            elif cmd == "/undo":
                from agent.tools.files import undo_file, undo_candidates
                target = arg.strip()
                if not target:
                    candidates = undo_candidates()
                    if not candidates:
                        conv.write("[yellow]Nothing to undo.[/yellow]")
                    else:
                        conv.write("Undo candidates: " + ", ".join(candidates))
                else:
                    r = undo_file(target)
                    if "error" in r:
                        conv.write(f"[red]{r['error']}[/red]")
                    else:
                        conv.write(f"[green]Restored {target}[/green]")
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
                            names = ", ".join(tc["function"]["name"] for tc in tool_calls if isinstance(tc, dict))
                            lines.append(f"**Agent** *(tools: {names})*: {content}\n")
                        else:
                            lines.append(f"**Agent:** {content}\n")
                md_text = "\n---\n".join(lines)
                target = arg.strip() or "session.md"
                Path(target).write_text(md_text, encoding="utf-8")
                conv.write(f"[dim]Exported to {target} ({len(lines)} turns).[/dim]")
            else:
                conv.write(f"[yellow]Unknown command '{cmd}'. Type /help.[/yellow]")

        async def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
            user_text = event.value.strip()
            if not user_text:
                return

            conv = self.query_one("#conversation", ConversationView)

            if user_text.startswith("/"):
                parts = user_text.split(None, 1)
                await self._run_slash(parts[0].lower(), parts[1] if len(parts) > 1 else "", conv)
                return

            conv.write(f"[bold cyan]You:[/bold cyan] {user_text}")
            self._last_tool_calls = []
            self._current_tool = None

            self.query_one("#input-bar", PromptInput).disabled = True
            self.query_one("#loading-indicator", LoadingIndicator).add_class("active")
            self.query_one("#context-panel", ContextPanel).set_context("[dim]thinking…[/dim]")

            self._start_chat(user_text)

        @work(exclusive=True, exit_on_error=False, name="chat")
        async def _start_chat(self, user_text: str) -> str:
            def on_tool(name: str, args: str) -> None:
                self._last_tool_calls.append(name)
                self._current_tool = name
                self.post_message(ToolCallEvent(name))

            return await self._agent.chat(user_text, on_tool_call=on_tool)

        def on_tool_call_event(self, event: ToolCallEvent) -> None:
            self.query_one("#conversation", ConversationView).write(f"[dim]  ⚙ {event.name}[/dim]")
            self.query_one("#context-panel", ContextPanel).set_context(f"[dim]⚙ {event.name}[/dim]")

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
                response = f"[red]Error: {event.worker.error}[/red]"
            else:
                response = None

            context_text = (
                f"context: {', '.join(self._last_tool_calls[-3:])}"
                if self._last_tool_calls else "context: (none)"
            )
            self.query_one("#context-panel", ContextPanel).set_context(context_text)

            if response:
                self.query_one("#conversation", ConversationView).write(
                    f"[bold green]Agent:[/bold green] {response}"
                )

            self.query_one("#token-bar", TokenBar).update_tokens(self._agent.token_estimate())
            self.call_later(self._refresh_git)

    return CodeAgentApp(agent)


# ── Simple (Claude Code-style) ───────────────────────────────────────────────

_HELP_TEXT = """
[bold]Slash commands[/bold]

  [cyan]/help[/cyan]               show this message
  [cyan]/compact[/cyan]            summarise old messages to free context space
  [cyan]/tokens[/cyan]             show token usage breakdown
  [cyan]/clear[/cyan]              clear the screen
  [cyan]/reset[/cyan]              drop conversation history (keep system prompt)
  [cyan]/save [name][/cyan]        save session under a name (default: current)
  [cyan]/load <name>[/cyan]        load a saved session into the current conversation
  [cyan]/sessions[/cyan]           list saved sessions
  [cyan]/tools[/cyan]              list available tools
  [cyan]/apply [file][/cyan]       write last code block to file (bypass tool calling)
  [cyan]/undo [file][/cyan]        restore last pre-write snapshot of a file
  [cyan]/export [file][/cyan]      export conversation as markdown

[dim]Ctrl+D or Ctrl+C to quit[/dim]
"""


def _token_bar(used: int, ctx: int, bar_len: int = 20) -> str:
    pct = used / ctx if ctx else 0
    filled = int(pct * bar_len)
    color = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
    bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"
    return f"[dim]tokens {used}/{ctx}[/dim] {bar}"


async def _run_spinner(status_ref: list[str], stop: asyncio.Event) -> None:
    import sys
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not stop.is_set():
        frame = frames[i % len(frames)]
        sys.stdout.write(f"\r\033[2m{frame} {status_ref[0]}\033[0m")
        sys.stdout.flush()
        i += 1
        await asyncio.sleep(0.08)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


async def simple_loop(agent: "Agent", session_name: str = "default") -> None:
    from rich.console import Console
    from rich.markdown import Markdown
    import readline  # enables arrow keys / history on Linux

    cfg = agent.config
    console = Console()

    console.print(f"[bold]local-code-agent[/bold]  [dim]{cfg.llm.model}  {cfg.llm.ctx_window} ctx[/dim]")
    console.print("[dim]/help /compact /tokens /reset /tools /apply /save /sessions  ·  Ctrl+D to quit[/dim]\n")

    while True:
        try:
            user_input = input("\033[1;36m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        # ── Slash commands ──────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                console.print(_HELP_TEXT)

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
                name = arg.strip() or session_name
                save_session(name, agent.messages)
                console.print(f"[dim]Saved as '{name}'.[/dim]")

            elif cmd == "/load":
                if not arg.strip():
                    console.print("[yellow]Usage: /load <session-name>[/yellow]")
                else:
                    from agent.memory.session import load_session
                    loaded_msgs, _ = load_session(arg.strip())
                    if not loaded_msgs:
                        console.print(f"[yellow]Session '{arg.strip()}' not found.[/yellow]")
                    else:
                        loaded_msgs = [{k: v for k, v in m.items() if not k.startswith("_")} for m in loaded_msgs]
                        agent.messages = loaded_msgs
                        session_name = arg.strip()
                        console.print(f"[dim]Loaded session '{session_name}' ({len(loaded_msgs)} messages).[/dim]")

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime
                for s in list_sessions():
                    ts = datetime.datetime.fromtimestamp(s["saved_at"]).strftime("%Y-%m-%d %H:%M") if s["saved_at"] else "?"
                    console.print(f"  [cyan]{s['name']}[/cyan]  {s['message_count']} msgs  [dim]{ts}[/dim]")

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
                target = arg.strip() or f"{session_name}.md"
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
        _spinner_task = asyncio.create_task(_run_spinner(_spinner_status, _spinner_stop))

        def on_tool(name: str, args_str: str) -> None:
            # Clear the spinner line before printing
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            console.print(f"  [dim]⚙ {name}[/dim]")
            tool_results.append(name)
            _spinner_status[0] = f"{name}…"

        def on_token(token: str) -> None:
            if not streaming_tokens:
                # First token — clear spinner line
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            streaming_tokens.append(token)
            console.print(token, end="", highlight=False)

        try:
            response = await agent.chat(user_input, on_tool_call=on_tool, on_token=on_token)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue
        finally:
            _spinner_stop.set()
            await _spinner_task

        # If streaming was active, the text is already printed; just add newline.
        # If no streaming occurred (tool-only turn), print the response normally.
        if streaming_tokens:
            console.print()  # end the streaming line
        elif response:
            console.print()
            console.print(Markdown(response))
        elif tool_results:
            console.print(f"[dim]Done. ({', '.join(tool_results)})[/dim]")
        else:
            console.print("[yellow]No response from model.[/yellow]")

        if cfg.ui.show_token_count:
            console.print(f"\n{_token_bar(agent.token_estimate(), cfg.llm.ctx_window)}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def run_ui(agent: "Agent", session_name: str = "default") -> None:
    cfg = agent.config
    if cfg.ui.mode == "textual":
        try:
            app = _build_textual_app(agent)
            app.run()
        except ImportError:
            print("Textual not available, falling back to simple mode.")
            asyncio.run(simple_loop(agent, session_name=session_name))
    else:
        asyncio.run(simple_loop(agent, session_name=session_name))
