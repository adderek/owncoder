"""Simple readline-based (non-Textual) interactive loop."""
from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from agent.ui.render import _render_context_report, _delatex
from agent.ui.spinner import _run_spinner, _fmt_tps
from agent.ui.colors import _hex_to_ansi

logger = logging.getLogger(__name__)


def _find_loop_guard_stop(messages: list[dict]) -> str | None:
    """Return the loop-guard stop note from the last assistant message, or None."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = (m.get("content") or "").strip()
            return content if content.startswith("[loop guard:") else None
    return None

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.config import ThemeConfig
    from agent.ui_server import UIServerProtocol


def _make_help_text(theme: "ThemeConfig") -> str:  # type: ignore[name-defined]
    c = theme.cmd_color
    return f"""
[bold]Slash commands[/bold]

  [{c}]/help[/{c}]               show this message
  [{c}]/compact[/{c}]            summarise old messages to free context space
  [{c}]/continue[/{c}] (or [{c}]continue[/{c}], Ctrl+R)  resume after iteration cap / truncation
  [{c}]/tokens[/{c}]             show token usage breakdown
  [{c}]/clear[/{c}]              clear the screen
  [{c}]/reset[/{c}]              drop conversation history (keep system prompt)
  [{c}]/save [name][/{c}]        save session under a name (default: current)
  [{c}]/load <name>[/{c}]        load a saved session into the current conversation
  [{c}]/sessions[/{c}]           list saved sessions
  [{c}]/tools[/{c}]              list available tools
  [{c}]/skills [show|history|rm <name>][/{c}]  manage saved skills
  [{c}]/checkpoint [new|rollback <id>][/{c}]  restore point across all files
  [{c}]/mcp[/{c}]                MCP server status + tools
  [{c}]/security[/{c}] [scan|diff|triage|selfaudit|report|baseline|airgap|integrity|weights|sbom|verify|full|review] [path]  local security audit
  [{c}]/exec <command>[/{c}]      run an OS command and show output
  [{c}]/apply [file][/{c}]       write last code block to file (bypass tool calling)
  [{c}]/undo [file][/{c}]        restore last pre-write snapshot of a file
  [{c}]/export [file][/{c}]      export conversation as markdown
  [{c}]/q[/{c}] · [{c}]/a[/{c}] · [{c}]/sparse[/{c}]    switch to Q / A / sparse tab  (click a line → jump to turn)
  [{c}]/analyze-asm <file>[/{c}]  LLM-driven assembly analysis and summarization
                       options: --resume  --force  --levels N
  [{c}]/think [level][/{c}]       thinking effort: off|low|normal|high|max ('-' resets)
  [{c}]/autonomy [level][/{c}]    autonomy: 0.0–1.0 (or %) or supervised|explain|balanced|brisk|autopilot ('-' resets)
  [{c}]/temperature [v][/{c}]     sampling temperature 0.0–2.0 (alias [{c}]/temp[/{c}]; '-' resets)
  [{c}]/max_tokens [args][/{c}]   set tokens: <n> | out <n> | in <n> | default
  [{c}]/context[/{c}] ([{c}]/ctx[/{c}], [{c}]/legend[/{c}])  context breakdown grid + color/marker key
  [{c}]/unlimited[/{c}] ([{c}]/nomax[/{c}])  toggle unlimited iterations (no iter cap)
  [{c}]/goal [text | $cmd | clear][/{c}]  set completion goal; agent runs until achieved
                       Ctrl+C while running: stop after current iteration (Ctrl+C again = cancel)

[dim]Ctrl+D or Ctrl+Q to quit[/dim]
"""


def _token_bar(used: int, ctx: int, bar_len: int = 20) -> str:
    pct = used / ctx if ctx else 0
    filled = int(pct * bar_len)
    color = "red" if pct > 0.85 else ("yellow" if pct > 0.65 else "green")
    bar = f"[{color}]{'█' * filled}{'░' * (bar_len - filled)}[/]"
    return f"[dim]tokens {used}/{ctx}[/dim] {bar}"



async def simple_loop(agent: "Agent", session=None, server: "UIServerProtocol | None" = None):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.markup import escape as _escape
    import readline  # enables arrow keys / history on Linux

    from agent.ui_server import LocalUIServer
    if server is None:
        server = LocalUIServer(agent)

    _ui_cfg = server.get_ui_config()
    _llm_cfg = server.get_llm_info()
    t = _ui_cfg["theme"]
    console = Console()

    if session is not None:
        server.set_session_id(session.id)

    prompt_esc = _hex_to_ansi(t.prompt)
    console.print(
        f"[bold {t.agent_color}]local-code-agent[/bold {t.agent_color}]  [dim]{_llm_cfg['model']}  {_llm_cfg['ctx_window']} ctx[/dim]"
    )
    console.print(
        f"[{t.text_dim}]/help /compact /tokens /reset /tools /exec /apply /save /sessions  ·  Ctrl+D to quit[/{t.text_dim}]\n"
    )

    while True:
        try:
            user_input = input(f"{prompt_esc}>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            pending = server.pending_background_count()
            if pending:
                console.print(
                    f"\n[{t.warning}]Finishing {pending} summary task(s)… "
                    f"Ctrl+C again to force exit.[/{t.warning}]"
                )
                try:
                    await server.wait_background(timeout=30.0)
                except KeyboardInterrupt:
                    server.cancel_background()
            console.print(f"[{t.text_dim}]Bye.[/{t.text_dim}]")
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
                used = server.token_estimate()
                console.print(_token_bar(used, server.get_llm_info()["ctx_window"]))
                console.print(f"  [dim]{server.message_count()} messages in context[/dim]")

            elif cmd == "/compact":
                before = server.token_estimate()
                console.print("[dim]Compacting…[/dim]")
                try:
                    await server.compact_messages()
                    after = server.token_estimate()
                    console.print(
                        f"[green]Compacted.[/green] {before} → {after} tokens  "
                        f"({server.message_count()} messages)"
                    )
                except Exception as e:
                    console.print(f"[red]Compact failed: {e}[/red]")

            elif cmd == "/clear":
                console.clear()

            elif cmd == "/reset":
                server.reset_messages()
                console.print("[dim]Conversation history cleared.[/dim]")

            elif cmd == "/save":
                if session is not None:
                    if arg.strip():
                        from agent.memory.session import _sanitize_short_name
                        session.short_name = _sanitize_short_name(arg.strip())
                        session.name = arg.strip()
                    server.save_session(session)
                    label = session.short_name or session.id
                    console.print(f"[dim]Saved session '{label}'.[/dim]")
                else:
                    console.print("[yellow]No active session.[/yellow]")

            elif cmd == "/load":
                if not arg.strip():
                    console.print(
                        "[yellow]Usage: /load <session-id-or-short-name>[/yellow]"
                    )
                else:
                    loaded_session, loaded_msgs = server.load_session(arg.strip())
                    if loaded_session is None:
                        console.print(
                            f"[yellow]Session '{arg.strip()}' not found.[/yellow]"
                        )
                    else:
                        server.set_messages(loaded_msgs)
                        session = loaded_session
                        label = session.short_name or session.id
                        console.print(
                            f"[dim]Loaded session '{label}' ({len(loaded_msgs)} messages).[/dim]"
                        )
                        visible = [m for m in loaded_msgs if m.get("role") in ("user", "assistant") and m.get("content")]
                        for m in visible[-4:]:
                            role_label = "You" if m["role"] == "user" else "Agent"
                            snippet = str(m["content"])[:120].replace("\n", " ")
                            console.print(f"[dim]  {role_label}: {snippet}[/dim]")
                        console.print("[dim]─── resumed ───[/dim]")
                        _loop_guard_note = _find_loop_guard_stop(loaded_msgs)
                        if _loop_guard_note:
                            console.print(
                                "[yellow]Note: session ended with loop-guard stop — "
                                "type a message to redirect (e.g. 're-read the file and retry').[/yellow]"
                            )
                            console.print(f"[dim]  {_loop_guard_note[:200]}[/dim]")

            elif cmd == "/sessions":
                from agent.memory.session import list_sessions
                import datetime

                for s in [s for s in list_sessions() if s.get("message_count", 0) > 0]:
                    ts_val = s.get("updated_at") or s.get("created_at")
                    try:
                        if isinstance(ts_val, (int, float)):
                            ts = datetime.datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M")
                        elif ts_val:
                            ts = datetime.datetime.fromisoformat(ts_val.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
                        else:
                            ts = "?"
                    except Exception:
                        ts = "?"
                    label = s.get("short_name") or s["id"]
                    name_part = (
                        f"  [dim]{s.get('name', '')}[/dim]" if s.get("name") else ""
                    )
                    console.print(
                        f"  [cyan]{label}[/cyan]{name_part}  {s['message_count']} msgs  [dim]{ts}[/dim]"
                    )

            elif cmd == "/tools":
                from agent.tools import get_schemas

                for schema in get_schemas():
                    fn = schema["function"]
                    console.print(
                        f"  [cyan]{fn['name']}[/cyan]  [dim]{fn.get('description', '')[:60]}[/dim]"
                    )

            elif cmd == "/skills":
                from agent.skills import run_skills_command
                console.print(run_skills_command(agent.config, arg))

            elif cmd in ("/checkpoint", "/cp"):
                from agent.core.checkpoint import run_checkpoint_command
                console.print(run_checkpoint_command(arg))

            elif cmd == "/mcp":
                from agent.mcp import run_mcp_command
                console.print(run_mcp_command(agent.config, arg))

            elif cmd in ("/security", "/sec", "/audit"):
                from agent.security.secaudit import run_security_command
                console.print(run_security_command(agent.config, arg))

            elif cmd == "/models":
                from agent.ui.slash import _render_models_table
                console.print(_render_models_table(agent.config))

            elif cmd == "/apply":
                from agent.core.history_ops import extract_last_code_block
                from agent.tools.files import write_file

                result = extract_last_code_block(server.get_messages())
                if not result:
                    console.print(
                        "[yellow]No code block found in recent messages.[/yellow]"
                    )
                else:
                    fname, code = result
                    target = arg.strip() or fname
                    console.print(f"[dim]Writing to {target}:[/dim]")
                    console.print(
                        f"[dim]{code[:200]}{'…' if len(code) > 200 else ''}[/dim]"
                    )
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
                for m in server.get_messages():
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
                    # skip tool result messages (role == "tool")
                md_text = "\n---\n".join(lines)
                _session_label = (
                    (session.short_name or session.id) if session else "session"
                )
                target = arg.strip() or f"{_session_label}.md"
                Path(target).write_text(md_text, encoding="utf-8")
                console.print(f"[dim]Exported to {target} ({len(lines)} turns).[/dim]")

            elif cmd == "/analyze-asm":
                from agent.tools.analyze_asm import analyze_asm, get_interrupt_flag

                parts = arg.split()
                if not parts:
                    console.print(
                        "[yellow]Usage: /analyze-asm <file> [--resume] [--force] [--levels N][/yellow]"
                    )
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
                    console.print(
                        f"[dim]Analyzing {path_arg}… (Ctrl+C to interrupt)[/dim]"
                    )
                    try:
                        kwargs = {
                            "path": path_arg,
                            "resume": resume,
                            "force": force_flag,
                        }
                        if max_lvls is not None:
                            kwargs["max_levels"] = max_lvls
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, lambda: analyze_asm(**kwargs)
                        )
                        if "error" in result:
                            console.print(f"[red]{result['error']}[/red]")
                        else:
                            console.print(
                                f"[green]{result.get('message', str(result))}[/green]"
                            )
                    except KeyboardInterrupt:
                        interrupt.set()
                        console.print("[yellow]Interrupted.[/yellow]")
                    except Exception as e:
                        console.print(f"[red]analyze-asm error: {e}[/red]")

            elif cmd == "/think":
                ok, msg = server.set_think_level(arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd in ("/autonomy", "/auto", "/verbose"):
                ok, msg = server.set_autonomy(arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd in ("/temperature", "/temp"):
                ok, msg = server.set_temperature(arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd == "/notify":
                ok, msg = server.set_notify(arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd == "/max_tokens":
                ok, msg = server.set_max_tokens(arg)
                console.print(f"[{'green' if ok else 'yellow'}]{msg}[/]")

            elif cmd == "/model":
                ok, msg = server.set_model(arg)
                for line in msg.splitlines():
                    console.print(f"[{'green' if ok else 'yellow'}]{line}[/]")

            elif cmd in ("/context", "/ctx", "/legend"):
                console.print(_render_context_report(server, t))

            elif cmd == "/goal":
                from agent.ui.slash import _apply_goal
                ok, msg = _apply_goal(server._agent, arg)
                for line in msg.splitlines():
                    console.print(f"[{'green' if ok else 'yellow'}]{line}[/]")

            elif cmd == "/idea":
                from agent.ui.slash_ideas import _apply_idea
                ok, msg = _apply_idea(agent, arg)
                for line in msg.splitlines():
                    console.print(f"[{'green' if ok else 'yellow'}]{line}[/]")

            elif cmd == "/ideas":
                from agent.ui.slash_ideas import _apply_ideas
                ok, msg = _apply_ideas(agent, arg)
                for line in msg.splitlines():
                    console.print(f"[{'green' if ok else 'yellow'}]{line}[/]")

            elif cmd in ("/quit", "/exit", "/q!"):
                console.print(f"[{t.text_dim}]Bye.[/{t.text_dim}]")
                return

            else:
                console.print(
                    f"[yellow]Unknown command '{cmd}'. Type /help for a list.[/yellow]"
                )

            continue

        # ── Normal message ──────────────────────────────────────────────────
        import sys

        import os as _os
        import json as _json

        _bell_enabled = _ui_cfg.get("bell_on_input_request", True)
        _title_mode = _ui_cfg.get("terminal_title", "auto")
        _title_icon = _ui_cfg.get("terminal_title_icon", "🌟")
        _title_session_mode = _ui_cfg.get("terminal_title_session", "name")

        def _session_suffix() -> str:
            if _title_session_mode == "off" or session is None:
                return ""
            if _title_session_mode == "name":
                part = session.short_name or session.id
            elif _title_session_mode == "id":
                part = session.id
            else:
                part = f"{session.short_name} ({session.id})" if session.short_name else session.id
            return f" [{part}]"

        def _set_term_title(title: str) -> None:
            if _title_mode == "off":
                return
            sys.stdout.write(f"\033]0;{title}\007")
            sys.stdout.flush()

        _set_term_title(f"{_title_icon} agent — working{_session_suffix()}")

        verbose = _os.environ.get("AGENT_VERBOSE", "").lower() in ("1", "true", "yes")

        tool_results: list[str] = []
        streaming_tokens: list[str] = []
        reasoning_active: list[bool] = [False]
        _spinner_status: list[str] = ["thinking…"]
        _spinner_stop = asyncio.Event()
        _spinner_task = asyncio.create_task(
            _run_spinner(_spinner_status, _spinner_stop, server=server)
        )

        def _clear_spinner() -> None:
            _spinner_stop.set()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

        def _args_preview(args_str: str) -> str:
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                if not isinstance(args, dict):
                    return ""
                return ", ".join(
                    f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2]
                )
            except Exception:
                return ""

        def on_tool(name: str, args_str: str) -> None:
            _clear_spinner()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            if reasoning_active[0]:
                console.print()
                reasoning_active[0] = False
            preview = _args_preview(args_str)
            suffix = f"[{t.text_dim}]({_escape(preview)})[/{t.text_dim}]" if preview else ""
            from agent.ui.render import tool_icon as _ti
            console.print(f"  [{t.tool_color}]{_ti(name)} {_escape(name)}[/{t.tool_color}] {suffix}")
            tool_results.append(name)

        def on_tool_result(name: str, ok: bool) -> None:
            mark = f"[{t.success}]✓[/{t.success}]" if ok else f"[{t.error}]✗[/{t.error}]"
            console.print(f"    {mark} [{t.text_dim}]{name}[/{t.text_dim}]")

        def on_progress(done: int, limit: int) -> None:
            _spinner_status[0] = f"iter {done}/{limit}…"

        def on_phase(label: str, detail: str = "") -> None:
            _clear_spinner()
            if streaming_tokens:
                console.print()
                streaming_tokens.clear()
            if reasoning_active[0]:
                console.print()
                reasoning_active[0] = False
            msg = f"  [{t.text_dim}]• {label}"
            if detail:
                msg += f": {detail}"
            msg += f"[/{t.text_dim}]"
            console.print(msg)
            _spinner_status[0] = f"{label}…"

        def on_reasoning(tok: str) -> None:
            if not verbose:
                return
            if not reasoning_active[0]:
                _clear_spinner()
                if streaming_tokens:
                    console.print()
                    streaming_tokens.clear()
                sys.stdout.write(f"\033[2m  ◦ ")
                reasoning_active[0] = True
            sys.stdout.write(tok)
            sys.stdout.flush()

        def on_token(token: str) -> None:
            if reasoning_active[0]:
                sys.stdout.write("\033[0m\n")
                sys.stdout.flush()
                reasoning_active[0] = False
            if not streaming_tokens:
                # First token — stop spinner and clear its line
                _spinner_stop.set()
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            streaming_tokens.append(token)
            console.print(token, end="", highlight=False)

        def _on_user_message() -> None:
            if session is not None:
                server.save_session(session)

        async def _on_loop_detected(summary: str, count: int) -> bool:
            # Pause spinner output, ask user, resume.
            _spinner_stop.set()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            console.print(
                f"[{t.warning}]⚠ loop guard: repeated tool calls ({summary}).[/{t.warning}]"
            )
            loop = asyncio.get_running_loop()
            try:
                answer = await loop.run_in_executor(
                    None, lambda: input("  Continue anyway? [y/N] ").strip().lower()
                )
            except (EOFError, KeyboardInterrupt):
                answer = ""
            return answer in ("y", "yes")

        try:
            response = await server.chat(
                user_input,
                session_id=session.id if session else "",
                on_tool_call=on_tool,
                on_tool_result=on_tool_result,
                on_token=on_token,
                on_user_message=_on_user_message,
                on_loop_detected=_on_loop_detected,
                on_progress=on_progress,
                on_phase=on_phase,
                on_reasoning=on_reasoning,
            )
        except Exception as e:
            logger.error("chat error: %s\n%s", e, traceback.format_exc())
            console.print(f"[{t.error}]Error: {e}[/{t.error}]")
            _set_term_title(f"{_title_icon} agent — error, waiting for input{_session_suffix()}")
            if _bell_enabled:
                sys.stdout.write("\007")
                sys.stdout.flush()
            continue
        finally:
            _spinner_stop.set()
            await _spinner_task

        if session is not None:
            server.save_session(session)

        _set_term_title(f"{_title_icon} agent — waiting for input{_session_suffix()}")
        if _bell_enabled:
            sys.stdout.write("\007")
            sys.stdout.flush()

        # If streaming was active, the text is already printed; just add newline.
        # If no streaming occurred (tool-only turn), print the response normally.
        if streaming_tokens:
            console.print()  # end the streaming line
        elif response:
            console.print()
            console.print(Markdown(_delatex(response)))
        elif tool_results:
            console.print(
                f"[{t.text_dim}]Done. ({', '.join(tool_results)})[/{t.text_dim}]"
            )
        else:
            console.print(f"[{t.warning}]No response from model.[/{t.warning}]")

        # Post-turn usage summary (persists after the spinner clears).
        s = server.stats()
        if s and s.get("calls", 0) > 0:
            parts = [
                f"↑{s['input_tokens']}",
                f"↓{s['output_tokens']}",
            ]
            if s.get("in_tps"):
                parts.append(f"{_fmt_tps(s['in_tps'])} in-tok/s")
            if s.get("out_tps"):
                parts.append(f"{_fmt_tps(s['out_tps'])} out-tok/s")
            if s.get("reasoning_tokens"):
                parts.append(f"think {s['reasoning_tokens']}")
            if s.get("tool_tokens"):
                parts.append(f"tool {s['tool_tokens']}")
            console.print(f"[{t.text_dim}]{'  '.join(parts)}[/{t.text_dim}]")

        if _ui_cfg["show_token_count"]:
            console.print(
                f"\n{_token_bar(server.token_estimate(), _llm_cfg['ctx_window'])}\n"
            )

    return session
