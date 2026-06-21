"""Slash-command handler mixin for CodeAgentApp.

Accesses self._t, self._wt, self._server, self._session, and helper methods
defined on CodeAgentApp (or ViewMixin): _write_sys, _begin_chat, etc.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from rich.markup import escape as _escape

from agent.ui.render import _render_context_report, _OUT_SEGMENT_COLORS

logger = logging.getLogger(__name__)


def _find_loop_guard_stop(messages: list[dict]) -> str | None:
    """Return the loop-guard stop note from the last assistant message, or None."""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            content = (m.get("content") or "").strip()
            return content if content.startswith("[loop guard:") else None
    return None


class SlashHandlerMixin:
    """Handles /command dispatch for the Textual app."""

    async def _run_slash(self, cmd: str, arg: str) -> None:
        t = self._t

        if cmd == "/help":
            from agent.ui.readline_loop import _make_help_text
            self._write_sys(_make_help_text(t))

        elif cmd == "/q":
            from textual.widgets import TabbedContent
            self.query_one(TabbedContent).active = "tab-q"

        elif cmd == "/a":
            from textual.widgets import TabbedContent
            self.query_one(TabbedContent).active = "tab-a"

        elif cmd == "/sparse":
            from textual.widgets import TabbedContent
            self.query_one(TabbedContent).active = "tab-sparse"

        elif cmd == "/paths":
            from textual.widgets import TabbedContent
            subcmd = arg.strip().split(None, 1)
            sub = subcmd[0].lower() if subcmd else ""
            rest = subcmd[1] if len(subcmd) > 1 else ""

            if not sub or sub == "show":
                self.query_one(TabbedContent).active = "tab-paths"
                try:
                    self._reload_paths_view()
                except Exception:
                    pass

            elif sub == "add":
                parts = rest.split()
                if not parts:
                    self._write_sys(
                        f"[{t.warning}]Usage: /paths add <path> [ro|rw][/{t.warning}]"
                    )
                else:
                    raw_path = parts[0]
                    mode = parts[1].lower() if len(parts) > 1 else "rw"
                    if mode not in ("ro", "rw"):
                        mode = "rw"
                    from agent.security import path_grants as _pg
                    from pathlib import Path as _Path
                    resolved = _Path(raw_path).resolve()
                    _pg.add_grant(resolved, mode, origin="user")
                    self._write_sys(
                        f"[{t.success}]✓ Added path grant: {resolved} ({mode})[/{t.success}]"
                    )
                    self.query_one(TabbedContent).active = "tab-paths"
                    try:
                        self._reload_paths_view()
                    except Exception:
                        pass

            elif sub == "remove":
                if not rest.strip():
                    self._write_sys(
                        f"[{t.warning}]Usage: /paths remove <path>[/{t.warning}]"
                    )
                else:
                    from agent.security import path_grants as _pg
                    from pathlib import Path as _Path
                    resolved = _Path(rest.strip()).resolve()
                    ok = _pg.remove_grant(resolved)
                    if ok:
                        self._write_sys(f"[{t.success}]Removed grant: {resolved}[/{t.success}]")
                    else:
                        self._write_sys(f"[{t.warning}]No removable grant found for: {resolved}[/{t.warning}]")
                    try:
                        self._reload_paths_view()
                    except Exception:
                        pass

            elif sub == "list":
                from agent.security import path_grants as _pg
                grants = _pg.get_all()
                if not grants:
                    self._write_sys(f"[{t.text_dim}](no path grants)[/{t.text_dim}]")
                else:
                    lines = ["[bold]Path grants:[/bold]"]
                    for g in grants:
                        state = f"[{t.warning}]pending[/{t.warning}]" if g.state == "pending" else f"[{t.success}]granted[/{t.success}]"
                        lines.append(f"  {g.path}  [{t.text_dim}]{g.origin}[/{t.text_dim}]  {g.mode.upper()}  {state}")
                    self._write_sys("\n".join(lines))

            else:
                self._write_sys(
                    f"[{t.warning}]Usage: /paths [show|add <path> [ro|rw]|remove <path>|list][/{t.warning}]"
                )

        elif cmd == "/compact":
            before = self._server.token_estimate()
            self._write_sys(f"[{t.text_dim}]Compacting…[/{t.text_dim}]")
            try:
                await self._server.compact_messages()
                after = self._server.token_estimate()
                self._write_sys(f"[{t.success}]Compacted.[/{t.success}] {before} → {after} tokens")
            except Exception as e:
                self._write_sys(f"[{t.error}]Compact failed: {e}[/{t.error}]")
            self._refresh_token_bar()

        elif cmd in ("/continue", "/c"):
            self._begin_chat("continue")

        elif cmd == "/goal":
            from agent.ui.slash import _apply_goal
            ok, msg = _apply_goal(self._server._agent, arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd == "/clear":
            self.query_one("#chat-log", self._wt.ConversationView).clear()
            self._switch_to_chat()

        elif cmd == "/tokens":
            used = self._server.token_estimate()
            info = self._server.get_llm_info()
            peak, last_peak = self._server.get_peak_tokens()
            self._write_sys(
                f"tokens: {used}/{info['ctx_window']}  "
                f"({self._server.message_count()} messages)  "
                f"peak: {peak}  prev-round peak: {last_peak}"
            )

        elif cmd in ("/context", "/ctx", "/legend"):
            self._write_sys(_render_context_report(self._server, t))
            self._refresh_token_bar()

        elif cmd in ("/output", "/out"):
            scope = (arg.strip().lower() or "session")
            if scope not in ("session", "last"):
                self._write_sys(f"[{t.warning}]Usage: /output [session|last][/{t.warning}]")
            else:
                breakdown = self._server.output_breakdown(scope)
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
                        f"  [{color}]█[/{color}] {seg['label']:<10} {tok:>7,}  ({pct:5.1f}%)"
                    )
                self._write_sys(f"  {'total':<12} {total:>7,}")
                try:
                    out_bar = self.query_one("#output-breakdown", self._wt.OutputBreakdownBar)
                    out_bar.set_segments(
                        breakdown,
                        scope_label="out" if scope == "session" else "turn",
                    )
                except Exception:
                    pass

        elif cmd in ("/perf", "/timing"):
            from agent.metrics.turn_metrics import run_perf_command
            _sl = getattr(self._server._agent, "_side_log", None)
            _dir = getattr(_sl, "session_dir", None) if _sl is not None else None
            self._write_sys(_escape(run_perf_command(_dir)))

        elif cmd == "/reset":
            self._server.reset_messages()
            self._write_sys(f"[{t.text_dim}]Conversation history cleared.[/{t.text_dim}]")

        elif cmd == "/tools":
            from agent.tools import get_schemas
            names = [s["function"]["name"] for s in get_schemas()]
            self._write_sys("Tools: " + "  ".join(names))

        elif cmd == "/skills":
            from agent.skills import run_skills_command
            self._write_sys(_escape(run_skills_command(self._server._agent.config, arg)))

        elif cmd in ("/checkpoint", "/cp"):
            from agent.core.checkpoint import run_checkpoint_command
            self._write_sys(_escape(run_checkpoint_command(arg)))

        elif cmd == "/mcp":
            from agent.mcp import run_mcp_command
            self._write_sys(_escape(run_mcp_command(self._server._agent.config, arg)))

        elif cmd in ("/security", "/sec", "/audit"):
            import asyncio
            from agent.security.secaudit import run_security_command, _security_start_banner
            _cfg = self._server._agent.config
            _parts = arg.strip().split()
            _sub = _parts[0].lower() if _parts else ""
            _slow = _sub in ("review", "triage", "verify", "full")
            if _sub == "review":
                # Direct call with live per-window progress marshalled to the UI thread.
                self._write_sys(_escape(_security_start_banner(_cfg, _parts)))
                from agent.security.review import run_review_command
                _rest = arg.strip()[len("review"):].strip()
                _prog = lambda m: self.call_from_thread(self._write_sys, _escape(m))  # noqa: E731
                _out = await asyncio.to_thread(run_review_command, _cfg, _rest, _prog)
            elif _slow:
                self._write_sys(_escape(_security_start_banner(_cfg, _parts)))
                # Offload the blocking run so the banner paints and the UI stays
                # responsive during the long LLM/scan work.
                _out = await asyncio.to_thread(run_security_command, _cfg, arg)
            else:
                _out = run_security_command(_cfg, arg)
            self._write_sys(_escape(_out))

        elif cmd == "/save":
            if arg.strip():
                from agent.memory.session import _sanitize_short_name
                self._session.short_name = _sanitize_short_name(arg.strip())
                self._session.name = arg.strip()
            self._server.save_session(self._session)
            label = self._session.short_name or self._session.id
            self._write_sys(f"[{t.text_dim}]Saved session '{label}'.[/{t.text_dim}]")

        elif cmd == "/load":
            if not arg.strip():
                self._write_sys(f"[{t.warning}]Usage: /load <session-id-or-short-name>[/{t.warning}]")
            else:
                loaded_session, loaded_msgs = self._server.load_session(arg.strip())
                if loaded_session is None:
                    self._write_sys(f"[{t.warning}]Session '{arg.strip()}' not found.[/{t.warning}]")
                else:
                    self._server.set_messages(loaded_msgs)
                    self._session = loaded_session
                    label = loaded_session.short_name or loaded_session.id
                    self._refresh_token_bar()
                    self._reload_qa_views()
                    self._restore_chat_history(
                        loaded_msgs,
                        resume_marker=True,
                        qa_entries=getattr(self, "_last_qa_entries", None),
                    )
                    self._switch_to_chat()
                    self._write_sys(
                        f"[{t.text_dim}]Loaded session '{label}' "
                        f"({len(loaded_msgs)} messages).[/{t.text_dim}]"
                    )
                    _loop_guard_resume_note = _find_loop_guard_stop(loaded_msgs)
                    if _loop_guard_resume_note:
                        self._write_sys(
                            f"[{t.warning}]Note: session ended with loop-guard stop — "
                            f"type a message to redirect (e.g. 're-read the file and retry').[/{t.warning}]"
                        )
                        self._write_sys(f"[{t.text_dim}]{_loop_guard_resume_note[:200]}[/{t.text_dim}]")

        elif cmd == "/sessions":
            from agent.memory.session import list_sessions
            import datetime
            sessions = [s for s in list_sessions() if s.get("message_count", 0) > 0]
            if not sessions:
                self._write_sys(f"[{t.text_dim}]No sessions found.[/{t.text_dim}]")
            for s in sessions:
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
                name_extra = (
                    f"  [{t.text_dim}]{s['name']}[/{t.text_dim}]" if s.get("name") else ""
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
            for m in self._server.get_messages():
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
            ok, msg = self._server.set_think_level(arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd in ("/autonomy", "/auto", "/verbose"):
            ok, msg = self._server.set_autonomy(arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd in ("/temperature", "/temp"):
            ok, msg = self._server.set_temperature(arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd == "/notify":
            ok, msg = self._server.set_notify(arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd in ("/maxiter", "/max_iter"):
            ok, msg = self._server.set_max_iter(arg)
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
            self.query_one("#chat-log", self._wt.ConversationView).clear()
            msgs = self._server.get_messages()
            if msgs:
                self._restore_chat_history(msgs)
            self._reload_qa_views()

        elif cmd in ("/round-summary", "/summary"):
            from agent.ui.prefs import load_prefs, save_prefs
            self._round_summary_enabled = not self._round_summary_enabled
            state = "enabled" if self._round_summary_enabled else "disabled"
            p = load_prefs()
            p["round_summary"] = self._round_summary_enabled
            save_prefs(p)
            self._write_sys(f"[{t.success}]Round summary {state}.[/{t.success}]")

        elif cmd == "/model":
            if arg.strip() == "refresh":
                self._write_sys(f"[{t.text_dim}]Probing model endpoints…[/{t.text_dim}]")
                try:
                    result = await asyncio.to_thread(self._server.refresh_model_info)
                    updated = result.get("updated", {})
                    llm_ctx = result.get("llm_ctx", 0)
                    if updated:
                        parts = [
                            f"{k}={v // 1024}k" if v >= 1024 else f"{k}={v}"
                            for k, v in updated.items()
                            if k != "__emb__"
                        ]
                        if "__emb__" in updated:
                            v = updated["__emb__"]
                            parts.append(f"emb={v // 1024}k" if v >= 1024 else f"emb={v}")
                        self._write_sys(
                            f"[{t.success}]Refreshed ctx: {', '.join(parts)}"
                            f"  llm_ctx={llm_ctx // 1024}k[/{t.success}]"
                        )
                    else:
                        self._write_sys(f"[{t.text_dim}]No context sizes returned from endpoints.[/{t.text_dim}]")
                except Exception as e:
                    self._write_sys(f"[{t.error}]Refresh failed: {e}[/{t.error}]")
                self._refresh_token_bar()
            else:
                ok, msg = self._server.set_model(arg)
                color = t.success if ok else t.warning
                for line in msg.splitlines():
                    self._write_sys(f"[{color}]{line}[/{color}]")
                if ok:
                    self._refresh_token_bar()

        elif cmd == "/models":
            import asyncio
            from agent.ui.slash import _render_models_table
            from rich.console import Console
            from io import StringIO
            cfg = self._server._agent.config

            def _render() -> str:
                # Probe runs off the UI thread — a dead endpoint must not freeze it.
                tbl = _render_models_table(cfg)
                buf = StringIO()
                Console(file=buf, highlight=False, width=300).print(tbl)
                return buf.getvalue().rstrip()

            self._write_sys(f"[{t.text_dim}]probing endpoints…[/{t.text_dim}]")
            out = await asyncio.to_thread(_render)
            self._write_sys(out)

        elif cmd == "/plan":
            ok, msg = self._server.set_plan(arg)
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
            ok, msg = self._server.set_plan(sub_map[cmd])
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd in ("/quit", "/exit", "/q!"):
            self.action_quit()
            return

        elif cmd in ("/unlimited", "/nomax"):
            enabled = not self._server.is_unlimited_mode()
            self._server.set_unlimited_mode(enabled)
            if enabled:
                self._write_sys(f"[{t.success}]Unlimited iterations ON — Ctrl+C to stop after current iteration.[/{t.success}]")
            else:
                limit = getattr(self._server._agent.config.llm, "max_iterations", 10)
                self._write_sys(f"[{t.text_dim}]Unlimited iterations OFF — limit restored to {limit}.[/{t.text_dim}]")

        elif cmd == "/idea":
            from agent.ui.slash_ideas import _apply_idea
            ok, msg = _apply_idea(self._server._agent, arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd == "/ideas":
            from agent.ui.slash_ideas import _apply_ideas
            ok, msg = _apply_ideas(self._server._agent, arg)
            color = t.success if ok else t.warning
            for line in msg.splitlines():
                self._write_sys(f"[{color}]{line}[/{color}]")

        elif cmd == "/recoveries":
            from agent.planning import recovery
            recs = recovery.scan_pending()
            if not recs:
                self._write_sys(f"[{t.text_dim}]No pending crash recoveries.[/{t.text_dim}]")
            else:
                for r in recs:
                    self._write_sys(
                        f"[{t.warning}]{r.session_id}[/{t.warning}] [dim]{r.exception}[/dim]"
                    )

        elif cmd == "/resummarize":
            if not self._session:
                self._write_sys(f"[{t.warning}]No active session.[/{t.warning}]")
            else:
                force = "--force" in (arg or "")
                sid = self._session.id
                self._write_sys(f"[{t.text_dim}]Re-summarizing {'all' if force else 'stale'} entries…[/{t.text_dim}]")

                async def _do_resummarize() -> None:
                    from agent.summarizer import resummarize_session
                    try:
                        updated, skipped = await resummarize_session(
                            self._server._agent.config, sid, force=force
                        )
                        self._write_sys(
                            f"[{t.success}]Re-summarized {updated} entr{'y' if updated == 1 else 'ies'},"
                            f" {skipped} skipped.[/{t.success}]"
                        )
                        self._reload_qa_views()
                    except Exception as exc:
                        self._write_sys(f"[{t.error}]resummarize failed: {exc}[/{t.error}]")

                self.run_worker(_do_resummarize(), exclusive=False, name="resummarize")

        else:
            self._write_sys(f"[{t.warning}]Unknown command '{cmd}'. Type /help.[/{t.warning}]")

    async def _run_analyze_asm(self, arg: str) -> None:
        t = self._t
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
