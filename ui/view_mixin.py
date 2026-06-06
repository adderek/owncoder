"""View helper mixin for CodeAgentApp.

Accesses self._t (theme), self._wt (widget-type namespace from build_widget_classes),
and standard app state attributes set in CodeAgentApp.__init__.
"""
from __future__ import annotations

import json as _json
import logging

from rich.markup import escape as _escape

logger = logging.getLogger(__name__)


class ViewMixin:
    """Chat/sys/QA view write helpers."""

    def _write_sys(self, text: str, switch_tab: bool = True) -> None:
        from textual.widgets import TabbedContent
        self._sys_messages.append(text)
        self.query_one("#sys-log", self._wt.SysView).write(text)
        if switch_tab:
            self.query_one(TabbedContent).active = "tab-sys"

    def _write_chat(self, text) -> None:
        self.query_one("#chat-log", self._wt.ConversationView).write(text)

    def _switch_to_chat(self) -> None:
        from textual.widgets import TabbedContent
        self.query_one(TabbedContent).active = "tab-chat"

    def _restore_chat_history(
        self,
        messages: list,
        resume_marker: bool = False,
        qa_entries: "list | None" = None,
    ) -> None:
        t = self._t
        _one_line = self._wt._one_line
        chat_log = self.query_one("#chat-log", self._wt.ConversationView)
        chat_log.clear()
        self._chat_user_lines = []
        self._chat_qa_data: list[tuple] = []
        self._chat_line_to_ordinal: list[int] = []
        current_ordinal: list[int] = [-1]  # mutable cell; -1 = pre-first-turn lines

        def _cw(line) -> None:
            """Write one line to chat_log and extend the ordinal map by however many visual lines result."""
            before = len(chat_log.lines)
            chat_log.write(line)
            added = len(chat_log.lines) - before
            self._chat_line_to_ordinal.extend([current_ordinal[0]] * max(added, 1))

        # Build parallel Q/A data and summary lists from Q/A log (best-effort by turn count).
        qa_list: list[tuple] = []
        q_summaries: list[str] = []
        a_summaries: list[str] = []
        if qa_entries:
            for _tid, q, a in qa_entries:
                qa_list.append((q or {}, a or {}))
                q_summaries.append((q or {}).get("summary_q", "") or "")
                a_summaries.append((a or {}).get("summary_a", "") or "")

        q_idx = 0
        a_idx = 0
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
                q_d, a_d = qa_list[q_idx] if q_idx < len(qa_list) else ({"content": content}, {})
                self._chat_qa_data.append((q_d, a_d))
                display = (q_summaries[q_idx] if q_idx < len(q_summaries) else "") or content
                current_ordinal[0] = q_idx
                q_idx += 1
                _cw(f"[bold {t.user_color}]You:[/bold {t.user_color}] {_escape(_one_line(display, wrap=False))}")
            elif role == "assistant":
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("function", {}).get("name", "?")
                        from agent.ui.render import tool_icon as _ti
                        _cw(f"[{t.tool_color}]  {_ti(name)} {name}[/{t.tool_color}]")
                if content:
                    summary = (a_summaries[a_idx] if a_idx < len(a_summaries) else "") or content
                    _cw(f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {_escape(_one_line(summary, wrap=False))}")
                    a_idx += 1
        if resume_marker:
            _cw(f"[{t.text_dim}]─── resumed ───[/{t.text_dim}]")

    def _reload_sys_view(self) -> None:
        sys_log = self.query_one("#sys-log", self._wt.SysView)
        sys_log.clear()
        for msg in self._sys_messages:
            sys_log.write(msg)

    def _reload_qa_views(self) -> None:
        if not self._session:
            return
        try:
            from agent.memory.qa_log import read_history_sync
            entries = read_history_sync(self._session.id)
        except Exception:
            logger.exception("_reload_qa_views: read_history_sync failed (ignored)")
            return
        self._last_qa_entries = entries
        try:
            self.query_one("#q-log", self._wt.QView).load_history(entries)
            self.query_one("#a-log", self._wt.AView).load_history(entries)
            self.query_one("#sparse-log", self._wt.SparseView).load_history(entries)
        except Exception:
            logger.exception("_reload_qa_views: view update failed (ignored)")
        # Load cached summaries from disk (no LLM call).
        self._load_cached_qa_summaries(entries)
        # Background mode: kick off session-level summarization only if new turns exist.
        mode = self._server.get_ui_config().get("qa_summary_mode", "lazy")
        if mode == "background" and entries and self._session_summary_is_stale(entries):
            self._start_qa_summary_worker(entries)

    def _session_summary_is_stale(self, entries: list) -> bool:
        """Return True if session-level Q/A summary needs regeneration."""
        if not entries or not self._session:
            return False
        try:
            from agent.memory.session import _get_session_dir, get_session_subpath
            from agent.memory.session_summarizer import load_stored
            session_dir = _get_session_dir() / get_session_subpath(self._session.id)
            last_turn_id = entries[-1][0]
            for scope in ("q", "a"):
                stored = load_stored(session_dir, scope)
                if not stored.get("content"):
                    return True
                if stored.get("summarized_up_to_turn", -1) < last_turn_id:
                    return True
            return False
        except Exception:
            return True  # on error, assume stale

    def _load_cached_qa_summaries(self, entries: list) -> None:
        if not self._session:
            return
        try:
            from agent.memory.session import _get_session_dir, get_session_subpath
            from agent.memory.session_summarizer import load_stored
            session_dir = _get_session_dir() / get_session_subpath(self._session.id)
            for scope, wid, cls in (
                ("q", "#q-summary-log", self._wt.QSummaryView),
                ("a", "#a-summary-log", self._wt.ASummaryView),
            ):
                stored = load_stored(session_dir, scope)
                content = stored.get("content", "")
                if content:
                    self.query_one(wid, cls).set_summary(content)
        except Exception:
            logger.exception("_load_cached_qa_summaries: failed (ignored)")

    def _write_round_summary(self, user_text: str, response: str) -> None:
        t = self._t
        _one_line = self._wt._one_line
        q = _one_line((user_text or "").strip(), limit=80, wrap=False)
        a_src = (response or "").strip()
        a = _one_line(a_src, limit=80, wrap=False) if a_src else ""
        tools = list(dict.fromkeys(self._last_tool_calls))
        action_bits: list[str] = []
        if tools:
            from agent.ui.render import tool_icon as _ti
            action_bits.append(", ".join(f"{_ti(n)} {n}" for n in tools[:4]))
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
        try:
            turn_id = self._server.get_turn_id()
            q_data = {"turn_id": turn_id, "content": user_text}
            a_data = {
                "turn_id": turn_id,
                "content": response or "",
                "tool_calls": list(self._last_tool_calls),
                "modified_files": list(self._modified_files),
            }
            # Keep chat_qa_data in sync so click-to-expand works for live turns too.
            if not hasattr(self, "_chat_qa_data"):
                self._chat_qa_data = []
            self._chat_qa_data.append((q_data, a_data))
            self.query_one("#q-log", self._wt.QView).add_turn(turn_id, q_data, a_data)
            self.query_one("#a-log", self._wt.AView).add_turn(turn_id, q_data, a_data)
            self.query_one("#sparse-log", self._wt.SparseView).add_turn(turn_id, q_data, a_data)
        except Exception:
            logger.exception("_append_qa_turn: failed (ignored)")
        # Invalidate or immediately re-summarize depending on mode
        mode = self._server.get_ui_config().get("qa_summary_mode", "lazy")
        if mode == "background":
            self._start_qa_summary_worker()
        elif mode == "lazy":
            self._qa_summary_dirty = True

    def _start_qa_summary_worker(self, entries: list | None = None) -> None:
        """Launch or re-launch the async Q/A summary worker."""
        self.run_worker(
            self._run_qa_summary(entries),
            exclusive=True,
            group="qa-summary",
            name="qa-summary",
        )

    async def _run_qa_summary(self, entries: list | None = None) -> None:
        """Async worker: call server to generate Q+A session summaries."""
        if not self._session:
            return
        try:
            if entries is None:
                import asyncio as _asyncio
                from agent.memory.qa_log import read_history_sync
                entries = await _asyncio.get_event_loop().run_in_executor(
                    None, read_history_sync, self._session.id
                )
        except Exception:
            logger.exception("_run_qa_summary: read_history_sync failed (ignored)")
            return

        if not entries:
            return

        try:
            self.query_one("#q-summary-log", self._wt.QSummaryView).set_loading()
            self.query_one("#a-summary-log", self._wt.ASummaryView).set_loading()
        except Exception:
            pass

        try:
            q_text, a_text = await self._server.summarize_session_qa(
                entries, session_id=self._session.id
            )
            self._qa_summary_dirty = False
        except Exception:
            logger.exception("_run_qa_summary: summarize_session_qa failed (ignored)")
            return

        try:
            self.query_one("#q-summary-log", self._wt.QSummaryView).set_summary(q_text)
            self.query_one("#a-summary-log", self._wt.ASummaryView).set_summary(a_text)
        except Exception:
            logger.exception("_run_qa_summary: widget update failed (ignored)")
