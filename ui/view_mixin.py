"""View helper mixin for CodeAgentApp.

Accesses self._t (theme), self._wt (widget-type namespace from build_widget_classes),
and standard app state attributes set in CodeAgentApp.__init__.
"""
from __future__ import annotations

import json as _json
import logging

from rich.markup import escape as _escape
from rich.markdown import Markdown

logger = logging.getLogger(__name__)


class ViewMixin:
    """Chat/sys/QA view write helpers."""

    def _write_sys(self, text: str) -> None:
        from textual.widgets import TabbedContent
        self._sys_messages.append(text)
        self.query_one("#sys-log", self._wt.SysView).write(text)
        self.query_one(TabbedContent).active = "tab-sys"

    def _write_chat(self, text) -> None:
        self.query_one("#chat-log", self._wt.ConversationView).write(text)

    def _switch_to_chat(self) -> None:
        from textual.widgets import TabbedContent
        self.query_one(TabbedContent).active = "tab-chat"

    def _restore_chat_history(self, messages: list) -> None:
        t = self._t
        _one_line = self._wt._one_line
        chat_log = self.query_one("#chat-log", self._wt.ConversationView)
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
                        chat_log.write(f"[{t.tool_color}]  ⚙ {name}[/{t.tool_color}]")
                if content:
                    if self._wrap_enabled:
                        chat_log.write(f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]")
                        chat_log.write(Markdown(content.strip()))
                    else:
                        chat_log.write(
                            f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}] {_escape(_one_line(content, wrap=False))}"
                        )

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
        try:
            self.query_one("#q-log", self._wt.QView).load_history(entries)
            self.query_one("#a-log", self._wt.AView).load_history(entries)
            self.query_one("#sparse-log", self._wt.SparseView).load_history(entries)
        except Exception:
            logger.exception("_reload_qa_views: view update failed (ignored)")

    def _write_round_summary(self, user_text: str, response: str) -> None:
        t = self._t
        _one_line = self._wt._one_line
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
        try:
            turn_id = getattr(self._agent, "_turn_id", 0)
            q_data = {"turn_id": turn_id, "content": user_text}
            a_data = {
                "turn_id": turn_id,
                "content": response or "",
                "tool_calls": list(self._last_tool_calls),
                "modified_files": list(self._modified_files),
            }
            self.query_one("#q-log", self._wt.QView).add_turn(turn_id, q_data, a_data)
            self.query_one("#a-log", self._wt.AView).add_turn(turn_id, q_data, a_data)
            self.query_one("#sparse-log", self._wt.SparseView).add_turn(turn_id, q_data, a_data)
        except Exception:
            logger.exception("_append_qa_turn: failed (ignored)")
