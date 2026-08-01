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
        """Rebuild the chat log from session data.

        Turns are rendered from ``_chat_qa_data``: the last
        ``chat_restore_expand_last`` turns in full (Markdown response, tool
        icons, file diffs, round-stats line — same output as before the app
        closed), older turns folded to a one-line summary. Clicking a turn
        toggles its fold."""
        # ── build per-turn (q_d, a_d) data: pair user messages with Q/A log
        #    entries by index, falling back to raw message content.
        qa_list: list[tuple] = [
            (q or {}, a or {}) for _tid, q, a in (qa_entries or [])
        ]
        pairs: list[tuple] = []   # (q_text, tool_names, a_texts)
        cur: "list | None" = None
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                continue
            content = m.get("content") or ""
            if isinstance(content, list):
                content = _json.dumps(content)
            if role == "user":
                if cur is not None:
                    pairs.append(tuple(cur))
                cur = [content, [], []]
            elif role == "assistant" and cur is not None:
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cur[1].append(tc.get("function", {}).get("name", "?"))
                if content:
                    cur[2].append(content)
        if cur is not None:
            pairs.append(tuple(cur))

        self._chat_qa_data = []
        for i, (q_text, tool_names, a_texts) in enumerate(pairs):
            if i < len(qa_list):
                q_d, a_d = dict(qa_list[i][0]), dict(qa_list[i][1])
            else:
                q_d, a_d = {}, {}
            q_d.setdefault("content", q_text)
            a_d.setdefault("content", "\n\n".join(a_texts))
            if not a_d.get("tool_calls"):
                a_d["tool_calls"] = tool_names
            self._chat_qa_data.append((q_d, a_d))

        # ── fold defaults: expand the tail, fold the rest
        try:
            expand_last = int(self._server.get_ui_config().get(
                "chat_restore_expand_last", 3))
        except Exception:
            expand_last = 3
        n = len(self._chat_qa_data)
        self._chat_folded = set(range(max(0, n - expand_last)))
        # Fresh session load: no turn has been touched by hand yet. Reset
        # (not merge) so a session switch never carries stale pins forward.
        self._chat_fold_pinned = set()
        self._chat_restored_count = n
        self._chat_resume_marker = resume_marker
        self._rerender_chat()

    def _rerender_chat(self) -> None:
        """Clear and re-render the whole chat log from ``_chat_qa_data``,
        honouring per-turn fold state and rebuilding all click-target maps."""
        t = self._t
        _one_line = self._wt._one_line
        chat_log = self.query_one("#chat-log", self._wt.ConversationView)
        chat_log.clear()
        self._chat_user_lines = []
        self._chat_model_lines = {}
        self._chat_file_lines = {}
        self._chat_changeset_lines = {}
        self._chat_line_to_ordinal = []
        folded = getattr(self, "_chat_folded", set())
        marker_after = (getattr(self, "_chat_restored_count", 0) - 1
                        if getattr(self, "_chat_resume_marker", False) else None)
        current_ordinal: list[int] = [-1]

        def _cw(line) -> None:
            """Write one item and extend the ordinal map by the visual lines it produced."""
            before = len(chat_log.lines)
            chat_log.write(line)
            added = len(chat_log.lines) - before
            self._chat_line_to_ordinal.extend([current_ordinal[0]] * max(added, 1))

        for ordinal, (q_d, a_d) in enumerate(self._chat_qa_data):
            current_ordinal[0] = ordinal
            self._chat_user_lines.append(len(chat_log.lines))
            if ordinal in folded:
                self._render_folded_turn(_cw, q_d, a_d, _one_line, t)
            else:
                self._render_full_turn(_cw, chat_log, q_d, a_d, t)
            if marker_after is not None and ordinal == marker_after:
                current_ordinal[0] = -1
                _cw(f"[{t.text_dim}]─── resumed ───[/{t.text_dim}]")
        chat_log.scroll_end(animate=False)

    def _render_folded_turn(self, _cw, q_d: dict, a_d: dict, _one_line, t) -> None:
        q = (q_d.get("summary_q") or q_d.get("content") or "").strip()
        a = (a_d.get("summary_a") or a_d.get("content") or "").strip()
        tools = a_d.get("tool_calls") or []
        a_part = a or (f"{len(tools)} tool call{'s' if len(tools) != 1 else ''}" if tools else "(no reply)")
        _cw(
            f"[{t.text_dim}]▸[/{t.text_dim}] "
            f"[bold {t.user_color}]You:[/bold {t.user_color}] {_escape(_one_line(q, limit=80, wrap=False))}"
            f"  [{t.text_dim}]·[/{t.text_dim}] "
            f"[{t.agent_color}]A:[/{t.agent_color}] [{t.text_dim}]{_escape(_one_line(a_part, limit=100, wrap=False))}[/{t.text_dim}]"
        )

    def _render_full_turn(self, _cw, chat_log, q_d: dict, a_d: dict, t) -> None:
        from rich.markdown import Markdown as _Markdown
        from agent.ui.render import _delatex, tool_icon as _ti
        q = (q_d.get("content") or "").strip()
        _cw(f"[{t.text_dim}]▾[/{t.text_dim}] [bold {t.user_color}]You:[/bold {t.user_color}] {_escape(q)}")
        for name in a_d.get("tool_calls") or []:
            _cw(f"[{t.tool_color}]  {_ti(name)} {name}[/{t.tool_color}]")
        # Restored turns render identically to live ones: from_a_data prefers
        # the record's own "changeset" key (exact diffs, foreign-edit flags)
        # and falls back to the legacy "modified_files" list for old sessions.
        from agent.core.changeset import from_a_data
        from agent.ui.event_mixin import write_changeset_rows
        write_changeset_rows(self, _cw, chat_log, from_a_data(a_d))
        a = (a_d.get("content") or "").strip()
        if a:
            _cw(f"[bold {t.agent_color}]Agent:[/bold {t.agent_color}]")
            _cw(_Markdown(_delatex(a)))
        # Round-stats line — clickable, same as live turns.
        mc = a_d.get("model_calls") or []
        dur = a_d.get("duration") or 0.0
        if mc:
            from agent.metrics import model_calls as _mcm
            from collections import Counter
            counts = Counter(c.get("tier", "local") for c in mc)
            line = _mcm.format_line(dict(counts), duration=dur or None)
            before = len(chat_log.lines)
            _cw(f"[{t.text_dim}]{line}  [i]▸ click for detail[/i][/{t.text_dim}]")
            payload = {"calls": mc, "duration": dur}
            for li in range(before, len(chat_log.lines)):
                self._chat_model_lines[li] = payload

    def _toggle_chat_fold(self, ordinal: int) -> None:
        """Fold/unfold one turn and re-render. No-op while a turn is streaming.

        A manual toggle pins the turn: auto-fold (see ``_auto_fold_turn``)
        will never touch it again, in either direction, until the session
        is restored fresh."""
        if getattr(self, "_agent_running", False):
            return
        if not (0 <= ordinal < len(getattr(self, "_chat_qa_data", []))):
            return
        if not hasattr(self, "_chat_folded"):
            self._chat_folded = set()
        if not hasattr(self, "_chat_fold_pinned"):
            self._chat_fold_pinned = set()
        self._chat_fold_pinned.add(ordinal)
        if ordinal in self._chat_folded:
            self._chat_folded.discard(ordinal)
        else:
            self._chat_folded.add(ordinal)
        self._rerender_chat()

    def _fold_journal_mode(self) -> str:
        """``config.ui.changeset.fold_journal``, read defensively — a bare or
        partial config (e.g. in tests, or an old config predating this
        setting) must still yield the documented default rather than raise."""
        try:
            cfg = getattr(getattr(self._server, "_agent", None), "config", None)
            changeset_cfg = getattr(getattr(cfg, "ui", None), "changeset", None)
            return getattr(changeset_cfg, "fold_journal", None) or "on_next_round"
        except Exception:
            return "on_next_round"

    def _auto_fold_turn(self, ordinal: int) -> None:
        """Fold turn *ordinal* unless the user pinned it (see
        ``_toggle_chat_fold``) or it is already folded. Re-renders through
        the same path a manual toggle uses — no second rendering route."""
        if not (0 <= ordinal < len(getattr(self, "_chat_qa_data", []))):
            return
        if ordinal in getattr(self, "_chat_fold_pinned", set()):
            return
        if not hasattr(self, "_chat_folded"):
            self._chat_folded = set()
        if ordinal in self._chat_folded:
            return
        self._chat_folded.add(ordinal)
        self._rerender_chat()

    def _auto_fold_finished_round(self) -> None:
        """Call once a round has just been appended to ``_chat_qa_data``
        (see ``_append_qa_turn``). Only ``fold_journal == "immediately"``
        acts here; the default "on_next_round" instead waits for
        ``_auto_fold_before_new_round``, and "never" never folds."""
        if self._fold_journal_mode() != "immediately":
            return
        self._auto_fold_turn(len(getattr(self, "_chat_qa_data", [])) - 1)

    def _auto_fold_before_new_round(self) -> None:
        """Call when a new round begins, before it is appended to
        ``_chat_qa_data`` — so ``_chat_qa_data[-1]`` is still the round that
        just became "previous". Only ``fold_journal == "on_next_round"``
        (the default) acts here."""
        if self._fold_journal_mode() != "on_next_round":
            return
        self._auto_fold_turn(len(getattr(self, "_chat_qa_data", [])) - 1)

    # ── per-session UI state (active tab, …) ────────────────────────────────

    def _ui_state_path(self):
        from agent.memory.session import _get_session_dir, get_session_subpath
        if not self._session:
            return None
        return _get_session_dir() / get_session_subpath(self._session.id) / "ui_state.json"

    def _save_ui_state(self) -> None:
        path = self._ui_state_path()
        if path is None:
            return
        try:
            from textual.widgets import TabbedContent
            state = {"active_tab": self.query_one(TabbedContent).active}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(state), encoding="utf-8")
        except Exception:
            logger.debug("_save_ui_state failed (ignored)", exc_info=True)

    def _restore_ui_state(self) -> None:
        path = self._ui_state_path()
        if path is None or not path.exists():
            return
        try:
            from textual.widgets import TabbedContent
            state = _json.loads(path.read_text(encoding="utf-8"))
            tab = state.get("active_tab")
            if tab:
                self.query_one(TabbedContent).active = tab
        except Exception:
            logger.debug("_restore_ui_state failed (ignored)", exc_info=True)

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
        cs = getattr(self, "_last_changeset", None)
        if cs:
            file_parts = []
            for fc in cs.files[:4]:
                stat = f"+{fc.added}/-{fc.removed}" if (fc.added or fc.removed) else ""
                file_parts.append(f"{fc.path} ({stat})" if stat else fc.path)
            action_bits.append("✎ " + " ".join(file_parts))
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
            cs = getattr(self, "_last_changeset", None)
            a_data = {
                "turn_id": turn_id,
                "content": response or "",
                "tool_calls": list(self._last_tool_calls),
                "modified_files": [
                    {"path": fc.path, "added": fc.added, "removed": fc.removed}
                    for fc in (cs.files if cs else [])
                ],
            }
            # Carry the full changeset too (diffs, foreign-edit flags) so a
            # fold/unfold re-render of this turn later in the same session
            # looks identical to how it looked live — see
            # view_mixin._render_full_turn / event_mixin.write_changeset_rows.
            if cs:
                from agent.core.changeset import to_json as _cs_to_json
                a_data["changeset"] = _cs_to_json(cs)
            # Round stats so fold/unfold re-renders keep the clickable line.
            try:
                from agent.metrics import model_calls as _mcm
                a_data["model_calls"] = _mcm.round_detail()
                a_data["duration"] = _mcm.round_duration()
            except Exception:
                pass
            # Keep chat_qa_data in sync so click-to-expand works for live turns too.
            if not hasattr(self, "_chat_qa_data"):
                self._chat_qa_data = []
            self._chat_qa_data.append((q_data, a_data))
            self._auto_fold_finished_round()
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
