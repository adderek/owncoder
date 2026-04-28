"""LocalUIServer — single-agent in-process implementation of UIServerProtocol."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.memory.session import Session

logger = logging.getLogger(__name__)


class LocalUIServer:
    """Wraps a single Agent; session_id is accepted but not used for routing (one agent).

    Satisfies UIServerProtocol. Later phases swap this for a transport-backed
    server that can route between multiple agents/sessions.
    """

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    # ── chat ────────────────────────────────────────────────────────────────

    async def chat(
        self,
        text: str,
        session_id: str = "",
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
        on_usage=None,
        on_progress=None,
        on_loop_detected=None,
        on_phase=None,
        on_reasoning=None,
        on_context_size=None,
        on_user_message=None,
    ) -> str:
        return await self._agent.chat(
            text,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_progress=on_progress,
            on_loop_detected=on_loop_detected,
            on_phase=on_phase,
            on_reasoning=on_reasoning,
            on_context_size=on_context_size,
            on_user_message=on_user_message,
        )

    # ── control ─────────────────────────────────────────────────────────────

    def inject(self, text: str, session_id: str = "") -> None:
        self._agent.inject(text)

    def pending_background_count(self, session_id: str = "") -> int:
        return self._agent.pending_background_count()

    def cancel_background(self, session_id: str = "") -> int:
        return self._agent.cancel_background()

    async def wait_background(self, session_id: str = "", timeout: float | None = None) -> int:
        return await self._agent.wait_background(timeout=timeout)

    # ── state queries ────────────────────────────────────────────────────────

    def stats(self, session_id: str = "") -> dict:
        return dict(self._agent.stats)

    def token_estimate(self, session_id: str = "") -> int:
        return self._agent.token_estimate()

    def context_breakdown(self, session_id: str = "") -> list[dict]:
        return self._agent.context_breakdown()

    def output_breakdown(self, session_id: str = "", scope: str = "session") -> list[dict]:
        return self._agent.output_breakdown(scope=scope)

    # ── session ──────────────────────────────────────────────────────────────

    def set_session_id(self, session_id: str) -> None:
        self._agent.set_session_id(session_id)

    # ── message management ────────────────────────────────────────────────────

    def message_count(self, session_id: str = "") -> int:
        return self._agent.message_count()

    def get_messages(self, session_id: str = "") -> list[dict]:
        return self._agent.get_messages()

    def set_messages(self, messages: list[dict], session_id: str = "") -> None:
        self._agent.set_messages(messages)

    def reset_messages(self, session_id: str = "") -> None:
        self._agent.reset_messages()

    async def compact_messages(self, session_id: str = "") -> None:
        await self._agent.compact_messages()

    # ── read-only state accessors ──────────────────────────────────────────────

    def get_llm_info(self, session_id: str = "") -> dict:
        cfg = self._agent.config
        return {
            "model": cfg.llm.model or "",
            "ctx_window": cfg.llm.ctx_window or 0,
            "compaction_threshold": getattr(cfg.llm, "compaction_threshold", 0.75),
        }

    def get_model_configs(self, session_id: str = "") -> dict:
        """Return display config for each model role (llm, emb, sum)."""
        cfg = self._agent.config
        llm = cfg.llm
        emb = cfg.embeddings
        # Resolve summarizer: check model_roles → model_entries, fallback to llm
        sum_entry = None
        roles = getattr(cfg, "model_roles", {})
        entries = getattr(cfg, "model_entries", {})
        sum_name = roles.get("summarizer") or roles.get("sum")
        if sum_name:
            sum_entry = entries.get(sum_name)
        if sum_entry is None:
            sum_entry = entries.get("summarizer")
        return {
            "llm": {
                "model": llm.model or "",
                "base_url": llm.base_url or "",
                "ctx_window": llm.ctx_window or 0,
                "max_output_tokens": llm.max_output_tokens,
                "temperature": llm.temperature,
                "think_level": getattr(llm, "think_level", "normal"),
                "max_iterations": getattr(llm, "max_iterations", 10),
            },
            "emb": {
                "model": emb.model or "",
                "base_url": emb.base_url or "",
                "dimensions": emb.dimensions,
                "max_tokens": emb.max_tokens,
            },
            "sum": {
                "model": sum_entry.model if sum_entry else llm.model or "",
                "base_url": sum_entry.base_url if sum_entry else llm.base_url or "",
                "ctx_window": sum_entry.ctx_window if sum_entry else llm.ctx_window or 0,
                "max_output_tokens": sum_entry.max_output_tokens if sum_entry else llm.max_output_tokens,
                "temperature": sum_entry.temperature if sum_entry else llm.temperature,
                "source": sum_name if sum_entry else "(fallback to llm)",
            },
        }

    def get_ui_config(self, session_id: str = "") -> dict:
        cfg = self._agent.config
        return {
            "theme": cfg.ui.theme,
            "mode": getattr(cfg.ui, "mode", "simple"),
            "chat_wrap": getattr(cfg.ui, "chat_wrap", "last used"),
            "round_summary": bool(getattr(cfg.ui, "round_summary", True)),
            "show_token_count": bool(getattr(cfg.ui, "show_token_count", False)),
            "reasoning_fold": getattr(cfg.ui, "reasoning_fold", "end_of_round"),
            "bell_on_input_request": bool(getattr(cfg.ui, "bell_on_input_request", True)),
            "terminal_title": getattr(cfg.ui, "terminal_title", "auto"),
        }

    def get_turn_id(self, session_id: str = "") -> int:
        return getattr(self._agent, "_turn_id", 0)

    def get_peak_tokens(self, session_id: str = "") -> "tuple[int, int]":
        return (
            getattr(self._agent, "round_peak_tokens", 0),
            getattr(self._agent, "last_round_peak_tokens", 0),
        )

    def get_store_stats(self, session_id: str = "") -> "dict | None":
        store = getattr(self._agent, "store", None)
        if store is None:
            return None
        try:
            return store.stats()
        except Exception:
            return None

    # ── runtime config mutation ────────────────────────────────────────────────

    def set_think_level(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_think
        return _apply_think(self._agent, arg)

    def set_temperature(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_temperature
        return _apply_temperature(self._agent, arg)

    def set_max_tokens(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_max_tokens
        return _apply_max_tokens(self._agent, arg)

    def set_model(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_model
        return _apply_model(self._agent, arg)

    def set_plan(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_plan
        return _apply_plan(self._agent, arg)

    # ── session persistence ────────────────────────────────────────────────────

    def save_session(self, session: "Session", session_id: str = "") -> None:
        from agent.memory.session import save_session
        save_session(session, self._agent.get_messages())

    def load_session(self, name: str, session_id: str = "") -> "tuple[Session | None, list[dict]]":
        from agent.memory.session import load_session
        session, messages = load_session(name)
        if session is not None:
            messages = [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in messages
            ]
        return session, messages
