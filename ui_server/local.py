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
