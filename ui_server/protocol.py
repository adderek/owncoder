"""UIServerProtocol — interface between user interfaces and the backend.

UIs should program against this protocol, not Agent directly.
Phase 1: LocalUIServer wraps a single Agent in-process.
Later: replace with a transport-backed implementation for remote/multi-UI access.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.memory.session import Session


@runtime_checkable
class UIServerProtocol(Protocol):
    """What every UI needs from the backend."""

    async def chat(
        self,
        text: str,
        session_id: str,
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
        """Send a user message; stream events via callbacks; return full response."""
        ...

    def inject(self, text: str, session_id: str) -> None:
        """Inject a message into an active turn (interrupt injection)."""
        ...

    def pending_background_count(self, session_id: str = "") -> int:
        """Number of background tasks still running."""
        ...

    def cancel_background(self, session_id: str = "") -> int:
        """Cancel pending background tasks. Returns number cancelled."""
        ...

    async def wait_background(self, session_id: str = "", timeout: float | None = None) -> int:
        """Wait for background tasks. Returns remaining count."""
        ...

    def stats(self, session_id: str) -> dict:
        """Token usage stats for the session."""
        ...

    def token_estimate(self, session_id: str) -> int:
        """Current context token estimate."""
        ...

    def context_breakdown(self, session_id: str) -> list[dict]:
        """Context breakdown by segment for display."""
        ...

    def output_breakdown(self, session_id: str, scope: str = "session") -> list[dict]:
        """Output token breakdown by type."""
        ...

    def set_session_id(self, session_id: str) -> None:
        """Associate backend state (qa_log, facts_store) with this session."""
        ...

    # ── message management ───────────────────────────────────────────────────

    def message_count(self, session_id: str = "") -> int:
        """Number of messages in context (including system messages)."""
        ...

    def get_messages(self, session_id: str = "") -> list[dict]:
        """Copy of current message list."""
        ...

    def set_messages(self, messages: list[dict], session_id: str = "") -> None:
        """Replace current message list."""
        ...

    def reset_messages(self, session_id: str = "") -> None:
        """Clear conversation history, keeping only the first system message."""
        ...

    async def compact_messages(self, session_id: str = "") -> None:
        """Compact messages in place using LLM summarization."""
        ...

    # ── read-only state accessors ─────────────────────────────────────────────

    def get_llm_info(self, session_id: str = "") -> dict:
        """LLM display info: model, ctx_window, compaction_threshold."""
        ...

    def get_ui_config(self, session_id: str = "") -> dict:
        """UI display config: theme, mode, chat_wrap, round_summary, show_token_count."""
        ...

    def get_peak_tokens(self, session_id: str = "") -> "tuple[int, int]":
        """Returns (round_peak_tokens, last_round_peak_tokens)."""
        ...

    def get_store_stats(self, session_id: str = "") -> "dict | None":
        """RAG store stats {"files": int, "chunks": int} or None if unavailable."""
        ...

    # ── runtime config mutation ───────────────────────────────────────────────

    def set_think_level(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        """Set/query think level. Returns (ok, message)."""
        ...

    def set_temperature(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        """Set/query temperature. Returns (ok, message)."""
        ...

    def set_max_tokens(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        """Set/query max output tokens / ctx_window. Returns (ok, message)."""
        ...

    def set_model(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        """Switch model entry. Returns (ok, message)."""
        ...

    def set_plan(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        """Handle plan subcommands. Returns (ok, message)."""
        ...

    # ── session persistence ───────────────────────────────────────────────────

    def save_session(self, session: "Session", session_id: str = "") -> None:
        """Persist session + current messages to disk."""
        ...

    def load_session(self, name: str, session_id: str = "") -> "tuple[Session | None, list[dict]]":
        """Load session by id/name. Returns (session, messages) or (None, [])."""
        ...
