"""LocalUIServer — single-agent in-process implementation of UIServerProtocol."""
from __future__ import annotations

import asyncio
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
        self._stop_event: asyncio.Event | None = None
        self._default_max_iterations: int | None = getattr(agent.config.llm, "max_iterations", 10)
        from agent.notify import NotifyBroker
        # Speech-to-text intake (off by default). Built before the broker so the
        # broker can route inbound chunked `voice` frames into it; the intake in
        # turn answers pending questions via the broker or starts a new turn.
        # UI-supplied handler that turns a dictated transcript into a real
        # prompt (starts a turn when idle, injects mid-turn). Set by the active
        # UI via set_external_prompt_handler; until then we fall back to inject.
        self._external_prompt = None
        self._intake = None
        on_voice = None
        if getattr(getattr(agent.config, "speech", None), "enabled", False):
            try:
                from agent.speech import SpeechIntake, get_transcriber
                self._intake = SpeechIntake(
                    transcriber=get_transcriber(agent.config),
                    config=agent.config.speech,
                    on_answer=lambda a: self._notify.submit_answer(a),
                    on_transcript=self._on_voice_transcript,
                    agent_dir=getattr(agent.config.tools, "agent_dir", ".agent"),
                )
                on_voice = self._intake.feed
                logger.info("speech: intake enabled (backend=%s model=%s)",
                            agent.config.speech.backend, agent.config.speech.model)
            except Exception:
                logger.exception("speech: intake disabled (init failed)")
                self._intake = None
                on_voice = None
        self._notify = NotifyBroker(agent.config, on_voice=on_voice)

    def start_notify(self) -> None:
        """Open notify channel connections proactively (so inbound answers/voice
        arrive before the agent's first outbound notice). Call from the UI once
        an event loop is running."""
        try:
            self._notify.start()
        except Exception:
            logger.exception("notify: proactive start failed")

    def set_external_prompt_handler(self, cb) -> None:
        """Register a UI callback (text -> None) that submits a dictated prompt
        as if the user typed and sent it. Thread-safe expectations are the UI's."""
        self._external_prompt = cb

    def submit_external_prompt(self, text: str, *, source: str = "external") -> None:
        """Submit `text` as if the user typed and sent it.

        Prefers the registered UI prompt handler (so it starts/steers a turn);
        falls back to the agent inject queue (picked up when idle). Shared by
        voice dictation and remote/delegated chat (build_ui_server on_chat).
        `source` only labels logging.
        """
        if not text:
            return
        cb = self._external_prompt
        if cb is not None:
            try:
                cb(text)
                logger.info("%s: prompt routed to UI handler: %r", source, text)
                return
            except Exception:
                logger.exception("%s: external prompt handler failed; falling back to inject", source)
        logger.warning("%s: no UI prompt handler — queued via inject (idle = waits for next input): %r",
                       source, text)
        self._agent.inject(text)

    def _on_voice_transcript(self, text: str) -> None:
        """Route a finished voice transcript via the shared external-prompt path."""
        self.submit_external_prompt(text, source="speech")

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
        on_signal=None,
        source: str = "terminal",
    ) -> str:
        from agent.core.turn_signals import parse_signal

        self._stop_event = asyncio.Event()

        ts_cfg = getattr(self._agent.config, "turn_signals", None)
        signals_enabled = ts_cfg is None or getattr(ts_cfg, "enabled", True)
        max_auto_steps = getattr(ts_cfg, "max_auto_steps", 20) if ts_cfg else 20

        current_input = text
        auto_step = 0

        while True:
            # Tell remote clients we are working on the reply (resets each
            # iteration so an auto-stepping loop keeps showing "busy").
            self._notify.notify_state("busy", session_id=session_id)
            try:
                response = await self._agent.chat(
                    current_input,
                    on_token=on_token,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_progress=on_progress,
                    on_loop_detected=on_loop_detected,
                    on_phase=on_phase,
                    on_reasoning=on_reasoning,
                    on_context_size=on_context_size,
                    on_user_message=on_user_message,
                    stop_event=self._stop_event,
                    source=source,
                )
            except Exception as exc:
                # Surface a crash to remote clients instead of leaving them stuck
                # on "busy"; then let the normal error handling propagate.
                self._notify.notify_state("error", str(exc)[:200], session_id)
                raise

            if not signals_enabled:
                self._notify.notify_response(response, session_id)
                self._notify.notify_state("idle", session_id=session_id)
                return response

            clean_response, signal = parse_signal(response)

            if signal is None:
                self._notify.notify_response(clean_response, session_id)
                self._notify.notify_state("idle", session_id=session_id)
                return clean_response

            # Strip signal text from the last assistant message in history.
            msgs = self._agent.get_messages()
            if msgs and msgs[-1].get("role") == "assistant":
                content = msgs[-1].get("content", "")
                if isinstance(content, str) and content != clean_response:
                    msgs[-1] = {**msgs[-1], "content": clean_response}
                    self._agent.set_messages(msgs)

            # Notify UI of the signal.
            if on_signal is not None:
                try:
                    on_signal(signal, clean_response)
                except Exception:
                    logger.exception("on_signal callback failed")

            # External notification channels (off by default).
            if signal.kind in ("ask_user", "blocked") and self._notify.remote_answers:
                # Bidirectional: push as Question, wait for a remote answer
                # (timeout falls through to the normal return-to-UI path).
                from agent.notify.messages import Question
                answer = await self._notify.ask(Question(
                    kind=signal.kind, text=signal.payload,
                    free_text=True, session=session_id,
                ))
                if answer is not None and (answer.text or answer.choice):
                    auto_step = 0  # remote answer counts as user input
                    self._stop_event = asyncio.Event()
                    current_input = answer.text or answer.choice
                    continue
            elif signal.kind == "done":
                # End of a Q/A round: deliver the full answer (clean_response),
                # not just the post-marker payload, so the remote client rounds
                # off with the reply in its list / TTS.
                self._notify.notify_response(
                    clean_response or signal.payload or "done", session_id)
            else:
                # Fire-and-forget notice (non-blocking).
                self._notify.handle_signal(signal.kind, signal.payload, session_id)

            if signal.kind == "next_step" and auto_step < max_auto_steps:
                if self._stop_event is not None and self._stop_event.is_set():
                    return clean_response
                auto_step += 1
                self._stop_event = asyncio.Event()
                current_input = signal.payload
                continue

            # done, ask_user, request_feedback, request_review, consult_crows, blocked
            # — all pause the auto-loop and return to the UI.
            waiting = signal.kind in (
                "ask_user", "blocked", "request_feedback", "request_review", "consult_crows",
            )
            self._notify.notify_state("waiting" if waiting else "idle", session_id=session_id)
            return clean_response

    # ── runtime controls ─────────────────────────────────────────────────────

    def stop_after_iteration(self, session_id: str = "") -> None:
        """Request graceful stop after the current tool-call iteration finishes."""
        if self._stop_event is not None:
            self._stop_event.set()

    def set_unlimited_mode(self, enabled: bool, session_id: str = "") -> None:
        """Toggle unlimited iterations (None) vs the configured default."""
        self._agent.config.llm.max_iterations = None if enabled else self._default_max_iterations

    def is_unlimited_mode(self, session_id: str = "") -> bool:
        return getattr(self._agent.config.llm, "max_iterations", 10) is None

    def get_goal(self, session_id: str = "") -> str | None:
        return getattr(self._agent.config.llm, "goal", None)

    def set_goal(self, goal: str | None, session_id: str = "") -> None:
        self._agent.config.llm.goal = goal
        if hasattr(self._agent.config, "agent"):
            self._agent.config.agent.goal = goal

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
        from agent.core.model_status import get_availability
        avail = get_availability()
        return {
            "llm": {
                "model": llm.model or "",
                "base_url": llm.base_url or "",
                "ctx_window": llm.ctx_window or 0,
                "max_output_tokens": llm.max_output_tokens,
                "temperature": llm.temperature,
                "think_level": getattr(llm, "think_level", "normal"),
                "max_iterations": getattr(llm, "max_iterations", 10),
                "goal": getattr(llm, "goal", None),
                "goal_max_iterations": getattr(llm, "goal_max_iterations", 200),
                "available": avail.get("llm"),
            },
            "emb": {
                "model": emb.model or "",
                "base_url": emb.base_url or "",
                "dimensions": emb.dimensions,
                "max_tokens": emb.max_tokens,
                "available": avail.get("emb"),
            },
            "sum": {
                "model": sum_entry.model if sum_entry else llm.model or "",
                "base_url": sum_entry.base_url if sum_entry else llm.base_url or "",
                "ctx_window": sum_entry.ctx_window if sum_entry else llm.ctx_window or 0,
                "max_output_tokens": sum_entry.max_output_tokens if sum_entry else llm.max_output_tokens,
                "temperature": sum_entry.temperature if sum_entry else llm.temperature,
                "source": sum_name if sum_entry else "(fallback to llm)",
                "available": avail.get("sum"),
            },
        }

    def probe_model_availability(self, session_id: str = "") -> dict:
        """Probe each role's endpoint for model availability; cache the result."""
        from agent.config.model_probe import check_model_availability
        from agent.core.model_status import set_availability
        try:
            avail = check_model_availability(self._agent.config)
        except Exception:
            avail = {}
        if avail:
            set_availability(avail)
        return avail

    def refresh_model_info(self, session_id: str = "") -> dict:
        from agent.config.model_probe import refresh_ctx_windows
        cfg = self._agent.config
        updated = refresh_ctx_windows(cfg)
        # Propagate to cfg.llm if the active default entry was updated
        active = cfg.model_roles.get("default", "")
        if active and active in updated:
            cfg.llm.ctx_window = updated[active]
        # Refresh availability alongside ctx sizes (same /models data).
        self.probe_model_availability(session_id)
        return {"updated": updated, "llm_ctx": cfg.llm.ctx_window}

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
            "terminal_title_icon": getattr(cfg.ui, "terminal_title_icon", "🌟"),
            "terminal_title_session": getattr(cfg.ui, "terminal_title_session", "name"),
            "qa_summary_mode": getattr(cfg.ui, "qa_summary_mode", "lazy"),
            "spinner_animation": getattr(cfg.ui, "spinner_animation", "box"),
            "show_active_models": getattr(cfg.ui, "show_active_models", "auto"),
        }

    def get_turn_id(self, session_id: str = "") -> int:
        return getattr(self._agent, "_turn_id", 0)

    async def summarize_session_qa(
        self,
        entries: list,
        session_id: str = "",
        force: bool = False,
    ) -> "tuple[str, str]":
        """Generate Q and A session-level summaries. Returns (q_text, a_text)."""
        import asyncio
        from agent.memory.session import _get_session_dir, get_session_subpath
        from agent.memory.session_summarizer import generate

        sid = session_id or getattr(self._agent, "_session_id", "") or ""
        if not sid or not entries:
            return "", ""
        session_dir = _get_session_dir() / get_session_subpath(sid)
        config = self._agent.config
        q_text, a_text = await asyncio.gather(
            generate(session_dir, entries, "q", config, force=force),
            generate(session_dir, entries, "a", config, force=force),
        )
        return q_text, a_text

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

    def set_autonomy(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_autonomy
        return _apply_autonomy(self._agent, arg)

    def set_temperature(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_temperature
        return _apply_temperature(self._agent, arg)

    def set_max_tokens(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_max_tokens
        return _apply_max_tokens(self._agent, arg)

    def set_max_iter(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_max_iter
        return _apply_max_iter(self._agent, arg)

    def set_model(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_model
        return _apply_model(self._agent, arg)

    def set_plan(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_plan
        return _apply_plan(self._agent, arg)

    def set_notify(self, arg: str, session_id: str = "") -> "tuple[bool, str]":
        from agent.ui.slash import _apply_notify
        return _apply_notify(self._agent, arg, self._notify)

    # ── session persistence ────────────────────────────────────────────────────

    def save_session(self, session: "Session", session_id: str = "") -> None:
        from agent.memory.session import save_session
        save_session(session, self._agent.get_messages())

    def rate_session(self, outcome: str, voter: str = "user", session_id: str = "") -> dict:
        from agent.tools.rate_session.rate_session import rate_session as _rate
        return _rate(outcome=outcome, voter=voter, session_id=session_id or None)

    def load_session(self, name: str, session_id: str = "") -> "tuple[Session | None, list[dict]]":
        from agent.memory.session import load_session
        session, messages = load_session(name)
        if session is not None:
            messages = [
                {k: v for k, v in m.items() if not k.startswith("_")}
                for m in messages
            ]
        return session, messages
