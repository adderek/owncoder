"""NotifyBroker — fan-out to channels, pending-question registry.

Never blocks the agent loop: notices are fire-and-forget tasks; questions
resolve via Future with timeout + configured default (same pattern as
ipc.messages.LoopDetectedEvent).

Answer validation (security boundary): an answer is data resolving exactly
one pending question — it must match a pending question id and, unless the
question allowed free text, one of the offered options. Question ids are
single-use; late or duplicate answers are dropped.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from agent.notify.channels import build_channel
from agent.notify.messages import Answer, Notice, Question
from agent.config.models import NotifyConfig

if TYPE_CHECKING:
    from agent.config.models import Config

logger = logging.getLogger(__name__)


class NotifyBroker:
    def __init__(self, config: "Config") -> None:
        # Tolerate partial configs (test fakes, older callers): no [notify]
        # section behaves as disabled with zero channels.
        self._cfg = getattr(config, "notify", None) or NotifyConfig()
        self._channels = [
            ch for cfg in self._cfg.channels
            if (ch := build_channel(cfg, on_answer=self._on_wire_answer)) is not None
        ]
        self._pending: dict[str, tuple[Question, asyncio.Future]] = {}
        self._tasks: set[asyncio.Task] = set()

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled and self._channels)

    @property
    def remote_answers(self) -> bool:
        """True when ask_user/blocked signals should wait for a remote answer."""
        return bool(
            self.enabled
            and getattr(self._cfg, "remote_answers", False)
            and any(ch.capability in ("choices", "chat") for ch in self._channels)
        )

    def status(self) -> str:
        cfg = self._cfg
        state = "on" if cfg.enabled else "off"
        if not self._channels:
            return f"notify {state} — no usable channels configured"
        chans = ", ".join(f"{c.name}[{c.capability}]" for c in self._channels)
        return (
            f"notify {state} — events: {', '.join(cfg.events)}; "
            f"channels: {chans}; pending questions: {len(self._pending)}"
        )

    # ── outbound ─────────────────────────────────────────────────────────────

    def handle_signal(self, kind: str, payload: str, session_id: str = "") -> None:
        """Push a turn signal to channels if enabled and subscribed. Non-blocking."""
        if not self.enabled or kind not in self._cfg.events:
            return
        self._spawn(self._fanout(Notice(kind=kind, text=payload, session=session_id)))

    async def ask(self, question: Question) -> "Answer | None":
        """Fan out a question; wait for first valid answer.

        Returns None when no answer arrived and on_timeout="continue" with no
        default option. Channels below "choices" capability get the question
        rendered as a notice.
        """
        if not self.enabled:
            return None
        cfg = self._cfg
        loop = asyncio.get_running_loop()
        if cfg.answer_timeout_s > 0:
            question.expires_at = loop.time() + cfg.answer_timeout_s
        future: asyncio.Future = loop.create_future()
        self._pending[question.id] = (question, future)
        try:
            await self._fanout(question)
            timeout = None if cfg.on_timeout == "wait" else cfg.answer_timeout_s
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                if question.default:
                    return Answer(question_id=question.id, choice=question.default, source="timeout")
                return None
        finally:
            self._pending.pop(question.id, None)

    # ── inbound ──────────────────────────────────────────────────────────────

    def _on_wire_answer(self, data: dict) -> None:
        """Channel callback: wire answer dict → validated Answer."""
        self.submit_answer(Answer(
            question_id=str(data.get("id", "")),
            choice=str(data.get("choice", "") or ""),
            text=str(data.get("text", "") or ""),
            source=str(data.get("from", "") or "user"),
        ))

    def submit_answer(self, answer: Answer) -> bool:
        """Validate and deliver an answer. Returns False if rejected."""
        entry = self._pending.get(answer.question_id)
        if entry is None:
            logger.warning("notify: answer for unknown/expired question %r dropped", answer.question_id)
            return False
        question, future = entry
        if answer.choice and answer.choice not in question.options:
            logger.warning("notify: answer choice %r not among offered options — dropped", answer.choice)
            return False
        if not answer.choice and not question.free_text:
            logger.warning("notify: free-text answer to options-only question %r — dropped", question.id)
            return False
        if future.done():
            return False
        future.set_result(answer)
        del self._pending[answer.question_id]
        self._spawn(self._fanout(Notice(
            kind="info", text=f"question {question.id} answered by {answer.source}",
            session=question.session,
        )))
        return True

    def stop(self) -> None:
        """Cancel background work (relay connections, in-flight fanouts)."""
        for ch in self._channels:
            stop = getattr(ch, "stop", None)
            if stop is not None:
                stop()
        for task in list(self._tasks):
            task.cancel()

    # ── internals ────────────────────────────────────────────────────────────

    async def _fanout(self, msg: "Notice | Question") -> None:
        results = await asyncio.gather(
            *(ch.send(msg) for ch in self._channels), return_exceptions=True
        )
        for ch, res in zip(self._channels, results):
            if isinstance(res, BaseException):
                logger.warning("notify channel %s raised: %s", ch.name, res)

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            # No running loop (sync caller outside async context) — skip silently.
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
