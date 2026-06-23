"""SpeechIntake — reassemble chunked ``voice`` frames and route the transcript.

A remote client streams one utterance as several ``voice`` frames (audio is too
large for a single relay frame). ``feed`` is called for each frame (sync, from
the relay channel's inbound pump); it buffers chunks per utterance id and, on
the final chunk, transcribes the reassembled blob off the event loop and routes
the text:

  - frame carries ``answer_to`` → ``on_answer(Answer(...))`` (resolves a pending
    notify question via the broker's existing single-use validation),
  - otherwise → ``on_transcript(text)`` (a brand-new user turn).

Best-effort, never backpressure: a partial utterance over the byte cap, older
than the TTL, or beyond the concurrent-utterance cap is dropped with a log —
the same philosophy as the relay send queue.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Callable, TYPE_CHECKING

from agent.notify.messages import Answer

if TYPE_CHECKING:
    from agent.config.models import SpeechConfig
    from agent.speech.stt import Transcriber

logger = logging.getLogger(__name__)


class _Utterance:
    __slots__ = ("chunks", "nbytes", "created", "fmt", "lang", "answer_to", "done")

    def __init__(self, fmt: str, lang: str, answer_to: str) -> None:
        self.chunks: dict[int, bytes] = {}
        self.nbytes = 0
        self.created = time.monotonic()
        self.fmt = fmt
        self.lang = lang
        self.answer_to = answer_to
        self.done = False


class SpeechIntake:
    def __init__(
        self,
        transcriber: "Transcriber",
        config: "SpeechConfig",
        on_answer: "Callable[[Answer], object]",
        on_transcript: "Callable[[str], object]",
    ) -> None:
        self._tx = transcriber
        self._cfg = config
        self._on_answer = on_answer
        self._on_transcript = on_transcript
        self._buffers: dict[str, _Utterance] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def pending(self) -> int:
        return len(self._buffers)

    def feed(self, wire: dict) -> None:
        """Ingest one decrypted ``voice`` frame. Sync; schedules transcription."""
        if not isinstance(wire, dict) or wire.get("type") != "voice":
            return
        uid = str(wire.get("id", "") or "")
        if not uid:
            return
        self._expire()
        ut = self._buffers.get(uid)
        if ut is None:
            if len(self._buffers) >= self._cfg.max_concurrent_utterances:
                logger.warning("speech: too many concurrent utterances — dropping %s", uid)
                return
            ut = _Utterance(
                fmt=str(wire.get("fmt", "wav") or "wav"),
                lang=str(wire.get("lang", "") or ""),
                answer_to=str(wire.get("answer_to", "") or ""),
            )
            self._buffers[uid] = ut

        try:
            seq = int(wire.get("seq", 0))
        except (TypeError, ValueError):
            seq = 0
        chunk = self._b64(wire.get("data", ""))
        if chunk and seq not in ut.chunks:
            if ut.nbytes + len(chunk) > self._cfg.max_utterance_bytes:
                logger.warning("speech: utterance %s over byte cap — dropping", uid)
                self._buffers.pop(uid, None)
                return
            ut.chunks[seq] = chunk
            ut.nbytes += len(chunk)

        if wire.get("last"):
            self._buffers.pop(uid, None)
            if not ut.done:
                ut.done = True
                self._spawn(self._finish(uid, ut))

    # ── internals ─────────────────────────────────────────────────────────────

    async def _finish(self, uid: str, ut: _Utterance) -> None:
        audio = b"".join(ut.chunks[s] for s in sorted(ut.chunks))
        if not audio:
            return
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None, self._tx.transcribe, audio, ut.fmt, ut.lang
            )
        except Exception:
            logger.exception("speech: transcription failed for utterance %s", uid)
            return
        text = (text or "").strip()
        if not text:
            logger.info("speech: utterance %s produced no text", uid)
            return
        logger.info("speech: utterance %s → %r (answer_to=%r)", uid, text, ut.answer_to)
        try:
            if ut.answer_to:
                self._on_answer(Answer(question_id=ut.answer_to, text=text, source="voice"))
            else:
                self._on_transcript(text)
        except Exception:
            logger.exception("speech: routing transcript failed for %s", uid)

    def _expire(self) -> None:
        ttl = self._cfg.utterance_ttl_s
        if ttl <= 0:
            return
        now = time.monotonic()
        stale = [k for k, v in self._buffers.items() if now - v.created > ttl]
        for k in stale:
            logger.warning("speech: utterance %s expired (ttl %ss) — dropping", k, ttl)
            self._buffers.pop(k, None)

    @staticmethod
    def _b64(data) -> bytes:
        if not isinstance(data, str) or not data:
            return b""
        try:
            return base64.b64decode(data, validate=False)
        except Exception:
            return b""

    def _spawn(self, coro) -> None:
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
