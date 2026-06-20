"""RelayLink — agent-side websocket link that carries event/control frames.

RelayChannel (agent/notify/channels.py) is coupled to Notice/Question messages.
The remote UI needs to push *arbitrary* event frames (RemoteBridge output) and
receive *control* frames, so this is a thin, frame-agnostic sibling: connect as
the agent role, send pre-serialized JSON frames, and hand each inbound frame to
a callback. Optional E2E reuses the same AES-GCM box as the notify channel.

Connection management (hello with version, reconnect/backoff, bounded queue,
frame-size cap) mirrors RelayChannel so behaviour is consistent across both.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from agent.notify.channels import (
    RELAY_QUEUE_MAX,
    RELAY_BACKOFF_MAX_S,
    RELAY_MAX_FRAME_BYTES,
)
from agent.notify.messages import NOTIFY_PROTOCOL_VERSION

logger = logging.getLogger(__name__)

OnFrame = Callable[[dict], Any]  # may be sync or return an awaitable


class RelayLink:
    def __init__(
        self,
        url: str,
        token: str,
        *,
        name: str = "agent-ui",
        on_frame: OnFrame | None = None,
        e2e: Any | None = None,
    ) -> None:
        self.url = url
        self._token = token
        self._name = name
        self._on_frame = on_frame
        self._e2e = e2e
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=RELAY_QUEUE_MAX)
        self._task: asyncio.Task | None = None

    # ── outbound ────────────────────────────────────────────────────────────

    def send_frame(self, frame: str) -> None:
        """Queue a serialized frame for delivery. Sync — safe from callbacks.

        Drops the oldest frame when the queue is full rather than blocking the
        turn (a slow/absent client must never stall the agent).
        """
        self._ensure_task()
        if self._e2e is not None:
            try:
                frame = json.dumps(self._e2e.encrypt(json.loads(frame)))
            except Exception:
                logger.exception("relay_link: e2e encrypt failed — dropping frame")
                return
        while True:
            try:
                self._queue.put_nowait(frame)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    logger.warning("relay_link %s: queue full — dropped oldest frame", self._name)
                except asyncio.QueueEmpty:
                    pass

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    def start(self) -> None:
        self._ensure_task()

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        import websockets
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url, max_size=RELAY_MAX_FRAME_BYTES) as ws:
                    await ws.send(json.dumps({
                        "type": "hello", "role": "agent", "v": NOTIFY_PROTOCOL_VERSION,
                        "token": self._token, "name": self._name,
                    }))
                    backoff = 1
                    pumps = (
                        asyncio.create_task(self._pump_out(ws)),
                        asyncio.create_task(self._pump_in(ws)),
                    )
                    try:
                        done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            t.result()
                    finally:
                        for t in pumps:
                            t.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("relay_link %s: %s — reconnecting in %ss", self._name, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RELAY_BACKOFF_MAX_S)

    async def _pump_out(self, ws) -> None:
        while True:
            await ws.send(await self._queue.get())

    async def _pump_in(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes) or self._on_frame is None:
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            if self._e2e is not None:
                if data.get("type") != "enc":
                    logger.warning("relay_link %s: plaintext frame dropped (e2e enabled)", self._name)
                    continue
                inner = self._e2e.decrypt(data)
                if inner is None:
                    logger.warning("relay_link %s: undecryptable frame dropped", self._name)
                    continue
                data = inner
            try:
                res = self._on_frame(data)
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                logger.exception("relay_link %s: on_frame handler failed", self._name)
