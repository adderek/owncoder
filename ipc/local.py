"""In-process transport backed by asyncio.Queue."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from .transport import AgentTransport

_SENTINEL = object()


class LocalTransport(AgentTransport):
    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    def send_nowait(self, event: Any) -> None:
        self._queue.put_nowait(event)

    async def close(self) -> None:
        self._queue.put_nowait(_SENTINEL)

    async def receive(self) -> AsyncIterator[Any]:
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                return
            yield item
