"""Abstract transport interface between AgentWorker and Controller."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class AgentTransport(ABC):
    @abstractmethod
    def send_nowait(self, event: Any) -> None:
        """Send event without blocking (used from sync callbacks inside run_turn)."""

    @abstractmethod
    async def close(self) -> None:
        """Signal end of event stream."""

    @abstractmethod
    def receive(self) -> AsyncIterator[Any]:
        """Async-iterate over events until stream closes."""
