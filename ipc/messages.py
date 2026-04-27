"""Event messages passed from AgentWorker → Controller over a transport."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenEvent:
    token: str


@dataclass
class ReasoningEvent:
    token: str


@dataclass
class ToolCallEvent:
    name: str
    args: str


@dataclass
class ToolResultEvent:
    name: str
    ok: bool


@dataclass
class PhaseEvent:
    label: str
    detail: str = ""


@dataclass
class UsageEvent:
    data: dict[str, Any]


@dataclass
class ContextSizeEvent:
    tokens: int


@dataclass
class ProgressEvent:
    current: int
    total: int


@dataclass
class TruncationEvent:
    pass


@dataclass
class LoopDetectedEvent:
    """Bidirectional: worker sends, controller resolves _decision to unblock worker."""
    summary: str
    max_count: int
    _decision: asyncio.Future | None = field(default=None, repr=False, compare=False)

    async def resolve(self, decision: bool) -> None:
        if self._decision is not None and not self._decision.done():
            self._decision.set_result(decision)

    async def wait(self) -> bool:
        if self._decision is None:
            return False
        return await self._decision


@dataclass
class TurnDoneEvent:
    response: str
    messages: list[dict]


@dataclass
class ErrorEvent:
    exception: BaseException
