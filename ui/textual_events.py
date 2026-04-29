"""Textual Message events for agent UI callbacks."""
from __future__ import annotations


def build_event_classes() -> "SimpleNamespace":
    """Build Textual Message event classes. Called once inside the app factory."""
    from types import SimpleNamespace
    from textual.message import Message

    class ToolCallEvent(Message):
        def __init__(self, name: str, args: str = "") -> None:
            super().__init__()
            self.name = name
            self.args = args

    class ToolResultEvent(Message):
        def __init__(self, name: str, ok: bool) -> None:
            super().__init__()
            self.name = name
            self.ok = ok

    class TokenStreamEvent(Message):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    class IterationProgressEvent(Message):
        def __init__(self, done: int, limit: int) -> None:
            super().__init__()
            self.done = done
            self.limit = limit

    class PhaseEvent(Message):
        def __init__(self, label: str, detail: str = "") -> None:
            super().__init__()
            self.label = label
            self.detail = detail

    class ReasoningTokenEvent(Message):
        def __init__(self, token: str) -> None:
            super().__init__()
            self.token = token

    class ContextSizeEvent(Message):
        def __init__(self, tokens: int) -> None:
            super().__init__()
            self.tokens = tokens

    class JumpToTurn(Message):
        """Posted when a user clicks an entry in Q/A/Sparse views."""

        def __init__(self, ordinal: int) -> None:
            super().__init__()
            self.ordinal = ordinal

    return SimpleNamespace(
        ToolCallEvent=ToolCallEvent,
        ToolResultEvent=ToolResultEvent,
        TokenStreamEvent=TokenStreamEvent,
        IterationProgressEvent=IterationProgressEvent,
        PhaseEvent=PhaseEvent,
        ReasoningTokenEvent=ReasoningTokenEvent,
        ContextSizeEvent=ContextSizeEvent,
        JumpToTurn=JumpToTurn,
    )
