"""Wire messages for the notification channel.

JSON envelope (one object per message) shared by all channel types:
  {"type": "notice",   "id": "n-1", "kind": "done", "text": "...", "session": "..."}
  {"type": "question", "id": "q-1", "kind": "ask_user", "text": "...",
   "options": ["accept", "refuse"], "free_text": true, "expires_at": 1760000000.0}
  {"type": "answer",   "id": "q-1", "choice": "accept", "text": null, "from": "user"}
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field

# Notify-wire protocol version. Sent in the relay hello so server and client can
# detect a breaking mismatch. Bump the major when notice/question/answer shapes
# change incompatibly; additive (new optional fields) stays the same version.
NOTIFY_PROTOCOL_VERSION = 1

# Per-process nonce: ids must be globally unique so that when several agents
# share one relay, an answer broadcast to all agents only matches the pending
# question of the agent that actually asked it. A bare per-process counter
# (the old scheme) collided across processes; the nonce disambiguates them.
_PROC_NONCE = uuid.uuid4().hex[:12]
_id_counter = itertools.count(1)


def _next_id(prefix: str) -> str:
    return f"{prefix}-{_PROC_NONCE}-{next(_id_counter)}"


@dataclass
class Notice:
    """Fire-and-forget display message (progress, done, error)."""
    kind: str            # signal kind or "info"
    text: str
    session: str = ""
    id: str = field(default_factory=lambda: _next_id("n"))

    def to_wire(self) -> dict:
        return {
            "type": "notice",
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "session": self.session,
        }

    def render_text(self) -> str:
        return f"[{self.kind}] {self.text}"


@dataclass
class Question:
    """Message awaiting an answer from a channel."""
    kind: str
    text: str
    options: list[str] = field(default_factory=list)
    free_text: bool = True
    default: str = ""    # option used by broker on timeout with on_timeout="continue"
    session: str = ""
    expires_at: float = 0.0   # set by broker from answer_timeout_s
    id: str = field(default_factory=lambda: _next_id("q"))

    def to_wire(self) -> dict:
        return {
            "type": "question",
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "options": self.options,
            "free_text": self.free_text,
            "session": self.session,
            "expires_at": self.expires_at,
        }

    def render_text(self) -> str:
        opts = f"  [{' | '.join(self.options)}]" if self.options else ""
        return f"[{self.kind}] {self.text}{opts}"


@dataclass
class Answer:
    """Response to a Question. Validated by broker against the pending question."""
    question_id: str
    choice: str = ""     # one of Question.options
    text: str = ""       # free-text reply (chat-capability channels)
    source: str = "user"  # "user" | "agent:<id>"

    def to_wire(self) -> dict:
        return {
            "type": "answer",
            "id": self.question_id,
            "choice": self.choice,
            "text": self.text,
            "from": self.source,
        }
