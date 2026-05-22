"""Turn signals — structured control tokens the model emits at end of response.

Signal syntax (one line, end of response):
  >>>NEXT: do the next thing
  >>>ASK: question for user
  >>>FEEDBACK: topic to get feedback on
  >>>REVIEW: scope to pass to a stronger model
  >>>DONE: completion summary
  >>>CROWS: problem for crowd consultation (many small models)
  >>>BLOCKED: reason | what would unblock

Signals are parsed by the meta-loop in LocalUIServer.chat() and stripped from
the assistant message before it enters conversation history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SIGNAL_LINE_RE = re.compile(
    r"^>>>(NEXT|ASK|FEEDBACK|REVIEW|DONE|CROWS|BLOCKED):\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_KIND_NORMALIZE: dict[str, str] = {
    "next": "next_step",
    "ask": "ask_user",
    "feedback": "request_feedback",
    "review": "request_review",
    "done": "done",
    "crows": "consult_crows",
    "blocked": "blocked",
}


@dataclass
class TurnSignal:
    kind: str   # next_step | ask_user | request_feedback | request_review | done | consult_crows | blocked
    payload: str


def parse_signal(response: str) -> tuple[str, TurnSignal | None]:
    """Return (clean_response, signal|None).

    Picks the last matching signal line. Strips all signal lines from clean_response.
    """
    matches = list(_SIGNAL_LINE_RE.finditer(response))
    if not matches:
        return response, None

    last = matches[-1]
    raw_kind = last.group(1).lower()
    kind = _KIND_NORMALIZE.get(raw_kind, raw_kind)
    payload = last.group(2).strip()

    clean = _SIGNAL_LINE_RE.sub("", response).strip()
    return clean, TurnSignal(kind=kind, payload=payload)
