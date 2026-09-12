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

# Match a signal line tolerantly: optional indent, optional space after >>>,
# the keyword, an OPTIONAL colon, and an OPTIONAL payload. Bare markers like
# `>>>DONE` (no colon/summary) are common and must still parse+strip — otherwise
# the marker leaks into history, display, and TTS. `\b` stops `>>>DONEISH`.
_SIGNAL_LINE_RE = re.compile(
    r"^[ \t]*>>>[ \t]*(NEXT|ASK|FEEDBACK|REVIEW|DONE|CROWS|BLOCKED)\b[ \t]*:?[ \t]*(.*)$",
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


def _closing_signal_line(text: str):
    """Match a signal ONLY as the message's closing line.

    Scanning the whole body would delete a signal the author merely quoted — a
    relayed reply, or prose discussing this format. That exactly matched the
    reported symptom of a `>>>DONE` line vanishing from the middle of a quoted
    passage. The documented contract is "one line, end of response", so the
    match is limited to the last non-empty line (trailing blank lines skipped).

    Returns (match, start, end) or None.
    """
    pos = len(text)
    for line in reversed(text.splitlines(keepends=True)):
        start = pos - len(line)
        if line.strip():
            m = _SIGNAL_LINE_RE.match(text[start:pos])
            if m is None:
                return None
            return m, start, pos
        pos = start
    return None


def strip_signals(text: str) -> str:
    """Remove the closing signal line from text (defensive use outside parse_signal)."""
    found = _closing_signal_line(text)
    if found is None:
        return text.strip()
    _, start, end = found
    return (text[:start] + text[end:]).strip()


def parse_signal(response: str) -> tuple[str, TurnSignal | None]:
    """Return (clean_response, signal|None).

    A signal counts only as the closing line; anything else is content and is
    left untouched. The closing line alone is stripped from clean_response.
    """
    found = _closing_signal_line(response)
    if found is None:
        return response, None

    m, start, end = found
    raw_kind = m.group(1).lower()
    kind = _KIND_NORMALIZE.get(raw_kind, raw_kind)
    payload = m.group(2).strip()

    clean = (response[:start] + response[end:]).strip()
    return clean, TurnSignal(kind=kind, payload=payload)
