"""Turn-signal tools — typed replacements for the >>>KIND: text markers.

The model calls one of these to control the harness loop instead of emitting a
`>>>ASK:` / `>>>DONE:` line in prose. `run_turn` intercepts a signal-tool call,
ends the turn, and surfaces a canonical signal line that the meta-loop parses
(`agent.core.turn_signals.parse_signal`). The regex parser stays as a fallback
for models that still emit the text form, so this is additive — both work.

Each handler returns a `signal_line` ("`>>>KIND: payload`") plus a short ack.
`run_turn` reads `signal_line` from the serialized result; nothing here writes
the signal into conversation history.
"""
from __future__ import annotations

import json

from agent.tools import register

# canonical signal kind -> >>> token (mirrors agent.core.turn_signals)
_KIND_TOKEN: dict[str, str] = {
    "next_step": "NEXT",
    "ask_user": "ASK",
    "request_feedback": "FEEDBACK",
    "request_review": "REVIEW",
    "done": "DONE",
    "consult_crows": "CROWS",
    "blocked": "BLOCKED",
}

# tool name -> canonical signal kind
SIGNAL_TOOL_KINDS: dict[str, str] = {
    "next_step": "next_step",
    "ask_user": "ask_user",
    "request_feedback": "request_feedback",
    "request_review": "request_review",
    "mark_done": "done",
    "consult_crows": "consult_crows",
    "blocked": "blocked",
}
SIGNAL_TOOL_NAMES = frozenset(SIGNAL_TOOL_KINDS)


def build_signal_line(kind: str, payload: str) -> str:
    """Canonical one-line signal: `>>>TOKEN: payload`. Empty/unknown → ''."""
    token = _KIND_TOKEN.get(kind)
    if token is None:
        return ""
    return f">>>{token}: {(payload or '').strip()}"


def _emit(kind: str, payload: str) -> dict:
    return {"signal_line": build_signal_line(kind, payload), "ack": f"{kind} signalled"}


def extract_signal_line(tool_calls, results) -> str | None:
    """First signal-tool result's canonical line, or None.

    `tool_calls` and `results` align by index (results are serialized strings).
    """
    for tc, result in zip(tool_calls, results):
        if tc.function.name not in SIGNAL_TOOL_NAMES:
            continue
        try:
            parsed = json.loads(result)
            line = parsed.get("signal_line") if isinstance(parsed, dict) else None
        except Exception:
            line = None
        if not line:
            # Result unreadable (truncated/redacted): rebuild from the call args.
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            payload = (args.get("question") or args.get("summary") or args.get("reason")
                       or args.get("instruction") or args.get("topic") or args.get("scope") or "")
            line = build_signal_line(SIGNAL_TOOL_KINDS[tc.function.name], str(payload))
        if line:
            return line
    return None


@register("next_step", {
    "description": "Continue autonomously to the next step of the task. Use when you "
                   "have more work to do and do not need the user. Ends this turn; the "
                   "harness immediately starts the next step with your instruction.",
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {"type": "string", "description": "What to do next."},
        },
        "required": ["instruction"],
    },
})
def next_step(instruction: str) -> dict:
    return _emit("next_step", instruction)


@register("ask_user", {
    "description": "Ask the user a question and pause. Use when you need a decision or "
                   "information only the user can give. Ends this turn.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question for the user."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "Optional suggested answers."},
        },
        "required": ["question"],
    },
})
def ask_user(question: str, options: list | None = None) -> dict:
    payload = question
    if options:
        payload = f"{question}  [{' | '.join(str(o) for o in options)}]"
    return _emit("ask_user", payload)


@register("request_feedback", {
    "description": "Request the user's feedback on a topic. Ends this turn.",
    "parameters": {
        "type": "object",
        "properties": {"topic": {"type": "string", "description": "What to get feedback on."}},
        "required": ["topic"],
    },
})
def request_feedback(topic: str) -> dict:
    return _emit("request_feedback", topic)


@register("request_review", {
    "description": "Hand off to a stronger model / reviewer for the given scope. Ends this turn.",
    "parameters": {
        "type": "object",
        "properties": {"scope": {"type": "string", "description": "What to review."}},
        "required": ["scope"],
    },
})
def request_review(scope: str) -> dict:
    return _emit("request_review", scope)


@register("mark_done", {
    "description": "Mark the task complete and end the turn. Use when nothing remains.",
    "parameters": {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "What was accomplished."}},
        "required": ["summary"],
    },
})
def mark_done(summary: str) -> dict:
    return _emit("done", summary)


@register("consult_crows", {
    "description": "Consult the crowd (many small models) on a problem. Ends this turn.",
    "parameters": {
        "type": "object",
        "properties": {"topic": {"type": "string", "description": "Problem to consult on."}},
        "required": ["topic"],
    },
})
def consult_crows(topic: str) -> dict:
    return _emit("consult_crows", topic)


@register("blocked", {
    "description": "Report that you are blocked and need manual intervention. Ends this turn.",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why you are blocked."},
            "unblock": {"type": "string", "description": "What would unblock you."},
        },
        "required": ["reason"],
    },
})
def blocked(reason: str, unblock: str = "") -> dict:
    payload = f"{reason} | {unblock}" if unblock else reason
    return _emit("blocked", payload)
