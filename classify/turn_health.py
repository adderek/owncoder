"""Turn health probe: is the model still making progress, or has it broken down?

Deterministic guards see single symptoms (a fake "[tool] x(...)" line, the same
file read N times) and answer each with a note. A weak model ignores the notes,
and the turn burns every nudge, every read, every iteration before stopping —
observed: 10 fabricated-call violations and one file read 28× in one session.

This asks the classifier ONE question about the turn as a whole, from metadata
only (tool names, targets, purposes, result kinds — never file contents):

  progressing    new information or changes toward the request
  circling       near-identical calls repeated without using their results
  format_broken  the model writes tool calls / results as text instead of calling
  needs_user     blocked on something only the user can answer

and maps the verdict to one action: continue, escalate to a stronger model
(auto_tier), or end the turn with a plain explanation. ``classify.turn_health``:
"off" (default) | "advisory" (log + notice only) | "act".
It only ever stops or escalates earlier than the guards would — never extends.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent.classify.client import ClassifierUnavailable, Probe, Verdict, classify

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

TURN_HEALTH = Probe(
    name="turn_health",
    task="the recent progress of an AI coding agent working on one user request",
    labels=(
        ("progressing", "its tool calls gather new information or change files toward "
                        "the request, and it uses what they return"),
        ("circling", "it repeats the same or near-identical calls (same file or range, "
                     "same search) without using their results"),
        ("format_broken", "it writes tool calls or their results as plain text instead "
                          "of real calls, or invents results"),
        ("needs_user", "it is blocked on missing information, an ambiguous request, or "
                       "a question only the user can answer"),
    ),
)

# Verdict must clear this probability before "act" does anything.
_ACT_AT = {"circling": 0.7, "format_broken": 0.6, "needs_user": 0.7}
_MAX_EVENTS = 12
MAX_PROBES_PER_TURN = 2

_STOP_TEXT = {
    "circling": ("Stopped: the model is repeating the same calls without using their "
                 "results ({summary}). Try a narrower instruction, point it at the exact "
                 "lines, or switch to a stronger model (/model)."),
    "format_broken": ("Stopped: the model keeps writing tool calls as text instead of "
                      "calling tools ({summary}) — its results were invented, nothing ran. "
                      "Switch to a stronger model (/model) or /compact the session; more "
                      "nudges will not help this model here."),
    "needs_user": ("Paused: the model appears blocked and needs your input ({summary}). "
                   "Tell it what is missing or clarify the request."),
}


@dataclass
class Decision:
    action: str                 # "continue" | "escalate" | "stop"
    verdict: Verdict | None = None
    text: str = ""              # stop message for the user (action == "stop")


def enabled(config: "Config") -> bool:
    cfg = getattr(config, "classify", None)
    return bool(cfg) and cfg.mode != "off" and cfg.turn_health in ("advisory", "act")


def _real_user(m: dict) -> bool:
    return (m.get("role") == "user" and not m.get("_injected_kind")
            and not m.get("_nudged") and isinstance(m.get("content"), str))


def _target(args: dict) -> str:
    for k in ("path", "argv", "cmd", "command", "url", "pattern", "query", "symbol"):
        if k in args:
            v = args[k]
            v = " ".join(map(str, v)) if isinstance(v, list) else str(v)
            rng = ""
            if "start_line" in args or "end_line" in args:
                rng = f" [{args.get('start_line', '')}-{args.get('end_line', '')}]"
            return (v + rng)[:120]
    return ""


def _result_kind(content: str) -> str:
    c = (content or "").strip()
    if not c:
        return "empty"
    head = c[:400]
    if '"truncated": true' in head:
        return "truncated"
    if "outline only" in head or '"outline_only": true' in c[-200:]:
        return "outline_only"
    if head.startswith("[released"):
        return "released"
    if head.startswith('{"error"') or '"permission_denied": true' in head:
        return "error"
    return "ok"


def _fabricated(text: str) -> bool:
    try:
        from agent.core.streaming import (_has_fake_tool_summary, _has_pseudo_tool_tag,
                                          _has_unexecuted_agent_exec)
    except ImportError:
        return False
    return (_has_fake_tool_summary(text) or _has_pseudo_tool_tag(text)
            or _has_unexecuted_agent_exec(text))


def build_state(messages: list[dict], counters: dict) -> dict:
    """Metadata-only view of the current turn (since the last real user message)."""
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if _real_user(messages[i]):
            start = i
            break
    request = messages[start]["content"][:300] if messages and _real_user(messages[start]) else ""
    events: list[dict] = []
    by_id: dict[str, dict] = {}
    for m in messages[start + 1:]:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {}
                ev = {"call": fn.get("name", "?"), "target": _target(args if isinstance(args, dict) else {}),
                      "purpose": str((args or {}).get("purpose", ""))[:100] if isinstance(args, dict) else ""}
                events.append(ev)
                by_id[tc.get("id", "")] = ev
        elif role == "tool":
            ev = by_id.get(m.get("tool_call_id", ""))
            if ev is not None:
                ev["result"] = _result_kind(str(m.get("content") or ""))
        elif role == "assistant":
            text = str(m.get("content") or "")
            if "NOT executed" in text or "[removed:" in text or _fabricated(text):
                # Never forward the fabricated line itself: the classifier reads
                # state literally and takes "[tool] x(...) → result" for progress.
                events.append({"wrote_tool_call_as_text": True})
            elif text.strip():
                events.append({"said": text.strip()[:100]})
        elif role == "user" and m.get("_injected_kind"):
            events.append({"harness_note": str(m["_injected_kind"])})
    return {"request": request, "recent": events[-_MAX_EVENTS:], "counters": counters}


async def assess(config: "Config", state: dict) -> Verdict | None:
    """Classify turn health. None when disabled or the classifier is unavailable."""
    if not enabled(config):
        return None
    from agent.classify import guard
    down, _ = guard.is_down()
    if down:
        return None
    try:
        v = await classify(config, TURN_HEALTH, state)
    except ClassifierUnavailable as e:
        guard.mark_down(str(e))
        return None
    guard.stats[f"turn:{v.label}"] += 1
    return v


def decide(config: "Config", v: Verdict | None, can_escalate: bool) -> Decision:
    """Verdict → action. Advisory mode never acts; unsure verdicts never act."""
    if v is None or v.label == "progressing":
        return Decision("continue", v)
    summary = f"turn_health={v.label} p={v.p:.2f} conf={v.confidence:.2f}"
    act = config.classify.turn_health == "act" and v.p >= _ACT_AT.get(v.label, 1.1)
    if not act:
        return Decision("continue", v)
    if v.label in ("circling", "format_broken") and can_escalate:
        return Decision("escalate", v)
    return Decision("stop", v, _STOP_TEXT[v.label].format(summary=summary))


def log(config: "Config", state: dict, v: Verdict | None, action: str, trigger: str) -> None:
    """Append to the verdict log and surface a one-line notice for non-progress."""
    from agent.classify import guard
    guard._log(config, "(turn_health)", json.dumps({"trigger": trigger, **state.get("counters", {})}),
               v, action)
    if v is not None and v.label != "progressing":
        guard._notice(f"⚠ classifier: turn_health={v.label} p={v.p:.2f} → {action} ({trigger})")
