"""Pre-tool classifier gate: verdict → escalation, never a grant.

Runs after the permission policy and pre-tool hooks allowed a call. The
verdict can only move a call toward stricter handling:

  advisory  a label over its ``ask_at`` threshold adds a note to the tool
            result (the model sees it) and a UI notice; the call still runs.
  enforce   over ``ask_at`` → one-shot approval prompt (no asker = deny);
            over ``deny_at`` → denied.

Confidence (how concentrated the distribution is) below
``review_below_confidence`` counts as an ``ask_at`` hit for any label,
``safe`` included — an unsure "safe" is not a pass.

A user who just approved this exact call interactively is not asked again —
the human already looked at it. Availability: a failed call marks the
classifier down for ``_RETRY_S`` so an outage costs one timeout, not one per
tool call. While down, advisory runs unclassified; enforce asks for approval
until the user runs ``/classify accept`` for the session.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from agent.classify.client import ACTION_RISK, ClassifierUnavailable, Verdict, classify

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_RETRY_S = 30.0
_CACHE_MAX = 256
_USER_APPROVED = ("allowed once", "allowed for session")

_down_until = 0.0
_down_reason = ""
_accepted_down = False          # /classify accept: run without it this session
_cache: "collections.OrderedDict[str, Verdict]" = collections.OrderedDict()
# tool_call_id → compact verdict for the UI (hover badge). Bounded: UIs that
# never pop (terminal) must not grow it.
_by_call: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
stats: collections.Counter = collections.Counter()


def reset() -> None:
    """Forget availability, acceptance, cache and counters (tests, session switch)."""
    global _down_until, _down_reason, _accepted_down
    _down_until, _down_reason, _accepted_down = 0.0, "", False
    _cache.clear()
    _by_call.clear()
    stats.clear()


def pop_call_verdict(call_id: str | None) -> dict | None:
    """Verdict recorded for one tool call (then forgotten), or None if the call
    was not classified."""
    if not call_id:
        return None
    return _by_call.pop(call_id, None)


def _remember(call_id: str | None, verdict: Verdict | None, action: str, error: str) -> None:
    if not call_id:
        return
    rec = {"action": action}
    if verdict is not None:
        rec.update(label=verdict.label, p=round(verdict.p, 3),
                   confidence=round(verdict.confidence, 3),
                   dist={k: round(v, 3) for k, v in verdict.dist.items()},
                   backend=verdict.backend, model=verdict.model, ms=verdict.ms)
    if error:
        rec["error"] = error
    _by_call[call_id] = rec
    if len(_by_call) > _CACHE_MAX:
        _by_call.popitem(last=False)


def mark_down(reason: str) -> None:
    global _down_until, _down_reason
    if not _down_reason:
        logger.info("classifier unavailable: %s", reason)
    _down_until, _down_reason = time.monotonic() + _RETRY_S, reason


def mark_up() -> None:
    global _down_until, _down_reason
    if _down_reason:
        logger.info("classifier back up")
    _down_until, _down_reason = 0.0, ""


def is_down() -> tuple[bool, str]:
    return bool(_down_reason), _down_reason


def accept_down() -> None:
    global _accepted_down
    _accepted_down = True


def down_accepted() -> bool:
    return _accepted_down


def _agent_dir(config) -> Path:
    tools = getattr(config, "tools", None)
    root = Path(getattr(tools, "working_dir", ".") or ".")
    agent_dir = Path(getattr(tools, "agent_dir", ".agent") or ".agent")
    return agent_dir if agent_dir.is_absolute() else root / agent_dir


def _log(config: "Config", tool: str, args_text: str, verdict: Verdict | None,
         action: str, error: str = "", call_id: str | None = None) -> None:
    _remember(call_id, verdict, action, error)
    if not config.classify.log_verdicts:
        return
    rec = {"ts": time.time(), "tool": tool, "args": args_text[:500], "action": action}
    if verdict is not None:
        rec.update(probe=verdict.probe, label=verdict.label, p=round(verdict.p, 4),
                   dist={k: round(v, 4) for k, v in verdict.dist.items()},
                   confidence=round(verdict.confidence, 4), mass=round(verdict.mass, 4),
                   backend=verdict.backend, model=verdict.model, ms=verdict.ms)
    if error:
        rec["error"] = error
    try:
        path = _agent_dir(config) / "classify" / "verdicts.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        logger.debug("classifier: verdict log write failed", exc_info=True)


def _over(thresholds: dict, verdict: Verdict) -> bool:
    try:
        limit = thresholds.get(verdict.label)
        return limit is not None and verdict.p >= float(limit)
    except (TypeError, ValueError):
        return False


def _args_text(config: "Config", args: dict) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(args)
    try:
        from agent.security.redaction import redact
        text = redact(text, config)
    except Exception:
        logger.debug("classifier: redaction unavailable", exc_info=True)
    return text


async def verdict_for(config: "Config", tool: str, args_text: str) -> Verdict:
    """Cached classification of one call. Raises ClassifierUnavailable."""
    if _down_reason and time.monotonic() < _down_until:
        raise ClassifierUnavailable(_down_reason)
    key = hashlib.sha256(f"{tool}\0{args_text}".encode()).hexdigest()
    hit = _cache.get(key)
    if hit is not None:
        _cache.move_to_end(key)
        return hit
    state = {"tool": tool, "args": args_text,
             "cwd": str(getattr(config.tools, "working_dir", "") or "")}
    try:
        v = await classify(config, ACTION_RISK, state)
    except ClassifierUnavailable as e:
        mark_down(str(e))
        raise
    mark_up()
    _cache[key] = v
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)
    return v


async def guard_tool_call(config: "Config", tool: str, args: dict,
                          decision=None, call_id: str | None = None) -> tuple[dict | None, str]:
    """→ (denial result or None, note to append to the tool result or "")."""
    cfg = getattr(config, "classify", None)
    if cfg is None or cfg.mode not in ("advisory", "enforce"):
        return None, ""
    if tool not in (cfg.tools or []) or getattr(config, "runtime_quarantined", False):
        return None, ""
    enforce = cfg.mode == "enforce"
    args_text = _args_text(config, args)
    user_approved = decision is not None and getattr(decision, "reason", "") in _USER_APPROVED

    try:
        v = await verdict_for(config, tool, args_text)
    except ClassifierUnavailable as e:
        stats["unavailable"] += 1
        if not enforce or _accepted_down or user_approved:
            _log(config, tool, args_text, None, "unclassified", str(e), call_id)
            return None, ""
        return await _escalate(config, tool, args, args_text, None,
                               f"action classifier unavailable ({e}); "
                               "approve manually or run /classify accept", call_id)

    stats[v.label] += 1
    summary = f"{v.probe}={v.label} p={v.p:.2f} conf={v.confidence:.2f}"
    uncertain = v.confidence < float(cfg.review_below_confidence or 0.0)
    flagged = v.label != "safe" and (_over(cfg.ask_at, v) or _over(cfg.deny_at, v))
    if not (flagged or uncertain):
        _log(config, tool, args_text, v, "pass", call_id=call_id)
        return None, ""
    if uncertain and not flagged:
        summary += " (low confidence)"
    note = f"[classifier] {summary} — treat this action with care."
    if not enforce or user_approved:
        _log(config, tool, args_text, v, "noted", call_id=call_id)
        _notice(f"⚠ classifier: {tool} flagged {summary}")
        return None, note
    if flagged and _over(cfg.deny_at, v):
        _log(config, tool, args_text, v, "denied", call_id=call_id)
        stats["denied"] += 1
        return _denial(tool, f"action classifier: {summary}"), ""
    return await _escalate(config, tool, args, args_text, v, f"action classifier: {summary}",
                           call_id)


async def _escalate(config, tool, args, args_text, v, reason,
                    call_id=None) -> tuple[dict | None, str]:
    from agent.security import permissions as perms
    question = perms.format_question(tool, args, perms.Decision(perms.ASK, reason), config)
    ok = await perms.confirm_action(question, config)
    _log(config, tool, args_text, v, "approved" if ok else "denied", "" if v else reason, call_id)
    if ok:
        return None, ""
    stats["denied"] += 1
    return _denial(tool, reason), ""


def _denial(tool: str, reason: str) -> dict:
    return {"error": f"Blocked for {tool}: {reason}", "tool": tool,
            "permission_denied": True, "blocked_by_classifier": True}


def _notice(text: str) -> None:
    try:
        from agent import ui_notice
        ui_notice.emit(text)
    except Exception:
        logger.debug("classifier: notice failed", exc_info=True)
