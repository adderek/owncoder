"""On-demand introspection — user-triggered self-diagnosis, heal and improve.

The automatic paths (``memory/reflector.py`` at session end, scheduled ``idle``
introspection jobs) only fire when the session is over or the agent is idle.
This module covers the case the user is actually in: the turn *just* went
sideways — repeated invalid tool calls, a model escalation that failed for the
same reason the cheap model did — and the human wants the agent to stop and
look at itself right now, without leaving the session view.

Shape of the feature: collect the machine-readable evidence the agent cannot
see from inside its own transcript (failure reports on disk, tool-error
clusters, the *schemas* of the tools that keep failing, the model/effort state
that was in play), render it into a prompt, and run that prompt **in the live
session**. It is a prompt rather than a side-channel LLM call on purpose: the
agent already holds the tools to act on what it finds (read/edit the tool
schema, adjust config, record a behavioral rule), and the user stays where they
were, watching the diagnosis stream into the same transcript.

Callers: ``POST /api/heal`` (browser UI), ``/heal`` slash command (terminal and
browser). Collection never raises — a missing signal degrades the report, it
does not block the heal.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Read the tail of index.jsonl only — the file grows across every session.
_TAIL_BYTES = 65536

_MAX_FAILURES = 25       # failure clusters carried into the prompt
_MAX_TOOL_ERRORS = 10    # distinct tools with failing results in-transcript
_MAX_SCHEMAS = 4         # full schemas inlined (they are large)
_SCHEMA_CHARS = 2500     # per-schema cap
_SAMPLE_CHARS = 300


def _agent_dir(config) -> Path:
    try:
        return Path(config.tools.working_dir) / config.tools.agent_dir
    except Exception:
        return Path(".agent")


def _session_start_iso(session_id: str) -> str:
    """Session ids are ``<UTC start>_<suffix>`` — recover the start stamp."""
    head = (session_id or "").split("_")[0]
    return head if len(head) >= 10 else ""


def _read_failure_index(config) -> list[dict]:
    path = _agent_dir(config) / "failures" / "index.jsonl"
    try:
        if not path.exists():
            return []
        with path.open("rb") as fbin:
            if path.stat().st_size > _TAIL_BYTES:
                fbin.seek(-_TAIL_BYTES, 2)
                fbin.readline()  # discard the partial first line
            raw = fbin.read().decode("utf-8", errors="replace").splitlines()
    except Exception:
        logger.debug("self_heal: failure index unreadable", exc_info=True)
        return []
    out = []
    for line in raw:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _failure_clusters(config, session_id: str, limit: int = _MAX_FAILURES) -> list[dict]:
    """Recent failures for this session, grouped by (kind, tool, reason).

    Grouping is the point: ten identical "unknown argument" reports are one
    diagnosis, and the count is the strongest signal in the whole report.
    """
    recs = _read_failure_index(config)
    if not recs:
        return []
    mine = [r for r in recs if r.get("session_id") == session_id] if session_id else []
    if not mine:
        # Same fallback reflector uses: a mislabelled batch would otherwise
        # hide the entire session's material.
        start = _session_start_iso(session_id)
        mine = [r for r in recs if str(r.get("ts") or "") >= start] if start else recs

    counts: Counter = Counter()
    samples: dict[tuple, dict] = {}
    for rec in mine:
        key = (
            str(rec.get("kind") or ""),
            str(rec.get("tool") or ""),
            str(rec.get("reason") or "")[:120],
        )
        counts[key] += 1
        samples.setdefault(key, rec)

    out = []
    for (kind, tool, reason), count in counts.most_common(limit):
        rec = samples[(kind, tool, reason)]
        out.append({
            "kind": kind,
            "tool": tool,
            "reason": reason,
            "count": count,
            "error": str(rec.get("error") or "")[:_SAMPLE_CHARS],
            "file": rec.get("file") or "",
        })
    return out


def _tool_result_error(content: Any) -> str:
    """The error string of a failed tool result, or "" if it succeeded.

    Same envelope rule the turn engine uses: a JSON object carrying "error".
    """
    try:
        parsed = json.loads(content or "")
    except Exception:
        return ""
    if isinstance(parsed, dict) and "error" in parsed:
        return str(parsed.get("error"))[:_SAMPLE_CHARS]
    return ""


def _tool_errors(messages, limit: int = _MAX_TOOL_ERRORS) -> list[dict]:
    """Failing tool results in the live transcript, grouped by tool name.

    The transcript carries what the failure reports do not: the arguments the
    model actually sent. That pairing — schema vs. what the model tried — is
    what turns "the tool keeps failing" into "the schema is ambiguous".
    """
    names: dict[str, str] = {}   # tool_call_id -> tool name
    args: dict[str, str] = {}    # tool_call_id -> raw arguments
    counts: Counter = Counter()
    samples: dict[str, dict] = {}
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")) or ""
                raw = (fn.get("arguments") if isinstance(fn, dict)
                       else getattr(fn, "arguments", "")) or ""
                if tc_id:
                    names[str(tc_id)] = str(name)
                    args[str(tc_id)] = str(raw)[:_SAMPLE_CHARS]
        elif role == "tool":
            error = _tool_result_error(m.get("content"))
            if not error:
                continue
            tc_id = str(m.get("tool_call_id") or "")
            name = names.get(tc_id) or str(m.get("name") or "") or "?"
            counts[name] += 1
            samples.setdefault(name, {"error": error, "args": args.get(tc_id, "")})

    out = []
    for name, count in counts.most_common(limit):
        s = samples[name]
        out.append({"tool": name, "count": count,
                    "error": s["error"], "args": s["args"]})
    return out


def _schemas_for(names: list[str], limit: int = _MAX_SCHEMAS) -> list[dict]:
    """Current JSON schemas of the suspect tools.

    The motivating case for this whole feature: the root cause is the tool
    contract, not the model. The agent can only see that if the schema is put
    in front of it.
    """
    if not names:
        return []
    try:
        from agent.tools import get_schemas
        schemas = get_schemas()
    except Exception:
        logger.debug("self_heal: tool schemas unavailable", exc_info=True)
        return []
    by_name = {s.get("function", {}).get("name"): s for s in schemas}
    out = []
    for name in names[:limit]:
        schema = by_name.get(name)
        if schema is None:
            continue
        try:
            text = json.dumps(schema.get("function", schema), indent=2, default=str)
        except Exception:
            continue
        out.append({"tool": name, "schema": text[:_SCHEMA_CHARS]})
    return out


def _model_state(config) -> dict:
    def _get(path: str, default: str = "") -> str:
        obj: Any = config
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        return str(obj)

    return {
        "model": _get("llm.model"),
        "effort": _get("models.effort"),
        "mode": _get("models.mode"),
        "max_iterations": _get("llm.max_iterations"),
    }


def collect_signals(config, session_id: str = "", messages=None) -> dict:
    """Evidence for one heal run. Never raises."""
    signals: dict = {"session_id": session_id, "failures": [], "tool_errors": [],
                     "schemas": [], "model": {}}
    try:
        signals["failures"] = _failure_clusters(config, session_id)
    except Exception:
        logger.debug("self_heal: failure collection failed", exc_info=True)
    try:
        signals["tool_errors"] = _tool_errors(messages)
    except Exception:
        logger.debug("self_heal: transcript scan failed", exc_info=True)
    # Suspects: tools that failed in-transcript first (we have their arguments),
    # then tools named by failure reports.
    suspects: list[str] = []
    for entry in signals["tool_errors"]:
        if entry["tool"] and entry["tool"] != "?" and entry["tool"] not in suspects:
            suspects.append(entry["tool"])
    for entry in signals["failures"]:
        if entry["tool"] and entry["tool"] not in suspects:
            suspects.append(entry["tool"])
    signals["suspects"] = suspects
    signals["schemas"] = _schemas_for(suspects)
    try:
        signals["model"] = _model_state(config)
    except Exception:
        logger.debug("self_heal: model state unavailable", exc_info=True)
    signals["counts"] = {
        "failures": sum(f["count"] for f in signals["failures"]),
        "failure_kinds": len(signals["failures"]),
        "tool_errors": sum(t["count"] for t in signals["tool_errors"]),
        "tools": len(suspects),
    }
    return signals


def summary_line(signals: dict) -> str:
    """One line for the UI: what the heal button would be working from."""
    c = signals.get("counts") or {}
    parts = []
    if c.get("failures"):
        parts.append(f"{c['failures']} failure report(s) in {c['failure_kinds']} cluster(s)")
    if c.get("tool_errors"):
        parts.append(f"{c['tool_errors']} failed tool result(s)")
    if c.get("tools"):
        parts.append(f"{c['tools']} suspect tool(s): " +
                     ", ".join((signals.get("suspects") or [])[:5]))
    return "; ".join(parts) if parts else "no failure signals recorded this session"


def format_evidence(signals: dict) -> str:
    """The evidence block, as the model reads it."""
    lines: list[str] = []
    model = signals.get("model") or {}
    if any(model.values()):
        lines.append("[STATE] " + ", ".join(
            f"{k}={v}" for k, v in model.items() if v))

    failures = signals.get("failures") or []
    if failures:
        lines.append("\n[FAILURE REPORTS] (.agent/failures/, grouped)")
        for f in failures:
            head = f"- x{f['count']} [{f['kind']}]"
            if f["tool"]:
                head += f" tool={f['tool']}"
            if f["reason"]:
                head += f" reason={f['reason']}"
            lines.append(head)
            if f["error"]:
                lines.append(f"    error: {f['error']}")
            if f["file"]:
                lines.append(f"    detail: .agent/failures/{f['file']}")

    tool_errors = signals.get("tool_errors") or []
    if tool_errors:
        lines.append("\n[FAILED TOOL RESULTS THIS SESSION]")
        for t in tool_errors:
            lines.append(f"- x{t['count']} {t['tool']}: {t['error']}")
            if t["args"]:
                lines.append(f"    last arguments sent: {t['args']}")

    schemas = signals.get("schemas") or []
    if schemas:
        lines.append("\n[CURRENT SCHEMAS OF THE SUSPECT TOOLS]")
        for s in schemas:
            lines.append(f"--- {s['tool']} ---\n{s['schema']}")

    if not lines:
        return "[NO RECORDED SIGNALS] Nothing was captured on disk — work from " \
               "the conversation above."
    return "\n".join(lines)


_INSTRUCTIONS = """\
SELF-DIAGNOSIS REQUEST (triggered by the user, mid-session).

Something in this session is going wrong and the user wants you to stop and
introspect instead of pushing on. Do not resume the previous task until this is
answered.

Work through it in this order:

1. ROOT CAUSE. Read the evidence below against the conversation above. Name the
   single most likely root cause, not the symptom. Consider explicitly whether
   the cause is *yours* (wrong approach, wrong assumption, ignoring a result) or
   *structural*: an ambiguous or wrong tool schema, a tool that reports success
   on failure, a missing capability, a bad config value, a model too weak or a
   context too full for the task. Escalating the model does not fix a broken
   tool contract — say so if that is what happened.
2. EVIDENCE. Quote the specific failures or messages that support the cause.
   If the evidence does not support any single cause, say what you would need
   to observe next to tell the candidates apart.
3. FIX. Apply what you can, now, with your tools — correct a tool schema or its
   description, fix a wrong argument habit, change the approach, adjust a
   setting. Verify each change if a cheap check exists. Do not touch anything
   unrelated to the diagnosed cause.
4. DURABLE RULE. If a mistake would repeat in a future session, record the rule
   that prevents it (project memory / notes), phrased as an action: "When X, do
   Y" or "Never Z".
5. HANDOFF. Finish with a short report: cause, what you changed, what you could
   not fix and why, and whether the interrupted task can now be resumed safely.

Keep it proportionate — this is a diagnosis, not a refactor.
"""


def build_prompt(signals: dict, focus: str = "") -> str:
    """The heal prompt submitted to the live session.

    *focus* is the user's own hint from the UI ("tool calls keep failing"),
    optional and quoted verbatim so it steers without overriding the protocol.
    """
    parts = [_INSTRUCTIONS]
    if (focus or "").strip():
        parts.append(f"USER'S OBSERVATION: {focus.strip()}")
    parts.append("EVIDENCE\n========\n" + format_evidence(signals))
    return "\n\n".join(parts)


def heal_request(config, session_id: str = "", messages=None,
                 focus: str = "") -> tuple[str, dict]:
    """Convenience: (prompt, signals) for one on-demand heal."""
    signals = collect_signals(config, session_id, messages)
    return build_prompt(signals, focus), signals
