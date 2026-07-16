"""Session performance summary — reads the per-call side-logs written during a
turn (``llm_calls.jsonl`` + ``tool_calls.jsonl``) and renders where wall-time
went: LLM generation vs tool execution, plus the slowest tools.

These side-logs are the persistent counterpart to the in-memory spinner stats:
they survive the session so a slow run can be diagnosed after the fact.
"""
from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return rows


def summarize(session_dir: str | Path) -> dict:
    """Aggregate timing from a session's side-logs into a stats dict."""
    d = Path(session_dir)
    llm = _read_jsonl(d / "llm_calls.jsonl")
    tools = _read_jsonl(d / "tool_calls.jsonl")

    llm_secs = sum((r.get("gen_seconds") or 0.0) for r in llm)
    out_tok = sum((r.get("output_tokens") or 0) for r in llm)
    in_tok = sum((r.get("input_tokens") or 0) for r in llm)
    ttfts = [r["ttft"] for r in llm if r.get("ttft")]

    tool_secs = sum((r.get("duration_ms") or 0.0) for r in tools) / 1000.0

    per_tool: dict[str, dict] = {}
    for r in tools:
        name = r.get("tool", "?")
        rec = per_tool.setdefault(name, {"calls": 0, "ms": 0.0, "errors": 0})
        rec["calls"] += 1
        rec["ms"] += r.get("duration_ms") or 0.0
        if not r.get("ok", True):
            rec["errors"] += 1

    return {
        "llm_calls": len(llm),
        "llm_seconds": round(llm_secs, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "avg_ttft": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
        "out_tps": round(out_tok / llm_secs, 1) if llm_secs > 0 else None,
        "tool_calls": len(tools),
        "tool_seconds": round(tool_secs, 1),
        "per_tool": per_tool,
    }


# Data-source classes for retrieval-cost analysis: which way of finding
# information the agent actually leans on (and what it costs in wall-time).
_SOURCE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grep/text",   ("grep", "ripgrep", "search_text")),
    ("index/RAG",   ("index", "semantic", "rag", "search_code", "graph")),
    ("memory",      ("recall", "notes", "kb")),
    ("web",         ("web_search", "web_fetch", "ask_internet")),
    ("filesystem",  ("read_file", "list_dir", "explore", "files", "file_stats")),
)


def source_class(tool_name: str) -> str | None:
    """Map a tool name to a data-source class, or None for non-retrieval tools."""
    n = (tool_name or "").lower()
    for cls, keys in _SOURCE_KEYWORDS:
        if any(k in n for k in keys):
            return cls
    return None


def summarize_all(sessions_base: str | Path) -> dict:
    """Aggregate tool side-logs across ALL stored sessions.

    Returns per-tool and per-source-class totals; used to spot which data
    sources dominate (candidates for lazy runs / cost saving) and which are
    dead weight.
    """
    base = Path(sessions_base)
    per_tool: dict[str, dict] = {}
    per_source: dict[str, dict] = {}
    n_sessions = 0
    if not base.exists():
        return {"sessions": 0, "per_tool": per_tool, "per_source": per_source}
    for f in base.rglob("tool_calls.jsonl"):
        rows = _read_jsonl(f)
        if not rows:
            continue
        n_sessions += 1
        for r in rows:
            name = r.get("tool", "?")
            rec = per_tool.setdefault(name, {"calls": 0, "ms": 0.0, "errors": 0})
            rec["calls"] += 1
            rec["ms"] += r.get("duration_ms") or 0.0
            if not r.get("ok", True):
                rec["errors"] += 1
            cls = source_class(name)
            if cls:
                src = per_source.setdefault(cls, {"calls": 0, "ms": 0.0, "errors": 0})
                src["calls"] += 1
                src["ms"] += r.get("duration_ms") or 0.0
                if not r.get("ok", True):
                    src["errors"] += 1
    return {"sessions": n_sessions, "per_tool": per_tool, "per_source": per_source}


def run_perf_all_command() -> str:
    """Render a cross-session data-source usage report (``/perf all``)."""
    from agent.memory.session import _get_session_dir
    s = summarize_all(_get_session_dir())
    if not s["per_tool"]:
        return "perf all: no tool side-logs found across sessions."
    lines = [f"Data-source usage across {s['sessions']} session(s):"]
    if s["per_source"]:
        total_calls = sum(r["calls"] for r in s["per_source"].values())
        lines.append("  retrieval sources (share of retrieval calls):")
        for cls, rec in sorted(s["per_source"].items(), key=lambda kv: kv[1]["calls"], reverse=True):
            pct = 100.0 * rec["calls"] / total_calls if total_calls else 0.0
            err = f"  {rec['errors']} err" if rec["errors"] else ""
            lines.append(f"    {cls:<12} {pct:5.1f}%  x{rec['calls']:<6} {rec['ms']:>9.0f}ms{err}")
        lines.append("  (a source with high share + high ms is a lazy-run candidate;")
        lines.append("   a source with ~0% share may be wasted setup cost)")
    ranked = sorted(s["per_tool"].items(), key=lambda kv: kv[1]["calls"], reverse=True)
    lines.append("  top tools (by calls):")
    for name, rec in ranked[:12]:
        err = f"  {rec['errors']} err" if rec["errors"] else ""
        lines.append(f"    {name:<22} x{rec['calls']:<6} {rec['ms']:>9.0f}ms{err}")
    return "\n".join(lines)


def run_perf_command(session_dir: str | Path | None) -> str:
    """Render a plain-text performance summary for the current session."""
    if not session_dir:
        return "perf: no active session side-log."
    s = summarize(session_dir)
    if not s["llm_calls"] and not s["tool_calls"]:
        return "perf: no metrics recorded yet this session."

    wall = s["llm_seconds"] + s["tool_seconds"]
    lines = ["Session performance:"]
    lines.append(
        f"  LLM:   {s['llm_seconds']:>7.1f}s  over {s['llm_calls']} calls"
        + (f"  ({s['out_tps']} tok/s out)" if s["out_tps"] else "")
    )
    if s["avg_ttft"] is not None:
        lines.append(f"  TTFT:  {s['avg_ttft']:>7.3f}s avg (prefill latency)")
    lines.append(f"  tools: {s['tool_seconds']:>7.1f}s  over {s['tool_calls']} calls")
    if wall > 0:
        llm_pct = 100.0 * s["llm_seconds"] / wall
        lines.append(f"  split: LLM {llm_pct:.0f}% / tools {100 - llm_pct:.0f}%  (instrumented wall {wall:.1f}s)")
    lines.append(f"  tokens: {s['input_tokens']} in / {s['output_tokens']} out")

    if s["per_tool"]:
        ranked = sorted(s["per_tool"].items(), key=lambda kv: kv[1]["ms"], reverse=True)
        lines.append("  slowest tools (total ms):")
        for name, rec in ranked[:8]:
            err = f"  {rec['errors']} err" if rec["errors"] else ""
            lines.append(f"    {name:<22} {rec['ms']:>9.0f}ms  x{rec['calls']}{err}")
    return "\n".join(lines)
