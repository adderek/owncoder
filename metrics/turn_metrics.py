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
