"""agent diag — tool health report from audit.jsonl."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_audit(agent_dir: Path) -> list[dict]:
    path = agent_dir / "audit.jsonl"
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if "tool" in d:
                    entries.append(d)
            except json.JSONDecodeError:
                pass
    return entries


def _load_failures(agent_dir: Path) -> list[dict]:
    idx = agent_dir / "failures" / "index.jsonl"
    if not idx.exists():
        return []
    out = []
    with idx.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _retry_stats(entries: list[dict]) -> dict[str, dict[str, int]]:
    """Detect consecutive same-tool error→any calls per session as retries."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        sid = e.get("session_id", "__no_session__")
        by_session[sid].append(e)

    retries: dict[str, dict[str, int]] = defaultdict(lambda: {"retried": 0, "total_errors": 0})
    for sid, seq in by_session.items():
        for i, e in enumerate(seq):
            tool = e.get("tool")
            outcome = e.get("outcome", "ok" if "error" not in e else "error")
            if outcome == "error" and tool:
                retries[tool]["total_errors"] += 1
                # Check if next call in session is same tool (retry pattern).
                if i + 1 < len(seq) and seq[i + 1].get("tool") == tool:
                    retries[tool]["retried"] += 1
    return dict(retries)


def cmd_diag(args: Any, config: Any) -> None:
    from pathlib import Path as _Path
    agent_dir = _Path(config.tools.working_dir) / config.tools.agent_dir

    entries = _load_audit(agent_dir)
    failures = _load_failures(agent_dir)

    if not entries and not failures:
        print("No audit data found.")
        return

    # Per-tool stats from audit.
    tool_stats: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "error_msgs": Counter()})
    for e in entries:
        tool = e.get("tool", "?")
        tool_stats[tool]["calls"] += 1
        outcome = e.get("outcome", "ok" if "error" not in e else "error")
        if outcome == "error":
            tool_stats[tool]["errors"] += 1
            msg = str(e.get("error", ""))[:100]
            tool_stats[tool]["error_msgs"][msg] += 1

    retry = _retry_stats(entries)

    # Pre-execution failures (invalid_tool_call) from failures/index.jsonl.
    pre_exec: dict[str, int] = Counter(
        f["tool"] for f in failures if f.get("kind") == "invalid_tool_call" and f.get("tool")
    )

    print(f"\n{'Tool':<22} {'Calls':>6} {'Errors':>7} {'Err%':>6} {'Retried':>8} {'Pre-exec':>9}")
    print("-" * 62)

    sorted_tools = sorted(tool_stats.items(), key=lambda x: -x[1]["errors"])
    for tool, s in sorted_tools:
        calls = s["calls"]
        errors = s["errors"]
        rate = errors / calls * 100 if calls else 0
        ret = retry.get(tool, {}).get("retried", 0)
        pre = pre_exec.get(tool, 0)
        print(f"{tool:<22} {calls:>6} {errors:>7} {rate:>5.0f}% {ret:>8} {pre:>9}")

    print()

    # Top error messages per failing tool.
    for tool, s in sorted_tools:
        if not s["error_msgs"]:
            continue
        print(f"{tool} — top errors:")
        for msg, cnt in s["error_msgs"].most_common(3):
            print(f"  [{cnt:>3}] {msg}")
        # Show error_detail for atomic_rollback from entries.
        if tool == "edit_file":
            detail_kinds: Counter = Counter()
            for e in entries:
                if e.get("tool") == "edit_file" and e.get("error") == "atomic_rollback":
                    for d in e.get("error_detail", []):
                        if isinstance(d, dict):
                            detail_kinds[d.get("kind", "?")] += 1
            if detail_kinds:
                print(f"  rollback inner kinds: {dict(detail_kinds.most_common(5))}")
        print()

    # Invalid tool calls (pre-execution).
    if pre_exec:
        print("Pre-execution failures (invalid_tool_call):")
        for tool, cnt in sorted(pre_exec.items(), key=lambda x: -x[1]):
            reasons = Counter(
                f.get("reason", "?")
                for f in failures
                if f.get("tool") == tool and f.get("kind") == "invalid_tool_call"
            )
            print(f"  {tool}: {cnt}  reasons={dict(reasons)}")
        print()
