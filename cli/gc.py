"""agent gc — housekeeping of the agent state directory. Dry-run unless --apply."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def cmd_gc(args, config) -> int:
    from agent import housekeeping

    agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
    do_apply = bool(getattr(args, "apply", False))
    report = housekeeping.run(agent_dir, config, do_apply=do_apply)
    if report is None:
        print("another housekeeping run holds the lock", file=sys.stderr)
        return 4

    if getattr(args, "json", False):
        print(json.dumps({
            "applied": report.applied,
            "freed": report.freed,
            "actions": [a.as_dict() for a in report.actions],
        }, indent=2))
        return 1 if report.errors else 0

    for a in report.actions:
        try:
            shown = a.path.relative_to(agent_dir)
        except ValueError:
            shown = a.path
        line = f"{a.kind:<8} {_mb(a.size):>10}  {shown}"
        if a.target is not None:
            line += f" -> {a.target.name}"
        line += f"  ({a.reason})"
        if a.error:
            line += f"  FAILED: {a.error}"
        print(line)
    if not report.actions:
        print("nothing to do", file=sys.stderr)
    elif report.applied:
        print(f"freed {_mb(report.freed)}", file=sys.stderr)
    else:
        total = sum(a.size for a in report.actions)
        print(f"dry run: {len(report.actions)} action(s) on {_mb(total)}; "
              "re-run with --apply", file=sys.stderr)
    return 1 if report.errors else 0
