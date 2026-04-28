"""DAG utilities for dependency-aware plan step ordering.

Steps declare deps as a list of step IDs they must wait on.
No external dependencies — pure Python.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.planning.plan import Step


def ready_steps(steps: list["Step"]) -> list["Step"]:
    """Pending steps whose every dep is completed or skipped."""
    done = {s.id for s in steps if s.status in ("completed", "skipped")}
    return [
        s for s in steps
        if s.status == "pending"
        and all(dep in done for dep in s.deps)
    ]


def blocked_steps(steps: list["Step"]) -> list["Step"]:
    """Pending steps blocked by at least one unresolved dep."""
    done = {s.id for s in steps if s.status in ("completed", "skipped")}
    return [
        s for s in steps
        if s.status == "pending"
        and s.deps
        and any(dep not in done for dep in s.deps)
    ]


def detect_cycles(steps: list["Step"]) -> list[str]:
    """Return sorted step IDs involved in cycles. Empty list = acyclic."""
    ids = {s.id for s in steps}
    adj: dict[str, list[str]] = {
        s.id: [d for d in s.deps if d in ids] for s in steps
    }

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {sid: WHITE for sid in ids}
    cycle_nodes: set[str] = set()

    def _dfs(node: str) -> bool:
        color[node] = GRAY
        for nb in adj.get(node, []):
            if color[nb] == GRAY:
                cycle_nodes.update((node, nb))
                return True
            if color[nb] == WHITE and _dfs(nb):
                cycle_nodes.add(node)
                return True
        color[node] = BLACK
        return False

    for sid in ids:
        if color[sid] == WHITE:
            _dfs(sid)

    return sorted(cycle_nodes)


def critical_path(steps: list["Step"]) -> list[str]:
    """Step IDs on the longest dep chain (by node count), root → leaf."""
    if not steps:
        return []
    ids = {s.id for s in steps}
    dep_map: dict[str, list[str]] = {
        s.id: [d for d in s.deps if d in ids] for s in steps
    }
    memo: dict[str, list[str]] = {}

    def _longest(node: str) -> list[str]:
        if node in memo:
            return memo[node]
        deps = dep_map.get(node, [])
        if not deps:
            memo[node] = [node]
            return [node]
        best = max((_longest(d) for d in deps), key=len)
        result = best + [node]
        memo[node] = result
        return result

    return max((_longest(s.id) for s in steps), key=len)
