"""Planning and crash-recovery subsystem.

Public API:
  - Plan, Step: dataclasses
  - create_plan, load_plan, save_plan, list_plans
  - configure: set storage dir
  - recovery.scan_pending, recovery.record_crash, recovery.resolve
"""
from __future__ import annotations

from agent.planning.plan import (
    Plan,
    Step,
    PLAN_STATUSES,
    STEP_STATUSES,
    configure as configure_plans,
    create_plan,
    load_plan,
    save_plan,
    list_plans,
    delete_plan,
)
from agent.planning import recovery
from agent.planning.increment import (
    RepoSnapshot,
    find_git_repos,
    snapshot_step,
    revert_to_snapshots,
    squash_snapshot,
)
from agent.planning.dag import (
    ready_steps as dag_ready_steps,
    blocked_steps as dag_blocked_steps,
    detect_cycles,
    critical_path as dag_critical_path,
)
from agent.planning.compact import compact_plan, compact_plan_sync

__all__ = [
    "Plan",
    "Step",
    "PLAN_STATUSES",
    "STEP_STATUSES",
    "configure_plans",
    "create_plan",
    "load_plan",
    "save_plan",
    "list_plans",
    "delete_plan",
    "recovery",
    "RepoSnapshot",
    "find_git_repos",
    "snapshot_step",
    "revert_to_snapshots",
    "squash_snapshot",
    "dag_ready_steps",
    "dag_blocked_steps",
    "detect_cycles",
    "dag_critical_path",
    "compact_plan",
    "compact_plan_sync",
]
