"""Plan/Step dataclasses + JSON persistence under .agent/plans/.

A Plan captures the agent's intended work for a single user goal, broken into
atomic Steps. Each Step carries its own tests (descriptions; red-green driven
implementation is enforced by the execution loop, not stored here) and status.

Persistence is one JSON file per plan: `.agent/plans/{plan_id}.json`.
"""
from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


PLAN_STATUSES = ("pending", "active", "completed", "aborted", "stashed", "paused")
STEP_STATUSES = ("pending", "in_progress", "completed", "failed", "skipped")


_plans_dir: Path | None = None


def configure(working_dir: str, agent_dir: str = ".agent") -> None:
    global _plans_dir
    _plans_dir = Path(working_dir) / agent_dir / "plans"


def _get_plans_dir() -> Path:
    return _plans_dir if _plans_dir is not None else Path(".agent") / "plans"


@dataclass
class Step:
    id: str
    description: str
    tests: list[str] = field(default_factory=list)
    status: str = "pending"
    notes: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    snapshot_refs: list[dict] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Plan:
    id: str
    goal: str
    session_id: str = ""
    status: str = "pending"
    steps: list[Step] = field(default_factory=list)
    notes: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def current_step(self) -> Step | None:
        for s in self.steps:
            if s.status == "in_progress":
                return s
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.status in ("completed", "skipped"))
        return done, len(self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        steps_raw = data.get("steps", []) or []
        steps = [Step(**{k: v for k, v in s.items() if k in Step.__dataclass_fields__}) for s in steps_raw]
        return cls(
            id=data["id"],
            goal=data.get("goal", ""),
            session_id=data.get("session_id", ""),
            status=data.get("status", "pending"),
            steps=steps,
            notes=data.get("notes", ""),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


def _new_plan_id() -> str:
    now = datetime.now(timezone.utc)
    ms = now.microsecond // 1000
    return now.strftime("%Y%m%dT%H%M%S.") + f"{ms:03d}Z_{secrets.token_hex(2)}"


def create_plan(goal: str, session_id: str = "", steps: list[dict] | None = None) -> Plan:
    pid = _new_plan_id()
    step_objs: list[Step] = []
    for i, s in enumerate(steps or [], 1):
        if isinstance(s, str):
            step_objs.append(Step(id=f"s{i}", description=s))
        else:
            step_objs.append(Step(
                id=str(s.get("id") or f"s{i}"),
                description=str(s.get("description", "")),
                tests=list(s.get("tests", []) or []),
                status=s.get("status", "pending"),
                notes=s.get("notes", ""),
            ))
    plan = Plan(id=pid, goal=goal, session_id=session_id, steps=step_objs)
    save_plan(plan)
    return plan


def save_plan(plan: Plan) -> Path:
    plan.updated_at = time.time()
    d = _get_plans_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{plan.id}.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_plan(plan_id: str) -> Plan | None:
    path = _get_plans_dir() / f"{plan_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Plan.from_dict(data)
    except Exception:
        return None


def list_plans() -> list[Plan]:
    d = _get_plans_dir()
    if not d.exists():
        return []
    out: list[Plan] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            out.append(Plan.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return out


def delete_plan(plan_id: str) -> bool:
    path = _get_plans_dir() / f"{plan_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def update_step(plan: Plan, step_id: str, **fields) -> Step | None:
    for s in plan.steps:
        if s.id == step_id:
            for k, v in fields.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            if fields.get("status") == "in_progress" and s.started_at is None:
                s.started_at = time.time()
            if fields.get("status") in ("completed", "skipped", "failed") and s.completed_at is None:
                s.completed_at = time.time()
            save_plan(plan)
            return s
    return None
