"""Crash recovery: persist a record when agent crashes, offer to resume later.

A record lives at `.agent/recovery/{session_id}.json` with status:
  - pending  — not yet acted on
  - recovered — user chose to resume
  - ignored  — user chose to skip
  - resolved — session closed cleanly after recovery

`scan_pending()` returns all pending records sorted newest first. The CLI
prompts per record at startup, subject to `[recovery] prompt_mode`:
  - ask          → interactive choice (default)
  - auto_recover → all pending treated as recover
  - auto_skip    → all pending ignored
"""
from __future__ import annotations

import json
import time
import traceback as _tb
from dataclasses import dataclass, field, asdict
from pathlib import Path


RECOVERY_STATUSES = ("pending", "recovered", "ignored", "resolved")

_recovery_dir: Path | None = None


def configure(working_dir: str, agent_dir: str = ".agent") -> None:
    global _recovery_dir
    _recovery_dir = Path(working_dir) / agent_dir / "recovery"


def _get_dir() -> Path:
    return _recovery_dir if _recovery_dir is not None else Path(".agent") / "recovery"


@dataclass
class CrashRecord:
    session_id: str
    crashed_at: float = field(default_factory=time.time)
    plan_id: str | None = None
    last_user_message: str = ""
    traceback: str = ""
    exception: str = ""
    status: str = "pending"
    recovered_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CrashRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _path_for(session_id: str) -> Path:
    safe = session_id.replace("/", "_")
    return _get_dir() / f"{safe}.json"


def record_crash(
    session_id: str,
    exc: BaseException,
    *,
    plan_id: str | None = None,
    last_user_message: str = "",
) -> Path:
    rec = CrashRecord(
        session_id=session_id,
        plan_id=plan_id,
        last_user_message=last_user_message,
        traceback="".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
        exception=f"{type(exc).__name__}: {exc}",
    )
    d = _get_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = _path_for(session_id)
    p.write_text(json.dumps(rec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def scan_pending() -> list[CrashRecord]:
    d = _get_dir()
    if not d.exists():
        return []
    out: list[CrashRecord] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rec = CrashRecord.from_dict(data)
            if rec.status == "pending":
                out.append(rec)
        except Exception:
            pass
    return out


def set_status(session_id: str, status: str) -> None:
    if status not in RECOVERY_STATUSES:
        raise ValueError(f"invalid status: {status}")
    p = _path_for(session_id)
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    data["status"] = status
    if status == "recovered":
        data["recovered_at"] = time.time()
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def delete(session_id: str) -> bool:
    p = _path_for(session_id)
    if p.exists():
        p.unlink()
        return True
    return False


def prompt_user_choice(rec: CrashRecord) -> str:
    """Interactive prompt for one pending record. Returns 'recover'|'ignore'|'delete'."""
    print(f"\n[recovery] Session {rec.session_id} crashed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rec.crashed_at))}")
    if rec.exception:
        print(f"  exception: {rec.exception}")
    if rec.last_user_message:
        snippet = rec.last_user_message[:200].replace("\n", " ")
        print(f"  last user msg: {snippet}")
    if rec.plan_id:
        print(f"  plan_id: {rec.plan_id}")
    while True:
        choice = input("  [r]ecover / [i]gnore / [d]elete / [s]kip? ").strip().lower()
        if choice in ("r", "recover"):
            return "recover"
        if choice in ("i", "ignore"):
            return "ignore"
        if choice in ("d", "delete"):
            return "delete"
        if choice in ("s", "skip", ""):
            return "ignore"


def handle_pending_at_startup(prompt_mode: str = "ask") -> list[CrashRecord]:
    """Apply `prompt_mode` to pending records. Returns records user chose to recover.

    prompt_mode ∈ {ask, auto_recover, auto_skip}.
    """
    pending = scan_pending()
    to_recover: list[CrashRecord] = []
    if not pending:
        return to_recover
    if prompt_mode == "auto_recover":
        for rec in pending:
            set_status(rec.session_id, "recovered")
            to_recover.append(rec)
        return to_recover
    if prompt_mode == "auto_skip":
        for rec in pending:
            set_status(rec.session_id, "ignored")
        return to_recover
    # ask mode
    for rec in pending:
        try:
            choice = prompt_user_choice(rec)
        except (EOFError, KeyboardInterrupt):
            choice = "ignore"
        if choice == "recover":
            set_status(rec.session_id, "recovered")
            to_recover.append(rec)
        elif choice == "delete":
            delete(rec.session_id)
        else:
            set_status(rec.session_id, "ignored")
    return to_recover
