"""Housekeeping for the agent state directory (``.agent/``).

Level 1 only: actions that lose nothing a user would miss.

- Rotated logs are gzipped. New rotations are compressed by the log handler
  itself (``gz_namer``/``gz_rotator``); this module converts the uncompressed
  ``agent.log.N`` files left by older versions, or by an older process still
  running next to this one.
- ``index.db.bak-*`` copies made before a re-embed are trimmed to the newest
  ``keep_index_backups``. Nothing else ever removed them, and each is a full
  copy of the index.

Session data, audit logs and diagnostics are never touched here.

One implementation, two front ends: ``agent gc`` (explicit, dry-run unless
``--apply``) and ``maybe_start_background`` (at chat start, at most once per
``interval_hours``, under a lock so two agents sharing a worktree do not race).
"""
from __future__ import annotations

import fcntl
import gzip
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LOG_NAME = "agent.log"
LOCK_NAME = "housekeeping.lock"
_TMP_SUFFIX = ".tmp-gc"
_LEGACY_LOG = re.compile(r"^agent\.log\.(\d+)$")
_GZ_LOG = re.compile(r"^agent\.log\.(\d+)\.gz$")
_SIDECARS = ("-wal", "-shm")


# -- log rotation hooks (used by cli/logging_setup.py) -----------------------

def gz_namer(name: str) -> str:
    return name + ".gz"


def gz_rotator(source: str, dest: str) -> None:
    with open(source, "rb") as src, gzip.open(dest, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    os.remove(source)


# -- plan --------------------------------------------------------------------

@dataclass
class Action:
    kind: str                      # "compress" | "delete"
    path: Path
    size: int
    reason: str
    target: Path | None = None     # compress only
    freed: int = 0                 # filled in by apply()
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "target": str(self.target) if self.target else None,
            "size": self.size,
            "freed": self.freed,
            "reason": self.reason,
            "error": self.error,
        }


@dataclass
class Report:
    applied: bool
    actions: list[Action] = field(default_factory=list)

    @property
    def freed(self) -> int:
        return sum(a.freed for a in self.actions)

    @property
    def errors(self) -> list[Action]:
        return [a for a in self.actions if a.error]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _plan_logs(agent_dir: Path, backup_count: int) -> list[Action]:
    legacy: list[tuple[int, Path]] = []
    gz: list[tuple[int, Path]] = []
    for p in agent_dir.glob(LOG_NAME + ".*"):
        if m := _LEGACY_LOG.match(p.name):
            legacy.append((int(m.group(1)), p))
        elif m := _GZ_LOG.match(p.name):
            gz.append((int(m.group(1)), p))
    actions: list[Action] = []
    # Uncompressed rotations are older than every .gz one (the handler only
    # writes .gz now), so they queue up behind the highest .gz index, keeping
    # their relative order. Whatever lands past backup_count is what the
    # handler would have rotated out already.
    next_idx = max((i for i, _ in gz), default=0) + 1
    for _, p in sorted(legacy):
        if next_idx > backup_count:
            actions.append(Action("delete", p, _size(p),
                                  f"rotated log beyond logs.backup_count={backup_count}"))
        else:
            target = agent_dir / f"{LOG_NAME}.{next_idx}.gz"
            actions.append(Action("compress", p, _size(p),
                                  "uncompressed rotated log", target=target))
            next_idx += 1
    for i, p in sorted(gz):
        if i > backup_count:
            actions.append(Action("delete", p, _size(p),
                                  f"rotated log beyond logs.backup_count={backup_count}"))
    return actions


def _plan_index_backups(agent_dir: Path, keep: int) -> list[Action]:
    if keep < 0:
        return []
    live = agent_dir / "index.db"
    # No usable live index: the backups may be the only copy left.
    if _size(live) == 0:
        return []
    backups = sorted(
        (p for p in agent_dir.glob("index.db.bak-*")
         if not p.name.endswith(_SIDECARS) and p.is_file()),
        key=lambda p: p.name,   # timestamp suffix sorts chronologically
    )
    doomed = backups[:-keep] if keep else backups
    actions: list[Action] = []
    for p in doomed:
        reason = f"index backup older than the newest {keep}"
        actions.append(Action("delete", p, _size(p), reason))
        for suffix in _SIDECARS:
            side = Path(str(p) + suffix)
            if side.exists():
                actions.append(Action("delete", side, _size(side), reason))
    return actions


def _plan_leftovers(agent_dir: Path) -> list[Action]:
    return [Action("delete", p, _size(p), "interrupted housekeeping run")
            for p in agent_dir.glob("*" + _TMP_SUFFIX)]


def plan(agent_dir: Path, *, backup_count: int = 5,
         keep_index_backups: int = 1) -> list[Action]:
    agent_dir = Path(agent_dir)
    if not agent_dir.is_dir():
        return []
    return (_plan_leftovers(agent_dir)
            + _plan_logs(agent_dir, backup_count)
            + _plan_index_backups(agent_dir, keep_index_backups))


# -- apply -------------------------------------------------------------------

def _compress(action: Action) -> None:
    src, target = action.path, action.target
    assert target is not None
    tmp = Path(str(target) + _TMP_SUFFIX)
    with open(src, "rb") as fh:
        st = os.fstat(fh.fileno())
        try:
            with open(tmp, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6,
                                   mtime=int(st.st_mtime)) as dst:
                    shutil.copyfileobj(fh, dst)
                raw.flush()
                os.fsync(raw.fileno())
            # link, not replace: a rotation that created the target meanwhile
            # must win, never be overwritten.
            os.link(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    os.utime(target, (st.st_atime, st.st_mtime))
    # An older process may have rotated a different file onto this name while
    # we were reading; only remove the file we actually compressed.
    try:
        if os.stat(src).st_ino == st.st_ino:
            os.unlink(src)
    except FileNotFoundError:
        pass
    action.freed = st.st_size - _size(target)


def apply(actions: list[Action]) -> None:
    for a in actions:
        try:
            if a.kind == "compress":
                _compress(a)
            else:
                a.path.unlink(missing_ok=True)
                a.freed = a.size
        except Exception as exc:  # one bad file must not stop the rest
            a.error = f"{type(exc).__name__}: {exc}"
            logger.warning("housekeeping: %s %s failed: %s", a.kind, a.path, a.error)


# -- front ends --------------------------------------------------------------

def _settings(config) -> dict:
    hk = getattr(config, "housekeeping", None)
    return {
        "backup_count": int(getattr(getattr(config, "logs", None), "backup_count", 5)),
        "keep_index_backups": int(getattr(hk, "keep_index_backups", 1)),
    }


def run(agent_dir: Path, config=None, *, do_apply: bool = False) -> Report | None:
    """Plan and (optionally) apply. Returns None when another run holds the lock."""
    agent_dir = Path(agent_dir)
    actions_kw = _settings(config)
    if not do_apply:
        return Report(False, plan(agent_dir, **actions_kw))
    if not agent_dir.is_dir():
        return Report(True, [])
    lock_path = agent_dir / LOCK_NAME
    with open(lock_path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return None
        actions = plan(agent_dir, **actions_kw)
        apply(actions)
        os.utime(lock_path)   # mtime of the lock file = time of the last run
    return Report(True, actions)


def due(agent_dir: Path, interval_hours: float) -> bool:
    try:
        last = (Path(agent_dir) / LOCK_NAME).stat().st_mtime
    except OSError:
        return True
    return time.time() - last >= interval_hours * 3600


def maybe_start_background(agent_dir: Path, config) -> threading.Thread | None:
    """Start one housekeeping pass in a daemon thread if one is due.

    Silent on screen: the outcome goes to the log.
    """
    hk = getattr(config, "housekeeping", None)
    if not getattr(hk, "enabled", True):
        return None
    try:
        from agent.security import vault
        if not vault.persist_allowed() or vault.encrypting():
            return None
    except Exception:
        return None
    if not due(agent_dir, float(getattr(hk, "interval_hours", 24))):
        return None

    def _work() -> None:
        try:
            report = run(agent_dir, config, do_apply=True)
            if report is None:
                logger.debug("housekeeping: another run holds the lock")
            elif report.actions:
                logger.info("housekeeping: %d action(s), %.1f MB freed, %d error(s)",
                            len(report.actions), report.freed / 1e6, len(report.errors))
        except Exception:
            logger.warning("housekeeping failed", exc_info=True)

    thread = threading.Thread(target=_work, name="housekeeping", daemon=True)
    thread.start()
    return thread
