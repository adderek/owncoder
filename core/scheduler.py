"""Scheduled jobs — cron-like and delayed execution for the agent.

Jobs are prompts the agent runs unattended: a daily press summary, a periodic
indicator check, a one-shot "in 20m" follow-up, or introspection that should
happen "when nothing else is going on". Each run executes in a *fresh* Agent
with its own session (``sched-<name>``) so scheduled work never pollutes the
user's conversation.

Three trigger paths share one job store:
- the in-process ticker thread (long-running ``agent chat``),
- the ``agent cron run`` CLI (external crontab / systemd-timer),
- the idle-task sweep (kind ``idle`` jobs only).

The store is ``<agent_dir>/schedule/jobs.json`` guarded by ``fcntl.flock`` —
claiming a due job (advancing ``next_run`` under the lock) is what prevents a
ticker and an external cron from double-running the same job.

Schedule spec grammar (parse_spec):
- ``at 2026-07-03T07:00``  or  ``at 2026-07-03 07:00``   one-shot, local time
- ``in 20m`` / ``in 2h`` / ``in 90s`` / ``in 1d``         one-shot delay
- ``every 30m`` / ``every 6h`` / ``every 1d``             recurring interval
- ``0 7 * * *``                                           5-field cron, local time
- ``@hourly`` / ``@daily`` / ``@midnight`` / ``@weekly``  cron shorthands
- ``idle``                                                one-shot at next idle sweep
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.config import Config
    from agent.core.agent import Agent

logger = logging.getLogger(__name__)

_CRON_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 7 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 7 * * 1",
}

_DURATION_RE = re.compile(r"^(\d+)\s*(s|m|h|d)$")
_DUR_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# How far ahead _cron_next searches before giving up (guards bad specs like
# "0 0 31 2 *" that never match).
_CRON_HORIZON_DAYS = 366 * 4


@dataclass
class Job:
    id: str = ""
    name: str = ""
    prompt: str = ""
    spec: str = ""            # original schedule text, reparsed to advance cron
    kind: str = "at"          # "cron" | "every" | "at" | "idle"
    interval: float = 0.0     # seconds, kind == "every"
    next_run: float = 0.0     # epoch seconds; 0 for kind == "idle"
    one_shot: bool = False    # disable after first run
    enabled: bool = True
    created_at: float = 0.0
    last_run: float = 0.0     # epoch seconds of last claim
    last_status: str = ""     # "" | "ok" | "error: …"
    last_session: str = ""    # session id of the last run
    last_result: str = ""     # first chars of the last reply


def _parse_duration(text: str) -> float | None:
    m = _DURATION_RE.match(text.strip().lower())
    if not m:
        return None
    return int(m.group(1)) * _DUR_SECONDS[m.group(2)]


def _parse_at(text: str) -> float | None:
    """Parse a local datetime; date-only means midnight, time-only means today
    (or tomorrow when the time already passed)."""
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text, fmt).time()
        except ValueError:
            continue
        candidate = datetime.now().replace(
            hour=t.hour, minute=t.minute, second=t.second, microsecond=0
        )
        if candidate.timestamp() <= time.time():
            candidate += timedelta(days=1)
        return candidate.timestamp()
    return None


def _parse_cron_field(field_text: str, lo: int, hi: int) -> set[int] | None:
    """One cron field → set of matching ints, or None on syntax error."""
    values: set[int] = set()
    for part in field_text.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, _, step_s = part.partition("/")
            if not step_s.isdigit() or int(step_s) < 1:
                return None
            step = int(step_s)
        if part in ("*", ""):
            lo2, hi2 = lo, hi
        elif "-" in part:
            a, _, b = part.partition("-")
            if not (a.isdigit() and b.isdigit()):
                return None
            lo2, hi2 = int(a), int(b)
        elif part.isdigit():
            lo2 = hi2 = int(part)
        else:
            return None
        if lo2 < lo or hi2 > hi or lo2 > hi2:
            return None
        values.update(range(lo2, hi2 + 1, step))
    return values


def _parse_cron(spec: str) -> list[set[int]] | None:
    """5-field cron → [minutes, hours, days-of-month, months, days-of-week]."""
    fields = spec.split()
    if len(fields) != 5:
        return None
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)]
    parsed = []
    for text, (lo, hi) in zip(fields, bounds):
        vals = _parse_cron_field(text, lo, hi)
        if vals is None:
            return None
        parsed.append(vals)
    # cron allows both 0 and 7 for Sunday
    if 7 in parsed[4]:
        parsed[4].add(0)
    return parsed


def _cron_next(fields: list[set[int]], after: float) -> float:
    """Next epoch time strictly after *after* matching the cron fields (local)."""
    minutes, hours, doms, months, dows = fields
    t = datetime.fromtimestamp(after).replace(second=0, microsecond=0) + timedelta(minutes=1)
    end = t + timedelta(days=_CRON_HORIZON_DAYS)
    while t < end:
        if t.month not in months:
            # jump to the 1st of the next month
            t = (t.replace(day=1) + timedelta(days=32)).replace(
                day=1, hour=0, minute=0
            )
            continue
        # standard cron: day matches if dom OR dow matches (when both restricted)
        dom_restricted = doms != set(range(1, 32))
        dow_restricted = dows != set(range(0, 8))
        cron_dow = (t.weekday() + 1) % 7  # Monday=0 → cron Sunday=0
        if dom_restricted and dow_restricted:
            day_ok = t.day in doms or cron_dow in dows
        else:
            day_ok = t.day in doms and cron_dow in dows
        if not day_ok:
            t = (t + timedelta(days=1)).replace(hour=0, minute=0)
            continue
        if t.hour not in hours:
            t = (t + timedelta(hours=1)).replace(minute=0)
            continue
        if t.minute not in minutes:
            t += timedelta(minutes=1)
            continue
        return t.timestamp()
    raise ValueError("cron spec never matches within the search horizon")


def parse_spec(spec: str, now: float | None = None) -> tuple[str, float, float, bool]:
    """Parse a schedule spec → (kind, next_run, interval, one_shot).

    Raises ValueError with a user-readable message on bad specs.
    """
    if now is None:
        now = time.time()
    text = spec.strip()
    low = text.lower()

    if low == "idle":
        return "idle", 0.0, 0.0, True

    if low in _CRON_ALIASES:
        text = _CRON_ALIASES[low]
        low = text

    if low.startswith("in "):
        secs = _parse_duration(text[3:])
        if secs is None:
            raise ValueError(f"bad delay {text[3:]!r} — use e.g. 'in 20m', 'in 2h'")
        return "at", now + secs, 0.0, True

    if low.startswith("at "):
        ts = _parse_at(text[3:])
        if ts is None:
            raise ValueError(
                f"bad time {text[3:]!r} — use 'at 2026-07-03T07:00' or 'at 07:00'"
            )
        return "at", ts, 0.0, True

    if low.startswith("every "):
        secs = _parse_duration(text[6:])
        if secs is None:
            raise ValueError(f"bad interval {text[6:]!r} — use e.g. 'every 30m'")
        return "every", now + secs, float(secs), False

    fields = _parse_cron(text)
    if fields is not None:
        return "cron", _cron_next(fields, now), 0.0, False

    raise ValueError(
        f"cannot parse schedule {spec!r} — use 'in 20m', 'at 07:00', "
        "'every 6h', a 5-field cron ('0 7 * * *'), '@daily', or 'idle'"
    )


# ── Store ────────────────────────────────────────────────────────────────────


def _schedule_dir(config: "Config") -> Path:
    agent_dir = Path(config.tools.agent_dir)
    if not agent_dir.is_absolute():
        agent_dir = Path(config.tools.working_dir) / agent_dir
    d = agent_dir / "schedule"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _locked_jobs:
    """Context manager: flock the store, yield the job list, save on change."""

    def __init__(self, config: "Config"):
        self._dir = _schedule_dir(config)
        self._path = self._dir / "jobs.json"
        self._lock_fh = None
        self.jobs: list[Job] = []
        self.dirty = False

    def __enter__(self) -> "_locked_jobs":
        self._lock_fh = open(self._dir / ".lock", "w")
        fcntl.flock(self._lock_fh, fcntl.LOCK_EX)
        self.jobs = _read_jobs(self._path)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.dirty and exc_type is None:
                tmp = self._path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(
                    {"jobs": [asdict(j) for j in self.jobs]}, indent=2
                ), encoding="utf-8")
                tmp.replace(self._path)
        finally:
            fcntl.flock(self._lock_fh, fcntl.LOCK_UN)
            self._lock_fh.close()
        return False


def _read_jobs(path: Path) -> list[Job]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("scheduler: unreadable %s — treating as empty", path)
        return []
    jobs = []
    known = {f.name for f in Job.__dataclass_fields__.values()}
    for item in data.get("jobs", []):
        if isinstance(item, dict):
            jobs.append(Job(**{k: v for k, v in item.items() if k in known}))
    return jobs


def _find(jobs: list[Job], id_or_name: str) -> Job | None:
    for j in jobs:
        if j.id == id_or_name or (j.name and j.name == id_or_name):
            return j
    return None


def add_job(config: "Config", spec: str, prompt: str, name: str = "") -> Job:
    """Validate the spec and persist a new job. Raises ValueError on bad input."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("empty prompt — the job needs something to do")
    kind, next_run, interval, one_shot = parse_spec(spec)
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-")[:40]
    job = Job(
        id=secrets.token_hex(3),
        name=name,
        prompt=prompt,
        spec=spec.strip(),
        kind=kind,
        interval=interval,
        next_run=next_run,
        one_shot=one_shot,
        created_at=time.time(),
    )
    with _locked_jobs(config) as store:
        if name and _find(store.jobs, name) is not None:
            raise ValueError(f"a job named {name!r} already exists")
        store.jobs.append(job)
        store.dirty = True
    return job


def remove_job(config: "Config", id_or_name: str) -> bool:
    with _locked_jobs(config) as store:
        job = _find(store.jobs, id_or_name)
        if job is None:
            return False
        store.jobs.remove(job)
        store.dirty = True
        return True


def set_enabled(config: "Config", id_or_name: str, enabled: bool) -> bool:
    with _locked_jobs(config) as store:
        job = _find(store.jobs, id_or_name)
        if job is None:
            return False
        job.enabled = enabled
        if enabled and job.kind in ("cron", "every"):
            # recompute so a long-disabled job doesn't fire immediately
            _, job.next_run, _, _ = parse_spec(job.spec)
        store.dirty = True
        return True


def list_jobs(config: "Config") -> list[Job]:
    return _read_jobs(_schedule_dir(config) / "jobs.json")


def has_due_jobs(config: "Config", kinds: tuple[str, ...], now: float | None = None) -> bool:
    """Cheap lock-free peek used by the ticker before spinning up a claim."""
    now = time.time() if now is None else now
    return any(_is_due(j, kinds, now) for j in list_jobs(config))


def _is_due(job: Job, kinds: tuple[str, ...], now: float) -> bool:
    if not job.enabled or job.kind not in kinds:
        return False
    if job.kind == "idle":
        return True
    return 0 < job.next_run <= now


def claim_due(config: "Config", kinds: tuple[str, ...], now: float | None = None) -> list[Job]:
    """Atomically claim due jobs: advance/disable them in the store and return
    copies for execution. Two concurrent claimers cannot get the same job."""
    now = time.time() if now is None else now
    claimed: list[Job] = []
    with _locked_jobs(config) as store:
        for job in store.jobs:
            if not _is_due(job, kinds, now):
                continue
            job.last_run = now
            job.last_status = "running"
            if job.kind == "every":
                job.next_run = now + job.interval
            elif job.kind == "cron":
                try:
                    fields = _parse_cron(_CRON_ALIASES.get(job.spec.lower(), job.spec))
                    job.next_run = _cron_next(fields, now)
                except Exception:
                    job.enabled = False
                    job.last_status = "error: cron spec no longer parses"
                    continue
            else:  # "at" / "idle" — one-shot
                job.enabled = False
            store.dirty = True
            claimed.append(Job(**asdict(job)))
    return claimed


def record_result(config: "Config", job_id: str, status: str,
                  session_id: str = "", result: str = "") -> None:
    with _locked_jobs(config) as store:
        job = _find(store.jobs, job_id)
        if job is None:
            return
        job.last_status = status
        job.last_session = session_id
        job.last_result = result[:400]
        store.dirty = True
    try:
        entry = {
            "ts": time.time(),
            "job": job_id,
            "name": job.name,
            "status": status,
            "session": session_id,
            "result": result[:400],
        }
        with open(_schedule_dir(config) / "runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        logger.debug("scheduler: runs log append failed", exc_info=True)


def unseen_results(config: "Config", limit: int = 10) -> list[dict]:
    """Run records appended since the last call (delivery cursor in runs.seen).

    Backs result delivery into the interactive session: the agent injects
    these as context on the next turn instead of letting them rot in
    sched-<name> sessions. First call initializes the cursor to 'now' so
    historical runs are not dumped wholesale."""
    path = _schedule_dir(config) / "runs.jsonl"
    cur = _schedule_dir(config) / "runs.seen"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    if not cur.exists():
        try:
            cur.write_text(str(len(lines)), encoding="utf-8")
        except Exception:
            pass
        return []
    try:
        seen = int(cur.read_text(encoding="utf-8").strip() or 0)
    except Exception:
        seen = 0
    if seen > len(lines):   # runs.jsonl truncated/rotated
        seen = 0
    new = lines[seen:]
    if not new:
        return []
    try:
        cur.write_text(str(len(lines)), encoding="utf-8")
    except Exception:
        pass
    out: list[dict] = []
    for line in new:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[-limit:]


def recent_runs(config: "Config", limit: int = 20) -> list[dict]:
    path = _schedule_dir(config) / "runs.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


# ── Execution ────────────────────────────────────────────────────────────────


async def execute_job(config: "Config", job: Job) -> str:
    """Run one job in a fresh Agent with its own session. Returns the reply."""
    from agent.core.agent import Agent
    from agent.data_provider import LocalDataProvider
    from agent.memory.session import new_session, save_session

    from agent.core import background
    bg_id = background.register_external(
        f"sched:{job.name or job.id} ({job.spec})", "scheduler")
    try:
        agent = Agent(config, data_provider=LocalDataProvider(config=config))
        session = new_session(
            short_name=f"sched-{job.name or job.id}",
            description=f"scheduled run of {job.spec!r}",
            tags=["scheduled"],
        )
        agent.session = session
        session_id = session.id
        try:
            reply = await agent.chat(job.prompt, source="scheduler")
            status = "ok"
        except Exception as exc:
            logger.warning("scheduler: job %s (%s) failed: %s", job.id, job.name, exc)
            reply = ""
            status = f"error: {exc}"[:200]
        try:
            # Give post-turn background work (QA summary etc.) a moment, then save.
            await agent.wait_background(timeout=30)
            save_session(session, agent.messages)
        except Exception:
            logger.debug("scheduler: session save failed", exc_info=True)
    finally:
        background.unregister(bg_id)
    record_result(config, job.id, status, session_id=session_id, result=reply)
    return reply


async def run_due_jobs_async(config: "Config",
                             kinds: tuple[str, ...] = ("cron", "every", "at"),
                             on_progress: Callable[[str], None] | None = None) -> int:
    """Claim and execute all due jobs sequentially. Returns count executed."""
    jobs = claim_due(config, kinds)
    for job in jobs:
        label = job.name or job.id
        if on_progress:
            on_progress(f"scheduler: running job '{label}' ({job.spec})")
        await execute_job(config, job)
        if on_progress:
            on_progress(f"scheduler: job '{label}' done")
    return len(jobs)


def run_due_jobs(config: "Config",
                 kinds: tuple[str, ...] = ("cron", "every", "at"),
                 on_progress: Callable[[str], None] | None = None) -> int:
    """Sync wrapper — safe to call from a plain thread (owns its event loop)."""
    return asyncio.run(run_due_jobs_async(config, kinds, on_progress))


# ── In-process ticker ────────────────────────────────────────────────────────


def start_ticker(config: "Config", agent: "Agent | None" = None) -> threading.Event:
    """Start the scheduler ticker in a daemon thread; returns its stop event.

    The ticker defers to the interactive agent: it skips a beat while a turn is
    running (``agent._turn_busy``) or too soon after one (``min_quiet_seconds``),
    so scheduled work never competes with the user for the local model.
    """
    stop = threading.Event()
    tick = max(5.0, float(getattr(config.scheduler, "tick_seconds", 60)))
    quiet = float(getattr(config.scheduler, "min_quiet_seconds", 30))

    def _loop() -> None:
        while not stop.wait(tick):
            try:
                if not has_due_jobs(config, ("cron", "every", "at")):
                    continue
                if agent is not None:
                    if getattr(agent, "_turn_busy", False):
                        continue
                    last = getattr(agent, "_last_turn_time", 0.0)
                    if last and (time.monotonic() - last) < quiet:
                        continue
                run_due_jobs(config)
            except Exception:
                logger.warning("scheduler: ticker sweep failed", exc_info=True)

    threading.Thread(target=_loop, daemon=True, name="scheduler-ticker").start()
    return stop


# ── Idle-sweep hook (kind == "idle" jobs) ────────────────────────────────────


async def _action_scheduled_idle(agent: "Agent") -> bool:
    config = getattr(agent, "config", None)
    if config is None or not getattr(config.scheduler, "enabled", True):
        return False
    if not has_due_jobs(config, ("idle",)):
        return False
    return await run_due_jobs_async(config, kinds=("idle",)) > 0


def register_idle_hook() -> None:
    from agent.core.idle_tasks import register_idle_action
    register_idle_action("scheduled-idle", _action_scheduled_idle)


# ── Shared slash-command handler ─────────────────────────────────────────────


def _fmt_ts(ts: float) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def run_schedule_command(config: "Config", arg: str) -> str:
    """Text handler for /schedule, shared by both UIs and the cron CLI.

    Subcommands: (list) | add <spec> :: <prompt> [:: <name>] | rm <id|name> |
    on/off <id|name> | runs | run
    """
    parts = arg.strip().split(None, 1)
    sub = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list", "ls"):
        jobs = list_jobs(config)
        if not jobs:
            return ("No scheduled jobs.\n"
                    "Add one: /schedule add <spec> :: <prompt> [:: <name>]\n"
                    "Specs: 'in 20m', 'at 07:00', 'every 6h', '0 7 * * *', '@daily', 'idle'")
        lines = [f"Scheduled jobs ({len(jobs)}):"]
        for j in jobs:
            state = "on " if j.enabled else "OFF"
            when = "next idle" if j.kind == "idle" else _fmt_ts(j.next_run)
            lines.append(
                f"  {j.id} {state} {j.name or '-':<16} {j.spec:<16} next: {when}"
                + (f"  last: {j.last_status} {_fmt_ts(j.last_run)}" if j.last_run else "")
            )
            lines.append(f"      {j.prompt[:100]}")
        return "\n".join(lines)

    if sub == "add":
        pieces = [p.strip() for p in rest.split("::")]
        if len(pieces) < 2 or not pieces[0] or not pieces[1]:
            return ("Usage: /schedule add <spec> :: <prompt> [:: <name>]\n"
                    "e.g.   /schedule add @daily :: gather a press summary :: press")
        name = pieces[2] if len(pieces) > 2 else ""
        try:
            job = add_job(config, pieces[0], pieces[1], name=name)
        except ValueError as exc:
            return f"Cannot add job: {exc}"
        when = "next idle sweep" if job.kind == "idle" else _fmt_ts(job.next_run)
        return f"Scheduled job {job.id} ({job.name or job.spec}) — next run: {when}"

    if sub in ("rm", "delete", "del", "cancel"):
        if not rest:
            return "Usage: /schedule rm <id|name>"
        return (f"Removed job {rest}." if remove_job(config, rest)
                else f"No job '{rest}'.")

    if sub in ("on", "enable"):
        return (f"Job {rest} enabled." if rest and set_enabled(config, rest, True)
                else f"No job '{rest}'." if rest else "Usage: /schedule on <id|name>")

    if sub in ("off", "disable"):
        return (f"Job {rest} disabled." if rest and set_enabled(config, rest, False)
                else f"No job '{rest}'." if rest else "Usage: /schedule off <id|name>")

    if sub == "runs":
        runs = recent_runs(config)
        if not runs:
            return "No scheduled runs yet."
        lines = ["Recent scheduled runs:"]
        for r in runs:
            lines.append(
                f"  {_fmt_ts(r.get('ts', 0))} {r.get('name') or r.get('job')}: "
                f"{r.get('status')}  session={r.get('session', '')[:24]}"
            )
            if r.get("result"):
                lines.append(f"      {r['result'][:120]}")
        return "\n".join(lines)

    if sub == "run":
        kinds = ("cron", "every", "at", "idle")
        if not has_due_jobs(config, kinds):
            return "No jobs due."

        def _bg() -> None:
            try:
                run_due_jobs(config, kinds)
            except Exception:
                logger.warning("scheduler: /schedule run failed", exc_info=True)

        threading.Thread(target=_bg, daemon=True, name="schedule-run").start()
        return "Running due job(s) in the background — check /schedule runs."

    return ("Unknown subcommand. Use: list | add <spec> :: <prompt> [:: <name>] | "
            "rm <id|name> | on/off <id|name> | runs | run")
