"""Wake the interactive agent's own conversation later.

``sleep_until`` arms a wake on the main agent; the model then ends its turn.
The first of these starts the next turn in the *same* conversation:

- a deadline (``in 10m``, ``at 07:00``),
- an event: background shell job finished (``job:<id>``), or a watch target
  (``file:`` / ``url:`` / ``cmd:`` / ``pid:``, same semantics as /watch),
- the user's next message — it supersedes every wake armed before it.

Contrast ``schedule_task``: that runs a fresh agent in a detached session.

In-process only: wakes die with the process. Delivery goes through the UI's
external-prompt path (``LocalUIServer.submit_external_prompt``), so it starts a
turn in UIs that register a handler (Textual, HTTP); elsewhere it falls back to
the inject queue and waits for the next input.
"""
from __future__ import annotations

import contextvars
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.core.agent import Agent
    from agent.core.scheduler import Job

logger = logging.getLogger(__name__)

#: A turn whose input starts with this was started by a wake, not the user.
WAKE_PREFIX = "[wake "

_POLL_SECONDS = 1.0
_FAST_WATCHES = ("file", "pid")  # cheap to poll; cmd/url use watch_tick_seconds

#: The agent whose turn is running in this task (set by Agent.chat). Tools are
#: plain functions with no agent handle; async tools see this context.
current_agent: contextvars.ContextVar["Agent | None"] = contextvars.ContextVar(
    "wake_current_agent", default=None)


@dataclass
class Wake:
    id: str
    note: str
    armed_at: float
    deadline: float = 0.0          # epoch seconds; 0 = no deadline
    job_id: int | None = None      # background shell job
    watch: "Job | None" = None     # scheduler probe Job; watch_state = baseline
    next_watch_poll: float = 0.0
    fired: str = ""                # reason, once fired; delivered when idle

    def describe(self) -> str:
        parts = []
        if self.deadline:
            parts.append("at " + datetime.fromtimestamp(self.deadline).strftime("%H:%M:%S"))
        if self.job_id is not None:
            parts.append(f"job:{self.job_id} done")
        if self.watch is not None:
            parts.append(f"{self.watch.watch_type}:{self.watch.watch_target}")
        return " or ".join(parts) or "user message"


@dataclass
class _State:
    lock: threading.Lock = field(default_factory=threading.Lock)
    wakes: list[Wake] = field(default_factory=list)
    dispatch: Callable[[str], None] | None = None
    thread: threading.Thread | None = None


def _state(agent: "Agent") -> _State:
    st = getattr(agent, "_wake_state", None)
    if st is None:
        st = _State()
        agent._wake_state = st
    return st


def bind(agent: "Agent", dispatch: Callable[[str], None]) -> None:
    """Make *agent* wakeable: *dispatch(text)* submits a turn as if typed.
    Called by the interactive UI server; headless agents stay unbound."""
    _state(agent).dispatch = dispatch


def is_bound(agent: "Agent | None") -> bool:
    return agent is not None and _state(agent).dispatch is not None


def pending(agent: "Agent") -> list[Wake]:
    st = _state(agent)
    with st.lock:
        return list(st.wakes)


# ── arming ──────────────────────────────────────────────────────────────────


def _parse_after(after: str, now: float) -> float:
    from agent.core.scheduler import _DURATION_RE, parse_spec
    text = after.strip()
    if _DURATION_RE.match(text.lower()):
        text = "in " + text  # bare "10m"
    kind, ts, _interval, _one = parse_spec(text, now)
    if kind != "at":
        raise ValueError("after must be a one-shot delay or time: '10m', 'in 2h', 'at 07:00'")
    return ts


def _parse_until(until: str, config) -> tuple[int | None, "Job | None"]:
    from agent.core import scheduler
    kind, _, target = until.strip().partition(":")
    kind, target = kind.strip().lower(), target.strip()
    if not target:
        raise ValueError("until must be 'job:<id>', 'file:<path>', 'url:<url>', "
                         "'cmd:<shell>' or 'pid:<n>'")
    if kind == "job":
        from agent.tools.shell import main as shell
        try:
            jid = int(target)
        except ValueError:
            raise ValueError(f"bad job id {target!r}") from None
        if "error" in shell.bg_output(jid):
            raise ValueError(f"no background job {jid}")
        return jid, None
    if kind not in ("file", "url", "cmd", "pid"):
        raise ValueError(f"unknown until kind {kind!r}")
    if kind == "cmd" and not getattr(getattr(config, "tools", None), "allow_shell", False):
        raise ValueError("cmd: needs shell access (tools.allow_shell)")
    probe = scheduler.Job(id="wake", kind="watch", watch_type=kind, watch_target=target)
    refusal = scheduler._watch_refusal(probe, config)
    if refusal:
        raise ValueError(refusal)
    return None, probe


def _check(w: Wake, config, now: float) -> str:
    """Reason the wake fires now, or "" if it does not. Advances watch baselines."""
    if w.job_id is not None:
        from agent.tools.shell import main as shell
        res = shell.bg_output(w.job_id)
        status = res.get("status", "gone")
        if status != "running":
            rc = (res.get("result") or {}).get("returncode")
            return f"background job {w.job_id} {status}" + (f" (exit {rc})" if rc is not None else "")
    if w.watch is not None and now >= w.next_watch_poll:
        from agent.core import scheduler
        wt = w.watch.watch_type
        tick = 2.0 if wt in _FAST_WATCHES else max(
            5.0, float(getattr(getattr(config, "scheduler", None), "watch_tick_seconds", 15)))
        w.next_watch_poll = now + tick
        cur = scheduler._watch_signal(w.watch, config)
        if cur:
            prev = w.watch.watch_state
            w.watch.watch_state = cur
            if scheduler._watch_should_fire(wt, prev, cur):
                return {"file": "file changed", "url": "url content changed",
                        "cmd": "command succeeded", "pid": "process exited"}[wt] \
                    + f": {w.watch.watch_target}"
    if w.deadline and now >= w.deadline:
        return "deadline reached"
    return ""


def arm(agent: "Agent", after: str = "", until: str = "", note: str = "") -> Wake:
    """Arm a wake on *agent*. Raises ValueError on bad input or when the
    condition already holds (nothing to wait for)."""
    if not is_bound(agent):
        raise ValueError("no interactive session to wake (headless or scheduled run) "
                         "— use schedule_task instead")
    if not (after.strip() or until.strip()):
        raise ValueError("give after (a time) and/or until (an event)")
    now = time.time()
    w = Wake(id=secrets.token_hex(2), note=note.strip()[:500], armed_at=now)
    if after.strip():
        w.deadline = _parse_after(after, now)
    if until.strip():
        w.job_id, w.watch = _parse_until(until, agent.config)
    # Take the watch baseline now; a condition that already holds is an answer,
    # not a reason to sleep.
    w.next_watch_poll = 0.0
    already = _check(w, agent.config, now)
    if already:
        raise ValueError(f"already satisfied: {already} — no need to sleep")
    st = _state(agent)
    with st.lock:
        st.wakes.append(w)
        if st.thread is None or not st.thread.is_alive():
            st.thread = threading.Thread(target=_loop, args=(agent, st), daemon=True,
                                         name="wake-watcher")
            st.thread.start()
    logger.info("wake %s armed: %s (%s)", w.id, w.describe(), w.note)
    return w


def cancel(agent: "Agent", wake_id: str) -> int:
    """Remove one wake by id, or all with "all". Returns count removed."""
    st = _state(agent)
    with st.lock:
        before = len(st.wakes)
        st.wakes = [] if wake_id == "all" else [w for w in st.wakes if w.id != wake_id]
        return before - len(st.wakes)


# ── turn hook + delivery ────────────────────────────────────────────────────


def begin_turn(agent: "Agent", user_input: str, source: str) -> str:
    """Called at the start of Agent.chat (not on /continue). A user turn
    supersedes pending wakes. Returns a context note for superseded wakes, or ""."""
    if source == "scheduler" or user_input.startswith(WAKE_PREFIX):
        return ""
    st = _state(agent)
    with st.lock:
        gone, st.wakes = st.wakes, []
    if not gone:
        return ""
    lines = [f"- {w.id}: {w.describe()} — {w.note}" for w in gone]
    return ("[waits superseded by this user message; call sleep_until again "
            "if still needed]\n" + "\n".join(lines))


def wake_text(w: Wake) -> str:
    armed = datetime.fromtimestamp(w.armed_at).strftime("%H:%M:%S")
    return (f"{WAKE_PREFIX}{w.id}] Automatic wake-up (not a user message): "
            f"{w.fired}. You armed it at {armed} with note: {w.note or '(none)'}")


def _loop(agent: "Agent", st: _State) -> None:
    while True:
        time.sleep(_POLL_SECONDS)
        now = time.time()
        with st.lock:
            if not st.wakes:
                st.thread = None
                return
            wakes = list(st.wakes)
        for w in wakes:
            if not w.fired:
                try:
                    w.fired = _check(w, agent.config, now)
                except Exception:
                    logger.warning("wake %s check failed", w.id, exc_info=True)
        # Deliver one fired wake once the agent is idle; the turn it starts
        # sets _turn_busy, which holds back the rest until it ends.
        if getattr(agent, "_turn_busy", False):
            continue
        with st.lock:
            ready = next((w for w in st.wakes if w.fired), None)
            if ready is None:
                continue
            st.wakes.remove(ready)
            dispatch = st.dispatch
        if dispatch is None:
            continue
        logger.info("wake %s fired: %s", ready.id, ready.fired)
        try:
            dispatch(wake_text(ready))
        except Exception:
            logger.warning("wake %s dispatch failed", ready.id, exc_info=True)
        # Let the dispatched turn flip _turn_busy before the next delivery.
        time.sleep(2.0)
