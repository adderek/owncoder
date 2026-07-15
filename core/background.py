"""Registry of labeled background jobs so UIs can show and kill them.

Anything that outlives a turn registers here: post-turn QA summarization,
idle compaction, scheduler job runs, delegate calls. Asyncio tasks are
cancellable from any thread (cancel is marshalled onto their loop); plain
thread/foreign-loop work registers with an optional cancel callable and is
otherwise just visible.
"""
from __future__ import annotations

import asyncio
import itertools
import threading
import time
from typing import Any, Callable

_lock = threading.Lock()
_jobs: dict[int, dict[str, Any]] = {}
_ids = itertools.count(1)


def register_task(task: asyncio.Task, label: str, kind: str = "task") -> asyncio.Task:
    """Track an asyncio task created on the currently running loop."""
    loop = asyncio.get_running_loop()

    def _cancel() -> None:
        loop.call_soon_threadsafe(task.cancel)

    jid = _register(label, kind, done=task.done, cancel=_cancel)
    task.add_done_callback(lambda _t: unregister(jid))
    return task


def register_external(label: str, kind: str,
                      cancel: Callable[[], None] | None = None) -> int:
    """Track non-asyncio work (thread, subprocess, remote run).

    Caller must unregister(jid) when done. Returns the job id."""
    return _register(label, kind, done=lambda: False, cancel=cancel)


def _register(label: str, kind: str, done, cancel) -> int:
    jid = next(_ids)
    with _lock:
        _jobs[jid] = {"id": jid, "label": label, "kind": kind,
                      "started": time.time(), "done": done, "cancel": cancel}
    return jid


def unregister(jid: int) -> None:
    with _lock:
        _jobs.pop(jid, None)


def jobs() -> list[dict[str, Any]]:
    """Snapshot of live jobs: id, label, kind, age (s), killable."""
    now = time.time()
    with _lock:
        items = list(_jobs.values())
    out = []
    for j in items:
        try:
            if j["done"]():
                continue
        except Exception:
            continue
        out.append({"id": j["id"], "label": j["label"], "kind": j["kind"],
                    "age": round(now - j["started"], 1),
                    "killable": j["cancel"] is not None})
    return out


def cancel(jid: int) -> bool:
    with _lock:
        j = _jobs.get(jid)
    if j is None or j["cancel"] is None:
        return False
    try:
        j["cancel"]()
        return True
    except Exception:
        return False


def cancel_all() -> int:
    n = 0
    for j in jobs():
        if j["killable"] and cancel(j["id"]):
            n += 1
    return n
