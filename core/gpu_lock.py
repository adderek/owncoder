"""Cross-process GPU concurrency limiter.

The in-process `asyncio.Semaphore` in `model_status` only bounds parallel GPU
calls within a single agent process. When several agents run at once against
one llama.cpp/vLLM endpoint they oversubscribe the GPU. This adds a counting
semaphore shared across processes via `flock`, so the global cap holds no
matter how many agents are running.

Mechanism: a directory of N empty lock files; acquiring a slot means taking an
exclusive `flock` on any one of them. Each acquisition opens its own fd, so
acquisitions contend even within the same process. `flock` is released when the
fd is closed *or the process dies*, so crashed agents never leave stale slots
(no PID files, no cleanup needed).

The lock dir is keyed off the GPU endpoint URL and placed in a shared,
per-user runtime location (XDG_RUNTIME_DIR or the temp dir) — NOT under a
project's `.agent/`, since agents in different working directories must share
one GPU's lock.
"""
from __future__ import annotations

import asyncio
import errno
import fcntl
import hashlib
import os
import random
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

_DISABLE_TOKENS = {"off", "none", "no", "disabled", "false", "0"}


def resolve_lock_dir(setting: str, endpoint_key: str) -> "Path | None":
    """Map a `gpu_lock_dir` config value to a directory, or None if disabled.

    "" / unset  → auto path derived from `endpoint_key`
    off/none/…  → None (cross-process lock disabled)
    <path>      → that explicit (shared) directory
    """
    s = (setting or "").strip()
    if s.lower() in _DISABLE_TOKENS:
        return None
    if s:
        return Path(s).expanduser()
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    digest = hashlib.sha1(endpoint_key.encode("utf-8")).hexdigest()[:16]
    return Path(base) / "owncoder-gpu" / digest


class CrossProcessSemaphore:
    """File-lock counting semaphore shared across processes."""

    def __init__(self, slots: int, lock_dir: "str | Path", poll_s: float = 0.05) -> None:
        self._slots = max(1, int(slots))
        self._poll = poll_s
        self._dir = Path(lock_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._paths = [self._dir / f"slot-{i}.lock" for i in range(self._slots)]
        for p in self._paths:
            # 0o600: only this user; lock files are empty and reused.
            os.close(os.open(p, os.O_RDWR | os.O_CREAT, 0o600))

    def _try_acquire(self) -> "int | None":
        """Return an fd holding a slot lock, or None if all slots are taken."""
        for p in self._paths:
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError as exc:
                os.close(fd)
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
        return None

    @asynccontextmanager
    async def acquire(self):
        """Block (cooperatively) until a global slot is free, then hold it."""
        fd: "int | None" = None
        try:
            while fd is None:
                fd = self._try_acquire()
                if fd is None:
                    # jittered poll to avoid lockstep thundering herd
                    await asyncio.sleep(self._poll * (1.0 + random.random()))
            yield
        finally:
            if fd is not None:
                os.close(fd)  # releases the flock
