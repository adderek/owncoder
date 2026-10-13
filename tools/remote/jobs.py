"""remote_start / remote_job — long-running remote commands that don't block a turn.

Each job is its own `ssh` subprocess (own connection, so jobs and remote_exec
calls run in parallel). A reader thread fills a byte ring buffer
(`remote.job_buffer_bytes`); the model reads new output incrementally, can
wait for a regex, type into stdin, and stop the job. Offsets are absolute
(bytes since start); the ring drops its head when full and reports the loss.

tty=True (default) runs `ssh -tt`: the remote side gets a pty, so ctrl-c
works and closing the connection SIGHUPs the remote process tree — a capture
started this way does not outlive the job. stderr is merged into the stream.
"""
from __future__ import annotations

import atexit
import os
import re
import signal
import subprocess
import threading
import time

from agent.tools import register
from agent.security import audit as _audit
from agent.tools.serial_console.port import clean as _clean
from . import ssh as _ssh

_lock = threading.Lock()
_jobs: dict[int, "Job"] = {}
_seq = 0
_READ_CAP = 16_000          # bytes returned per read; the rest waits for the next
_KEEP_FINISHED = 16         # finished jobs retained once delivered
_KEYS = {"ctrl-c": b"\x03", "ctrl-d": b"\x04", "ctrl-z": b"\x1a", "ctrl-\\": b"\x1c"}


class Job:
    def __init__(self, jid: int, host: str, cmd: str, tty: bool, buffer_bytes: int):
        self.id, self.host, self.cmd, self.tty = jid, host, cmd, tty
        self.buffer_bytes = buffer_bytes
        self.proc: subprocess.Popen | None = None
        self.buf = bytearray()
        self.base = 0               # absolute offset of buf[0]
        self.cursor = 0             # absolute offset the model has read up to
        self.cond = threading.Condition()
        self.started = time.time()
        self.finished = 0.0
        self.status = "running"     # running | exited | stopped | timeout | error
        self.returncode: int | None = None
        self.delivered = False
        self.stopping = False       # stop requested: a clean exit after ctrl-c is "stopped"
        self.bg_reg: int | None = None

    @property
    def end(self) -> int:
        return self.base + len(self.buf)

    def running(self) -> bool:
        return self.status == "running"

    # ── threads ──────────────────────────────────────────────────────────
    def reader(self) -> None:
        fd = self.proc.stdout.fileno()
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                break
            with self.cond:
                self.buf += chunk
                over = len(self.buf) - self.buffer_bytes
                if over > 0:
                    del self.buf[:over]
                    self.base += over
                self.cond.notify_all()
        rc = self.proc.wait()
        with self.cond:
            if self.status == "running":
                self.status = "stopped" if self.stopping else "exited"
            self.returncode = rc
            self.finished = time.time()
            self.cond.notify_all()
        _audit.record("remote_job_end", host=self.host, job_id=self.id,
                      status=self.status, returncode=rc)
        if self.bg_reg is not None:
            try:
                from agent.core import background
                background.unregister(self.bg_reg)
            except Exception:
                pass

    def watchdog(self, limit: float) -> None:
        try:
            self.proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            self.kill("timeout")

    # ── control ──────────────────────────────────────────────────────────
    def kill(self, status: str) -> None:
        with self.cond:
            if self.status != "running" or self.proc is None:
                return
            self.status = status
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self.proc.kill()

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def close_stdin(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass

    def wait(self, expect: re.Pattern | None, timeout: float) -> bool:
        """Block until *expect* matches past the cursor, the job ends, or
        *timeout*. No pattern → just collect for *timeout*. → matched."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                if expect is not None and expect.search(
                        _clean(bytes(self.buf[max(self.cursor, self.base) - self.base:]))):
                    return True
                if not self.running():
                    return False
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)

    def take(self, from_start: bool = False) -> dict:
        """New output since the cursor (capped), advancing it."""
        with self.cond:
            start = self.base if from_start else self.cursor
            lost = max(0, self.base - start)
            lo = max(start, self.base) - self.base
            chunk = bytes(self.buf[lo:lo + _READ_CAP])
            self.cursor = self.base + lo + len(chunk)
            more = self.cursor < self.end
            if not self.running() and not more:
                self.delivered = True
            out = {"job_id": self.id, "host": self.host, "status": self.status,
                   "output": _clean(chunk)}
            if self.returncode is not None:
                out["returncode"] = self.returncode
            if lost:
                out["lost_bytes"] = lost
            if more:
                out["more"] = True
            return out

    def summary(self) -> dict:
        return {"job_id": self.id, "host": self.host, "cmd": self.cmd, "status": self.status,
                "age_s": round(time.time() - self.started, 1),
                "unread_bytes": max(0, self.end - max(self.cursor, self.base))}


def _gate(host: str, what: str, confirmed: bool):
    from .main import _gate as g
    return g(host, what, confirmed)


def _compile(pattern: str | None) -> tuple[re.Pattern | None, dict | None]:
    if not pattern:
        return None, None
    try:
        return re.compile(pattern), None
    except re.error as e:
        return None, {"error": f"bad regex {pattern!r}: {e}"}


def _prune() -> None:
    """Under _lock: drop the oldest delivered finished jobs past _KEEP_FINISHED."""
    done = [j for j in _jobs.values() if not j.running() and j.delivered]
    for j in done[:max(0, len(done) - _KEEP_FINISHED)]:
        _jobs.pop(j.id, None)


@register(
    "remote_start",
    {
        "description": (
            "Start a long-running command on a configured remote host WITHOUT "
            "waiting (tcpdump, a server, a log tail, an interactive shell). "
            "Returns job_id; output goes to a buffer read with remote_job. "
            "Each job is its own SSH connection, so other remote calls run "
            "meanwhile. Use wait_for to return once it is ready, e.g. "
            "'listening on' for tcpdump."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Configured host name"},
                "cmd": {"type": "string", "description": "Command line for the remote shell"},
                "wait_for": {"type": "string", "description": "Regex to wait for before returning"},
                "timeout": {"type": "integer", "description": "Seconds to wait for wait_for (default 10)"},
                "tty": {"type": "boolean", "description": "Remote pty (default true): ctrl-c works, job end kills the remote process; stderr merged"},
            },
            "required": ["host", "cmd"],
        },
    },
)
def remote_start(host: str, cmd: str, wait_for: str | None = None, timeout: int | None = None,
                 tty: bool = True, _confirmed: bool = False) -> dict:
    if not cmd or not cmd.strip():
        return {"error": "cmd must be non-empty"}
    expect, err = _compile(wait_for)
    if err:
        return err
    h, err = _gate(host, f"Background command: {cmd}", _confirmed)
    if err is not None:
        return err
    from .main import _config
    rcfg = _config.remote
    global _seq
    with _lock:
        live = sum(1 for j in _jobs.values() if j.running())
        if live >= int(rcfg.max_jobs):
            return {"error": f"{live} remote jobs running (remote.max_jobs) — stop one first",
                    "jobs": [j.summary() for j in _jobs.values() if j.running()]}
        _seq += 1
        job = Job(_seq, h.name, cmd, bool(tty), int(rcfg.job_buffer_bytes))
        _jobs[job.id] = job
        _prune()
    argv = _ssh.build_argv(rcfg, h, cmd, tty=job.tty)
    try:
        job.proc = subprocess.Popen(argv, env=_ssh._env(), start_new_session=True,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    except OSError as e:
        with _lock:
            _jobs.pop(job.id, None)
        return {"error": f"{type(e).__name__}: {e}", "host": host}
    _audit.record("remote_start", host=h.name, target=f"{h.user}@{h.host}:{h.port}",
                  cmd=cmd, job_id=job.id, tty=job.tty)
    try:
        from agent.core import background
        job.bg_reg = background.register_external(
            f"remote:{h.name}:{cmd[:50]}", "remote-job", cancel=lambda: job.kill("stopped"))
    except Exception:
        pass
    threading.Thread(target=job.reader, daemon=True, name=f"remote-job-{job.id}").start()
    threading.Thread(target=job.watchdog, args=(float(rcfg.job_timeout_s),),
                     daemon=True, name=f"remote-job-{job.id}-wd").start()
    if expect is None:
        return {"job_id": job.id, "host": h.name, "status": "running",
                "note": f"Read output with remote_job(job_id={job.id}); stop with action=stop."}
    matched = job.wait(expect, float(timeout or 10))
    out = job.take()
    out["matched"] = matched
    return out


@register(
    "remote_job",
    {
        "description": (
            "Work with a remote_start job: read (new output since last read; "
            "optional wait_for regex / timeout to wait for more), send (type "
            "data and/or a key into its stdin), stop (ctrl-c, then kill). "
            "Omit job_id to list jobs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "action": {"type": "string", "enum": ["read", "send", "stop"], "description": "Default read"},
                "data": {"type": "string", "description": "send: text to type"},
                "newline": {"type": "boolean", "description": "send: append newline (default true)"},
                "key": {"type": "string", "enum": [*_KEYS, "eof"], "description": "send: control key; eof closes stdin"},
                "wait_for": {"type": "string", "description": "Regex to wait for in new output"},
                "timeout": {"type": "integer", "description": "Seconds to wait (default 10 with wait_for, else 0)"},
                "from_start": {"type": "boolean", "description": "read: whole buffer, not just new output"},
            },
        },
    },
)
def remote_job(job_id: int | None = None, action: str = "read", data: str = "",
               newline: bool = True, key: str | None = None, wait_for: str | None = None,
               timeout: int | None = None, from_start: bool = False,
               _confirmed: bool = False) -> dict:
    with _lock:
        if job_id is None:
            return {"jobs": [j.summary() for j in _jobs.values()]}
        job = _jobs.get(int(job_id))
    if job is None:
        return {"error": f"no remote job {job_id}"}
    expect, err = _compile(wait_for)
    if err:
        return err
    wait = float(timeout if timeout is not None else (10 if expect else 0))

    if action == "stop":
        if job.running():
            job.stopping = True
            if job.tty:
                try:
                    job.write(_KEYS["ctrl-c"])
                except OSError:
                    pass
                with job.cond:
                    job.cond.wait_for(lambda: not job.running(), timeout=2)
            job.kill("stopped")
            with job.cond:
                job.cond.wait_for(lambda: not job.running() and job.returncode is not None,
                                  timeout=5)
        _audit.record("remote_job", host=job.host, job_id=job.id, action="stop")
        return job.take(from_start)

    if action == "send":
        if not job.running():
            return {"error": f"job {job.id} is {job.status}", **job.take()}
        if key and key not in _KEYS and key != "eof":
            return {"error": f"unknown key {key!r}"}
        if key == "ctrl-c" and not job.tty:
            return {"error": "job has no tty — ctrl-c needs tty=true; use action=stop"}
        payload = (data + ("\n" if newline else "")).encode() if data else b""
        if key and key != "eof":
            payload += _KEYS[key]
        what = f"Input to job {job.id} ({job.cmd}): {data!r}" + (f" + {key}" if key else "")
        _, err = _gate(job.host, what, _confirmed)
        if err is not None:
            return err
        try:
            if payload:
                job.write(payload)
            if key == "eof":
                job.close_stdin()
        except OSError as e:
            return {"error": f"write failed: {e}", **job.take()}
        _audit.record("remote_job", host=job.host, job_id=job.id, action="send",
                      data=data, key=key)
        if wait:
            job.wait(expect, wait)
        return job.take()

    if action != "read":
        return {"error": f"unknown action {action!r}"}
    matched = job.wait(expect, wait) if wait else False
    out = job.take(from_start)
    if expect is not None:
        out["matched"] = matched
    return out


def kill_host(name: str | None = None) -> int:
    """Kill running jobs (all, or one host's). → count."""
    with _lock:
        live = [j for j in _jobs.values() if j.running() and (name is None or j.host == name)]
    for j in live:
        j.kill("stopped")
    return len(live)


def reset() -> None:
    kill_host()
    with _lock:
        _jobs.clear()


def drain_finished() -> list[dict]:
    """Finished jobs whose end the model has not seen → summaries, marked delivered."""
    out = []
    with _lock:
        for j in _jobs.values():
            if not j.running() and not j.delivered:
                j.delivered = True
                with j.cond:
                    tail = _clean(bytes(j.buf[-300:])).strip()
                out.append({**j.summary(), "returncode": j.returncode, "tail": tail})
    return out


atexit.register(kill_host)
