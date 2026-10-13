"""One open serial console: background reader, byte ring buffer, armed
auto-responses and an optional on-disk transcript.

The console is a stream, not request/response: the reader thread runs from
the first open until the process exits, so boot logs and panics that arrive
between model turns are kept. Offsets are absolute (bytes since open); the
ring drops its head past ``buffer_bytes`` and remembers how much it dropped.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_ANSI = re.compile(rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_SCAN_OVERLAP = 512        # bytes re-scanned so a pattern split across reads still matches


class PortError(Exception):
    pass


@dataclass
class Trigger:
    pattern: re.Pattern
    response: bytes
    once: bool
    expires: float
    scan_from: int
    fired: int = 0


def clean(b: bytes) -> str:
    """Console bytes → model text: no ANSI, no CR noise, invalid UTF-8 replaced."""
    b = _ANSI.sub(b"", b).replace(b"\r\n", b"\n").replace(b"\r", b"")
    return b.decode("utf-8", errors="replace")


@dataclass
class Port:
    cfg: object                    # SerialPortConfig
    buffer_bytes: int = 1024 * 1024
    log_dir: Path | None = None    # None = no transcript
    _ser: object = None
    _buf: bytearray = field(default_factory=bytearray)
    _base: int = 0                 # absolute offset of _buf[0]
    cursor: int = 0                # what the model has been shown
    last_rx: float = 0.0
    error: str = ""
    _triggers: list = field(default_factory=list)
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _console_log: object = None
    _actions_log: object = None

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._thread is not None and self._thread.is_alive()

    def open(self) -> None:
        if self.is_open:
            return
        try:
            import serial  # pyserial, optional extra [serial]
        except ImportError as e:
            raise PortError("pyserial not installed: pip install -e '.[serial]'") from e
        c = self.cfg
        kw = dict(baudrate=int(c.baud), bytesize=int(c.bytesize), parity=str(c.parity).upper(),
                  stopbits=c.stopbits, rtscts=bool(c.rtscts), xonxoff=bool(c.xonxoff),
                  timeout=0.1, write_timeout=5)
        if "://" not in c.url:
            kw["exclusive"] = True   # flock: a second opener (picocom, other agent) fails
        try:
            self._ser = serial.serial_for_url(c.url, **kw)
        except (serial.SerialException, ValueError, OSError) as e:
            self._ser = None
            raise PortError(f"cannot open {c.url}: {e}") from e
        self.error = ""
        self._stop.clear()
        self._open_logs()
        self._action("open", url=c.url, baud=c.baud)
        self._thread = threading.Thread(target=self._reader, name=f"serial-{c.name}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._thread = None
        for f in (self._console_log, self._actions_log):
            if f is not None:
                f.close()
        self._console_log = self._actions_log = None

    def _open_logs(self) -> None:
        if self.log_dir is None or self._console_log is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._console_log = open(self.log_dir / "console.log", "ab", buffering=0)
        self._actions_log = open(self.log_dir / "actions.jsonl", "a", encoding="utf-8", buffering=1)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._console_log.write(f"\n===== opened {self.cfg.url} @ {self.cfg.baud} {stamp} =====\n".encode())

    def _action(self, action: str, **fields) -> None:
        if self._actions_log is not None:
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "action": action, **fields}
            self._actions_log.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── reader ─────────────────────────────────────────────────────────────
    def _reader(self) -> None:
        ser = self._ser
        while not self._stop.is_set():
            try:
                chunk = ser.read(4096)
            except Exception as e:  # unplugged adapter, closed socket
                with self._cond:
                    self.error = f"{type(e).__name__}: {e}"
                    self._cond.notify_all()
                self._action("error", error=self.error)
                try:
                    ser.close()
                except Exception:
                    pass
                self._ser = None
                return
            if not chunk:
                continue
            if self._console_log is not None:
                self._console_log.write(chunk)
            with self._cond:
                self._buf += chunk
                self.last_rx = time.monotonic()
                over = len(self._buf) - self.buffer_bytes
                if over > 0:
                    del self._buf[:over]
                    self._base += over
                fire = self._match_triggers()
                self._cond.notify_all()
            for t, resp in fire:
                self._write(resp)
                self._action("trigger", pattern=t.pattern.pattern.decode(errors="replace"),
                             sent=resp.decode(errors="replace"))

    def _match_triggers(self) -> list:
        """Under _cond. Returns (trigger, response) to send outside the lock."""
        now = time.monotonic()
        end = self.end
        fire = []
        keep = []
        for t in self._triggers:
            if now > t.expires:
                continue
            start = max(t.scan_from, self._base)
            m = t.pattern.search(self._buf, start - self._base)
            if m:
                t.fired += 1
                t.scan_from = self._base + m.end()
                fire.append((t, t.response))
                if t.once:
                    continue
            else:
                t.scan_from = max(t.scan_from, end - _SCAN_OVERLAP)
            keep.append(t)
        self._triggers = keep
        return fire

    # ── buffer access ──────────────────────────────────────────────────────
    @property
    def end(self) -> int:
        return self._base + len(self._buf)

    def slice(self, start: int) -> tuple[bytes, int]:
        """Bytes from absolute *start* to end → (data, bytes lost to the ring)."""
        with self._cond:
            lost = max(0, self._base - start)
            return bytes(self._buf[max(start, self._base) - self._base:]), lost

    def wait(self, start: int, expect: re.Pattern | None, timeout: float,
             idle: float) -> tuple[bool, bool]:
        """Block until *expect* matches after *start*, or (no expect) the line is
        quiet for *idle* seconds after some output, or *timeout*.
        → (matched, timed_out)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if expect is not None:
                    if expect.search(self._buf, max(start, self._base) - self._base):
                        return True, False
                elif self.end > start and time.monotonic() - self.last_rx >= idle:
                    return False, False
                if self.error:
                    return False, False
                now = time.monotonic()
                if now >= deadline:
                    return False, True
                step = deadline - now
                if expect is None and self.end > start:
                    step = min(step, max(0.01, idle - (now - self.last_rx)))
                self._cond.wait(step)

    # ── output ─────────────────────────────────────────────────────────────
    def _write(self, data: bytes) -> None:
        ser = self._ser
        if ser is None:
            raise PortError(self.error or "port closed")
        ser.write(data)
        ser.flush()

    def send(self, data: bytes, what: str) -> int:
        """Write *data*; returns the absolute offset output is read from."""
        with self._cond:
            pos = self.end
        self._write(data)
        self._action("send", what=what, data=data.decode(errors="replace"))
        return pos

    def send_break(self, duration: float = 0.25) -> int:
        with self._cond:
            pos = self.end
        if self._ser is None:
            raise PortError(self.error or "port closed")
        self._ser.send_break(duration)
        self._action("send", what="break")
        return pos

    def arm(self, pattern: re.Pattern, response: bytes, once: bool, ttl_s: float) -> None:
        with self._cond:
            self._triggers.append(Trigger(pattern, response, once,
                                          time.monotonic() + ttl_s, self.end))
        self._action("arm", pattern=pattern.pattern.decode(errors="replace"),
                     response=response.decode(errors="replace"), once=once, ttl_s=ttl_s)

    def disarm(self) -> int:
        with self._cond:
            n = len(self._triggers)
            self._triggers = []
        self._action("disarm", count=n)
        return n

    def triggers(self) -> list[dict]:
        now = time.monotonic()
        with self._cond:
            return [{"pattern": t.pattern.pattern.decode(errors="replace"),
                     "response": t.response.decode(errors="replace"),
                     "once": t.once, "fired": t.fired,
                     "expires_in_s": max(0, round(t.expires - now))}
                    for t in self._triggers if t.expires > now]
