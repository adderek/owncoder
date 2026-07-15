"""Prompt loop — repeat one prompt in the SAME session until stopped.

Unlike /schedule (fresh unattended session per run), a prompt loop re-runs
its prompt as normal turns of the current conversation, keeping context.
Syntax parsed here; each UI drives its own PromptLoop instance:

  /loop <prompt>              repeat back-to-back until stopped
  /loop 5m <prompt>           wait 5m between iterations
  /loop 30s x20 <prompt>      at most 20 iterations
  /loop stop                  stop (also: off, clear)
  /loop                       status
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Unit is mandatory ("30s", "5m") — a bare number could be the prompt's
# first word ("/loop 5 ideas for X") and must not be eaten as an interval.
_INTERVAL_RE = re.compile(r"^(\d+(?:\.\d+)?)([smh])$")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0}


@dataclass
class PromptLoop:
    active: bool = False
    prompt: str = ""
    interval: float = 0.0   # seconds between iterations (0 = back-to-back)
    limit: int = 0          # 0 = until stopped
    done: int = 0
    started_at: float = field(default=0.0)

    def start(self, prompt: str, interval: float, limit: int) -> None:
        self.active = True
        self.prompt = prompt
        self.interval = interval
        self.limit = limit
        self.done = 0
        self.started_at = time.time()

    def stop(self) -> None:
        self.active = False

    def record_iteration(self) -> bool:
        """Count a finished iteration. Returns True if the loop continues."""
        if not self.active:
            return False
        self.done += 1
        if self.limit and self.done >= self.limit:
            self.active = False
            return False
        return True

    def status_line(self) -> str:
        if not self.active:
            return "no prompt loop running. Usage: /loop [<interval>] [xN] <prompt> | stop"
        iv = f"every {int(self.interval)}s" if self.interval else "back-to-back"
        lim = f"{self.done}/{self.limit}" if self.limit else f"{self.done} done"
        mins = (time.time() - self.started_at) / 60
        return (f"loop active ({iv}, {lim}, running {mins:.0f}m): {self.prompt[:120]}\n"
                f"stop with: /loop stop")


def parse_loop_args(arg: str) -> tuple[str, dict]:
    """Parse /loop args. Returns (action, params).

    action: "status" | "stop" | "start" | "error"
    start params: prompt, interval (s), limit; error params: msg."""
    v = arg.strip()
    if not v:
        return "status", {}
    if v.lower() in ("stop", "off", "clear", "cancel"):
        return "stop", {}
    interval = 0.0
    limit = 0
    tokens = v.split()
    while tokens:
        tok = tokens[0]
        m = _INTERVAL_RE.match(tok.lower())
        if m and interval == 0.0:
            interval = float(m.group(1)) * _UNIT_S[m.group(2)]
            tokens.pop(0)
            continue
        if tok.lower().startswith("x") and tok[1:].isdigit() and limit == 0:
            limit = int(tok[1:])
            tokens.pop(0)
            continue
        break
    prompt = " ".join(tokens).strip()
    if not prompt:
        return "error", {"msg": "Usage: /loop [<interval like 30s|5m|1h>] [xN] <prompt> | stop"}
    if interval and interval < 5:
        interval = 5.0  # floor: don't hammer the model
    return "start", {"prompt": prompt, "interval": interval, "limit": limit}
