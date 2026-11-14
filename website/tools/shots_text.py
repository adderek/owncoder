#!/usr/bin/env python3
"""Reproducible SVG screenshots of the plain text-flow UI (`agent chat --ui simple`).

Runs the REAL CLI in a child process under a pty (fixed window size), against
the scripted fake LLM (tests/_mock_llm.py), inside the same throw-away HOME +
project sandbox as shots.py. The captured byte stream is replayed through a
tiny line-oriented terminal model (so spinner `\\r` / erase-line output leaves
no garbage) and exported with rich.

    shots_text.py [--out DIR] [--only NAME ...] [--size 100x30] [--list]

Prose goes to stderr, written file paths to stdout. Exit 0 only if every
requested shot was written.

Spinner handling: the simple UI has no switch to disable its spinner, so the
emulator resolves `\\r`, `ESC[K`/`ESC[2K` (overwrite / erase), drops OSC
(title) and private-mode CSI, and keeps only SGR colours. Spinner frames are
overwritten by the real output exactly as a terminal would show them.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shots  # noqa: E402  (stdlib-only at import time; no agent import)

PKG_DIR = shots.PKG_DIR
DEFAULT_OUT = shots.DEFAULT_OUT
SCENES_DIR = shots.SCENES_DIR
TURN_TIMEOUT = 60.0
SANDBOX_WAIT = 90.0
TITLE = "owncoder"
PROMPT_RE = re.compile(r"(?:^|\n)> $|Index\? \[w/p/s\]: $")   # simple UI input prompt, ANSI-stripped
FAKE_IN_RATE = 1480
FAKE_OUT_RATE = 42

log = shots.log


def scene_names() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("text-*.json"))


# ── terminal model ──────────────────────────────────────────────────────────
_TOKEN = re.compile(
    r"\x1b\[([0-9;:?<=>]*)([ -/]*)([@-~])"   # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"    # OSC
    r"|\x1b[()][0-9A-Za-z]|\x1b[=>78cDEHMNOZ]"  # other short escapes
    r"|\r|\n|\x08|\t|[\x00-\x07\x0b-\x0c\x0e-\x1a\x1c-\x1f\x7f]"
    r"|[^\x00-\x1f\x7f\x1b]+")


def render_lines(data: str, width: int) -> list[list[tuple[str, str]]]:
    """Replay `data` into lines of (char, sgr) cells. Handles \\r, \\n, \\b,
    ESC[K/1K/2K, ESC[nC/D (same row), SGR; ignores everything else."""
    lines: list[list[tuple[str, str]]] = [[]]
    col = 0
    sgr = ""
    for m in _TOKEN.finditer(data):
        tok = m.group(0)
        line = lines[-1]
        if m.group(3):  # CSI
            final, args = m.group(3), m.group(1)
            if args.startswith(("?", "<", "=", ">")):
                continue
            nums = [int(x) if x.isdigit() else 0 for x in re.split(r"[;:]", args)] if args else []
            n = nums[0] if nums and nums[0] else 1
            if final == "m":
                sgr = "" if args in ("", "0") else (sgr + m.group(0) if sgr else m.group(0))
                if args in ("", "0"):
                    sgr = ""
            elif final == "K":
                mode = nums[0] if nums else 0
                if mode == 0:
                    del line[col:]
                elif mode == 1:
                    for i in range(min(col, len(line))):
                        line[i] = (" ", "")
                else:
                    line.clear()
            elif final == "C":
                col += n
            elif final == "D":
                col = max(0, col - n)
            elif final == "G":
                col = max(0, n - 1)
            continue
        if tok == "\r":
            col = 0
        elif tok == "\n":
            lines.append([])
            col = 0
        elif tok == "\x08":
            col = max(0, col - 1)
        elif tok == "\t":
            col += 8 - col % 8
        elif tok.startswith("\x1b") or ord(tok[0]) < 32 or tok[0] == "\x7f":
            continue
        else:
            for ch in tok:
                while len(line) < col:
                    line.append((" ", ""))
                if col < len(line):
                    line[col] = (ch, sgr)
                else:
                    line.append((ch, sgr))
                col += 1
    return lines


def lines_to_ansi(lines: list[list[tuple[str, str]]]) -> str:
    out = []
    for line in lines:
        while line and line[-1][0] == " " and not line[-1][1]:
            line = line[:-1]
        s, cur = "", ""
        for ch, st in line:
            if st != cur:
                s += "\x1b[0m" + st
                cur = st
            s += ch
        out.append(s + ("\x1b[0m" if cur else ""))
    while out and not out[-1].strip("\x1b[0m"):
        out.pop()
    return "\n".join(out)


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07", "", s)


# ── normalisation ───────────────────────────────────────────────────────────
_NORM = [
    (re.compile(r"\d+(?:\.\d+)?\s*in-tok/s"), f"{FAKE_IN_RATE} in-tok/s"),
    (re.compile(r"\d+(?:\.\d+)?\s*out-tok/s"), f"{FAKE_OUT_RATE} out-tok/s"),
    (re.compile(r"\d{8}T\d{6}\.\d{3}Z_[0-9a-f]{4}"), "20260101T000000.000Z_0001"),
    # durations such as "1.2s" / "12ms" (model-call line, elapsed)
    (re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:ms|s)\b"), "0.5s"),
]


def normalise_cells(lines: list[list[tuple[str, str]]]) -> None:
    """Volatile values (token rates, durations, plan ids) -> fixed ones, same
    idea as shots._title_and_normalise. Works on cells so that rich's
    per-number colouring (SGR splits inside "1.1s") cannot defeat the regex;
    the replacement keeps the style of the first matched cell."""
    for line in lines:
        for rx, rep in _NORM:
            while True:
                plain = "".join(c for c, _ in line)
                m = next((m for m in rx.finditer(plain) if plain[m.start():m.end()] != rep), None)
                if not m:
                    break
                st = line[m.start()][1]
                line[m.start():m.end()] = [(c, st) for c in rep]


def to_svg(ansi: str, size: tuple[int, int], stem: str) -> str:
    from rich.console import Console
    from rich.terminal_theme import MONOKAI
    from rich.text import Text

    con = Console(record=True, width=size[0], file=open(os.devnull, "w"),
                  force_terminal=True, color_system="truecolor", legacy_windows=False)
    for ln in ansi.split("\n"):
        con.print(Text.from_ansi(ln), no_wrap=True, overflow="crop", soft_wrap=False)
    svg = con.export_svg(title=TITLE, theme=MONOKAI, unique_id=stem)
    return svg


# ── pty driver ──────────────────────────────────────────────────────────────
class Pty:
    def __init__(self, argv: list[str], env: dict, size: tuple[int, int], cwd: str):
        cols, rows = size
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        self.proc = subprocess.Popen(
            argv, stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd,
            start_new_session=True, close_fds=True)
        os.close(slave)
        self.buf = b""

    def pump(self, timeout: float) -> bool:
        r, _, _ = select.select([self.master], [], [], timeout)
        if not r:
            return False
        try:
            d = os.read(self.master, 65536)
        except OSError:
            return False
        if not d:
            return False
        self.buf += d
        return True

    def text(self) -> str:
        return self.buf.decode("utf-8", "replace")

    def send(self, s: str) -> None:
        os.write(self.master, s.encode())

    def wait_prompt(self, after: int, deadline: float) -> bool:
        """True once the input prompt is back after output offset `after`."""
        quiet_since = None
        while time.monotonic() < deadline:
            got = self.pump(0.05)
            if got:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.monotonic()
            tail = strip_ansi(self.buf[after:].decode("utf-8", "replace"))
            tail = render_plain_tail(tail)
            if PROMPT_RE.search(tail) and quiet_since and time.monotonic() - quiet_since > 0.25:
                return True
            if self.proc.poll() is not None:
                return False
        return False

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        try:
            os.close(self.master)
        except OSError:
            pass


def render_plain_tail(s: str) -> str:
    return lines_to_text(render_lines(s, 0))


def lines_to_text(lines) -> str:
    return "\n".join("".join(c for c, _ in ln).rstrip() if i < len(lines) - 1
                     else "".join(c for c, _ in ln) for i, ln in enumerate(lines))


def _expect_list(scene: dict) -> list[str]:
    e = scene.get("expect") or []
    return [e] if isinstance(e, str) else list(e)


def run_scene(name: str, out_dir: Path, size: tuple[int, int]) -> Path:
    scene = json.loads((SCENES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(PKG_DIR / "tests"))
    from _mock_llm import MockLLM  # noqa: E402

    # Wait for the exclusive sandbox (another shots run may hold it).
    t_end = time.monotonic() + SANDBOX_WAIT
    while True:
        try:
            sb = shots.FixedSandbox()
            tmp = sb.__enter__()
            break
        except RuntimeError as exc:
            if time.monotonic() > t_end:
                raise
            log(f"sandbox busy ({exc}); retrying")
            time.sleep(2)
    term = None
    try:
        with MockLLM(scene["scenario"]) as mock:
            proj = shots.make_sandbox(tmp, mock.base_url)
            env = dict(os.environ)          # sandbox HOME/XDG/... set by make_sandbox
            env.update(AGENT_UI_MODE="simple", TERM="xterm-256color",
                       COLUMNS=str(size[0]), LINES=str(size[1]),
                       PYTHONPATH=str(PKG_DIR.parent), PYTHONUNBUFFERED="1",
                       NO_COLOR="", PYTHONIOENCODING="utf-8", LANG="C.UTF-8")
            env.pop("NO_COLOR")
            env.pop("COLORTERM", None)
            term = Pty([sys.executable, "-m", "agent", "chat", "--ui", "simple"],
                       env, size, str(proj))
            deadline = time.monotonic() + TURN_TIMEOUT
            if not term.wait_prompt(0, deadline):
                raise TimeoutError("start-up prompt never appeared:\n" + term.text()[-800:])
            if "Index? [w/p/s]" in strip_ansi(term.text()):
                # first-run question (no way to pre-answer it): skip indexing
                skip_mark = len(term.buf)
                term.send("s\r")
                if not term.wait_prompt(skip_mark, deadline):
                    raise TimeoutError("no chat prompt after index question")
            # screenshot starts at the banner, not at start-up noise
            banner = term.text().find("local-code-agent")
            start = term.text().rfind("\n", 0, banner) + 1 if banner >= 0 else 0
            mark = len(term.buf)
            term.send(scene["prompt"] + "\r")
            if not term.wait_prompt(mark, time.monotonic() + TURN_TIMEOUT):
                raise TimeoutError("turn did not finish:\n" + term.text()[-800:])
            data = term.text()[start:]
            term.send("/quit\r")
            try:
                term.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                raise TimeoutError("/quit did not exit the app")
    finally:
        if term is not None:
            term.kill()
        sb.__exit__(None, None, None)

    cells = render_lines(data, size[0])
    normalise_cells(cells)
    ansi = lines_to_ansi(cells)
    svg = to_svg(ansi, size, name)
    plain = strip_ansi(ansi)
    missing = [e for e in _expect_list(scene) if e not in plain]
    if missing:
        raise AssertionError(f"expected text missing from {name}: {missing}\n{plain}")
    target = out_dir / f"{name}.svg"
    target.write_text(svg, encoding="utf-8")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--size", default="100x30")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    names = scene_names()
    if a.list:
        print("\n".join(names))
        return 0
    todo = a.only or names
    bad = [n for n in todo if n not in names]
    if bad:
        log(f"unknown scene(s): {', '.join(bad)}; known: {', '.join(names)}")
        return 2
    try:
        w, h = (int(x) for x in a.size.lower().split("x"))
    except ValueError:
        log("--size must be WxH, e.g. 100x30")
        return 2
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # One scene per child process: HOME/cwd/env are process-global.
    if len(todo) > 1 or os.environ.get("_SHOTS_CHILD") != "1":
        rc = 0
        for n in todo:
            env = dict(os.environ, _SHOTS_CHILD="1")
            r = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(out),
                 "--only", n, "--size", a.size],
                env=env, stdout=subprocess.PIPE, text=True)
            sys.stdout.write(r.stdout)
            sys.stdout.flush()
            rc |= r.returncode != 0
        return 1 if rc else 0

    n = todo[0]
    try:
        p = run_scene(n, out, (w, h))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log(f"FAILED {n}: {exc}")
        return 1
    log(f"wrote {p}")
    print(p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
