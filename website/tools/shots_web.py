#!/usr/bin/env python3
"""Reproducible PNG screenshots of the browser UI (`agent chat --ui http`).

Real agent as a child process, scripted fake LLM (tests/_mock_llm.py), the
same throw-away HOME + project sandbox as shots.py, system Chromium headless.

    shots_web.py [--out DIR] [--only NAME ...] [--list] [--chromium PATH]

Prose goes to stderr, written file paths to stdout. Exit 0 only if every
requested shot was written; exit 3 if no Chromium binary is found.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shots  # noqa: E402  (FixedSandbox, make_sandbox; imports no agent code)

PKG_DIR = shots.PKG_DIR
DEFAULT_OUT = shots.DEFAULT_OUT
SCENES_DIR = HERE / "scenes"
TURN_TIMEOUT = 60.0
CHROMIUMS = ("chromium", "chromium-browser", "google-chrome")
SANDBOX_WAIT = 90.0
URL_RE = re.compile(r"http://127\.0\.0\.1:(\d+)/\?token=([\w.~-]+)")

log = shots.log


def scene_names() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("web-*.json"))


def find_chromium(explicit: str | None) -> str | None:
    if explicit:
        return shutil.which(explicit) or (explicit if Path(explicit).exists() else None)
    for c in CHROMIUMS:
        p = shutil.which(c)
        if p:
            return p
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Api:
    """Tiny cookie-keeping client for the UI's HTTP API."""

    def __init__(self, port: int, token: str) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self.op = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.op.open(f"{self.base}/?token={token}", timeout=10).read()  # sets cookie

    def get(self, path: str) -> dict:
        return json.loads(self.op.open(self.base + path, timeout=10).read())

    def post(self, path: str, obj: dict) -> dict:
        req = urllib.request.Request(
            self.base + path, json.dumps(obj).encode(),
            {"Content-Type": "application/json", "Origin": self.base})
        return json.loads(self.op.open(req, timeout=10).read())


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def start_agent(proj: Path, port: int, env: dict) -> tuple[subprocess.Popen, str, list[str]]:
    code = "import sys; from agent.cli.main import main; sys.exit(main())"
    proc = subprocess.Popen(
        [sys.executable, "-c", code, "chat", "--ui", "http"], cwd=proj, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True)
    buf: list[str] = []

    def pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            buf.append(line)

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        m = URL_RE.search("".join(buf))
        if m:
            return proc, m.group(2), buf
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    stop(proc)
    raise RuntimeError("agent did not print the UI URL; output:\n" + "".join(buf)[-2000:])


class Cdp:
    """Headless Chromium driven over --remote-debugging-pipe (stdlib only: JSON
    messages, NUL-terminated, on fds 3 (to chrome) and 4 (from chrome)).

    Needed because the page keeps an SSE stream open, so --screenshot /
    --virtual-time-budget either fire before the conversation is rendered or
    never finish.
    """

    FLAGS = ["--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
             "--disable-extensions", "--disable-background-networking", "--disable-sync",
             "--disable-component-update", "--hide-scrollbars", "--force-color-profile=srgb",
             "--remote-debugging-pipe", "about:blank"]

    def __init__(self, binary: str, profile: Path) -> None:
        to_r, to_w = os.pipe()
        from_r, from_w = os.pipe()

        def pre() -> None:
            os.dup2(to_r, 3)
            os.dup2(from_w, 4)

        self.proc = subprocess.Popen(
            [binary, f"--user-data-dir={profile}", *self.FLAGS], preexec_fn=pre,
            close_fds=False, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        os.close(to_r)
        os.close(from_w)
        self._w = os.fdopen(to_w, "wb", buffering=0)
        self._r = os.fdopen(from_r, "rb", buffering=0)
        self._buf = b""
        self._id = 0
        self.session = ""

    def _read(self, timeout: float) -> dict:
        import select
        end = time.monotonic() + timeout
        while b"\0" not in self._buf:
            left = end - time.monotonic()
            if left <= 0 or not select.select([self._r], [], [], left)[0]:
                raise TimeoutError("no answer from chromium")
            chunk = self._r.read(1 << 20)
            if not chunk:
                raise RuntimeError("chromium closed the debugging pipe")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\0", 1)
        return json.loads(raw)

    def call(self, method: str, params: dict | None = None, timeout: float = 30) -> dict:
        self._id += 1
        msg = {"id": self._id, "method": method, "params": params or {}}
        if self.session:
            msg["sessionId"] = self.session
        self._w.write(json.dumps(msg).encode() + b"\0")
        end = time.monotonic() + timeout
        while True:
            m = self._read(max(0.1, end - time.monotonic()))
            if m.get("id") == self._id:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        if "exceptionDetails" in r:
            raise RuntimeError(f"js error: {r['exceptionDetails'].get('text')}")
        return r["result"].get("value")

    def open_page(self, w: int, h: int, mobile: bool) -> None:
        tid = self.call("Target.createTarget", {"url": "about:blank"})["targetId"]
        self.session = self.call("Target.attachToTarget",
                                 {"targetId": tid, "flatten": True})["sessionId"]
        self.call("Page.enable")
        self.call("Emulation.setDeviceMetricsOverride", {
            "width": w, "height": h, "deviceScaleFactor": 1, "mobile": mobile})

    def wait_js(self, expr: str, timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.js(expr):
                return True
            time.sleep(0.3)
        return False

    def close(self) -> None:
        try:
            self.call("Browser.close", timeout=5)
        except Exception:
            pass
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            pass
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
                self.proc.wait(5)
            except Exception:
                pass
        try:
            os.killpg(self.proc.pid, 9)   # crashpad/zygote stragglers
        except OSError:
            pass


def run_scene(name: str, out_dir: Path, binary: str, ident: list[str]) -> Path:
    scene = json.loads((SCENES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    w, h = scene.get("viewport", [1280, 800])
    expect = shots._expect_list(scene)
    sys.path.insert(0, str(PKG_DIR / "tests"))
    from _mock_llm import MockLLM  # noqa: E402

    deadline = time.monotonic() + SANDBOX_WAIT
    while True:
        try:
            sb = shots.FixedSandbox()
            tmp = sb.__enter__()
            break
        except RuntimeError as exc:
            if time.monotonic() > deadline:
                raise
            log(f"sandbox busy ({exc}); retrying")
            time.sleep(2)
    proc = None
    try:
        with MockLLM(scene["scenario"]) as mock:
            proj = shots.make_sandbox(tmp, mock.base_url)
            port = free_port()
            toml = proj / "agent.toml"
            txt = toml.read_text(encoding="utf-8")
            assert '[ui]\nmode = "textual"' in txt
            txt = txt.replace('[ui]\nmode = "textual"', (
                '[ui]\nmode = "http"\nhttp_host = "127.0.0.1"\n'
                f'http_port = {port}\nhttp_tls_auto = "off"\nconnect_approval = false'
                '\n\n[classify]\nhide_unconfigured_notice = true'))
            toml.write_text(txt, encoding="utf-8")
            env = dict(os.environ, AGENT_UI_MODE="http", COLUMNS="250", NO_COLOR="1",
                       PYTHONUNBUFFERED="1")
            proc, token, _buf = start_agent(proj, port, env)
            api = Api(port, token)
            # the header flashes "pick work mode" until one is chosen this session
            api.post("/api/model", {"action": "mode", "mode": "private"})
            r = api.post("/api/chat", {"text": scene["prompt"]})
            if not r.get("ok"):
                raise RuntimeError(f"chat rejected: {r}")
            end = time.monotonic() + TURN_TIMEOUT
            stable = 0
            while time.monotonic() < end:
                st = api.get("/api/state")
                done = (not st.get("busy")
                        and all(e in json.dumps(st.get("messages")) for e in expect))
                stable = stable + 1 if done else 0
                if stable >= 3:
                    break
                time.sleep(0.3)
            else:
                raise TimeoutError("turn did not finish")
            if "demo-project" not in str(st.get("workdir")):
                raise AssertionError(f"unexpected workdir {st.get('workdir')!r}")
            url = f"http://127.0.0.1:{port}/?token={token}"
            out = out_dir / f"{name}.png"
            cdp = Cdp(binary, tmp / "chromium-profile")
            try:
                cdp.open_page(w, h, bool(scene.get("mobile")))
                for k, v in (scene.get("local_storage") or {}).items():
                    cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{port}/?token={token}"})
                    cdp.wait_js("document.readyState === 'complete'", 15)
                    cdp.js(f"localStorage.setItem({json.dumps(k)}, {json.dumps(v)})")
                cdp.call("Page.navigate", {"url": url})
                probe = " && ".join(f"document.body.innerText.includes({json.dumps(e)})"
                                    for e in expect) or "true"
                if not cdp.wait_js(probe, 30):
                    raise TimeoutError("conversation never rendered in the page")
                time.sleep(1.5)   # fade-in animations, drawer loaders
                dom = cdp.js("document.documentElement.outerHTML") or ""
                cdp.js("history.replaceState(null, '', '/')")  # token must not be on screen
                png = cdp.call("Page.captureScreenshot", {"format": "png"})["data"]
            finally:
                cdp.close()
            import base64
            out.write_bytes(base64.b64decode(png))
            leaks = [s for s in [token, *ident] if s and s in dom]
            if leaks:
                out.unlink()
                raise AssertionError(f"DOM contains identifying strings: {len(leaks)} hit(s)")
            return out
    finally:
        stop(proc)
        sb.__exit__(None, None, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--chromium", metavar="PATH")
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
    binary = find_chromium(a.chromium)
    if not binary:
        log(f"no Chromium found (tried {', '.join(CHROMIUMS)}); use --chromium PATH")
        return 3
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if os.environ.get("_SHOTS_CHILD") != "1" or len(todo) > 1:
        rc = 0
        for n in todo:
            r = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--out", str(out),
                 "--only", n, "--chromium", binary],
                env=dict(os.environ, _SHOTS_CHILD="1"), stdout=subprocess.PIPE, text=True)
            sys.stdout.write(r.stdout)
            sys.stdout.flush()
            rc |= r.returncode != 0
        return 1 if rc else 0

    # Identifying strings, captured before make_sandbox replaces HOME.
    ident = [os.path.expanduser("~"), os.environ.get("USER", ""), socket.gethostname()]
    ident = [s for s in ident if s and len(s) > 2 and s != "/"]
    try:
        p = run_scene(todo[0], out, binary, ident)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log(f"FAILED {todo[0]}: {exc}")
        return 1
    log(f"wrote {p}")
    print(p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
