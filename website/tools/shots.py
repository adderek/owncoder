#!/usr/bin/env python3
"""Reproducible SVG screenshots of the Textual TUI.

Runs the real app in-process under Textual's pilot, against a scripted fake LLM
(tests/_mock_llm.py), inside a throw-away HOME + project sandbox.

    shots.py [--out DIR] [--only NAME ...] [--size 100x30] [--list]

Prose goes to stderr, written file paths to stdout. Exit 0 only if every
requested shot was written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent.parent          # the `agent` package directory
DEFAULT_OUT = HERE.parent / "docs" / "img" / "shots"
SCENES_DIR = HERE / "scenes"
SAMPLE = HERE / "sample_project"
TURN_TIMEOUT = 60.0
TITLE = "owncoder"
MARKER = ".owncoder-shots"
GIT_DATE = "2026-01-01T00:00:00Z"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def scene_names() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("*.json"))


class FixedSandbox:
    """Constant-path sandbox /tmp/owncoder-demo-<uid> (the path reaches the
    system prompt, so a random name would change token counts between runs).

    Refuses to touch an existing dir unless it carries our marker and its
    recorded pid is dead; removed on exit.
    """

    def __init__(self) -> None:
        self.path = Path("/tmp") / f"owncoder-demo-{os.getuid()}"

    def __enter__(self) -> Path:
        p = self.path
        if p.exists() or p.is_symlink():
            mk = p / MARKER
            pid = 0
            if not p.is_symlink() and mk.is_file():
                try:
                    pid = int(mk.read_text().strip() or 0)
                except ValueError:
                    pid = 0
                alive = False
                if pid:
                    try:
                        os.kill(pid, 0)
                        alive = True
                    except OSError:
                        pass
                if alive:
                    raise RuntimeError(f"{p} in use by pid {pid}")
                shutil.rmtree(p)
            else:
                raise RuntimeError(f"{p} exists and is not ours; refusing to touch it")
        p.mkdir(mode=0o700)
        (p / MARKER).write_text(str(os.getpid()))
        return p

    def __exit__(self, *exc) -> None:
        os.chdir("/")
        shutil.rmtree(self.path, ignore_errors=True)


def git_init(proj: Path) -> None:
    """Make the sample project a one-commit git repo with a fixed identity/date."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
               GIT_AUTHOR_DATE=GIT_DATE, GIT_COMMITTER_DATE=GIT_DATE)

    def g(*a: str) -> None:
        subprocess.run(["git", "-C", str(proj), *a], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    g("init", "-q", "-b", "main")
    g("config", "user.name", "demo")
    g("config", "user.email", "demo@example.invalid")
    g("config", "commit.gpgsign", "false")
    (proj / ".gitignore").write_text(".agent/\n", encoding="utf-8")
    g("add", "-A")
    g("commit", "-q", "-m", "Initial commit")


def make_sandbox(tmp: Path, base_url: str) -> Path:
    """Create HOME + project under tmp and point the process at them.

    Must run before anything from `agent` is imported.
    """
    home = tmp / "home"
    cfg_home = home / ".config"
    cfg_home.mkdir(parents=True)
    proj = tmp / "demo-project"
    shutil.copytree(SAMPLE, proj)
    (proj / ".agent").mkdir()
    (proj / ".agent" / ".configured").touch()
    (proj / "agent.toml").write_text(
        f'''[models]
default = "mock"

[models.mock]
base_url = "{base_url}"
api_key = "mock"
model = "mock-model"
ctx_window = 32768
max_output_tokens = 4096

[failover]
enabled = false

[agent]
startup_profile = "off"
auto_name_sessions = false
narration_fallback = false

[housekeeping]
enabled = false

[compile_prompts]
enabled = false
auto_spawn = false

[rag]
embed_server_autostart = "off"

[[permissions.rules]]
tool = "run_argv"
match = "make*"
verdict = "ask"
reason = "builds run project code"

[ui]
mode = "textual"
round_summary = false
qa_summary_mode = "off"
''', encoding="utf-8")
    os.environ["GIT_CONFIG_GLOBAL"] = "/dev/null"
    os.environ["GIT_CONFIG_SYSTEM"] = "/dev/null"
    git_init(proj)
    os.environ["HOME"] = str(home)
    os.environ["XDG_CONFIG_HOME"] = str(cfg_home)
    os.environ["XDG_DATA_HOME"] = str(home / ".local" / "share")
    os.environ["XDG_STATE_HOME"] = str(home / ".local" / "state")
    os.environ["XDG_CACHE_HOME"] = str(home / ".cache")
    os.environ["AGENT_UI_MODE"] = "textual"
    os.environ["AGENT_SECURITY_BACKEND"] = "none"
    os.environ["AGENT_SECURITY_REQUIRE_SANDBOX"] = "false"
    os.environ["TERM"] = "xterm-256color"
    for k in [k for k in os.environ if k.startswith("AGENT_LLM_")
              or k.startswith("AGENT_EMBEDDINGS_")]:
        del os.environ[k]
    os.chdir(proj)
    # Isolation guard: the resolved user config dir must be inside the sandbox.
    resolved = (Path.home() / ".config" / "agent").resolve()
    assert tmp.resolve() in resolved.parents, f"config dir escapes sandbox: {resolved}"
    return proj


def build_app(proj: Path):
    sys.path.insert(0, str(PKG_DIR.parent))
    from agent.config import load_config
    from agent.core.agent import Agent
    from agent.data_provider import LocalDataProvider
    from agent.memory.session import new_session
    from agent.ui.terminal import _build_textual_app

    cfg = load_config([proj / "agent.toml"])
    cfg.tools.working_dir = str(proj)
    cfg.ui.mode = "textual"
    agent = Agent(cfg, data_provider=LocalDataProvider(
        store=None, embedder=None, asm_store=None, config=cfg))
    session = new_session(short_name="demo", mode="incognito")
    agent.session = session
    agent.set_session_mode(session.mode)
    return _build_textual_app(agent, session=session)


def _title_and_normalise(svg: str, stem: str) -> str:
    """Make the SVG byte-stable. Regexes (all applied to the raw SVG text):

    * `terminal-<digits>` CSS prefix (content hash) -> `terminal-<scene>`.
    * status-line rates `<n> in-tok/s` / `<n> out-tok/s` -> fixed plausible
      values (the fake LLM answers in a few ms, real timing is meaningless).
    * plan id `\d{8}T\d{6}\.\d{3}Z_[0-9a-f]{4}` -> fixed id.
    * token counts vary with timing-dependent request bodies; they are only
      normalised if still unstable (see TOKEN_PATTERNS).
    """
    svg = re.sub(r"terminal-\d+", f"terminal-{stem}", svg)
    svg = re.sub(r"\d+(?:\.\d+)?(?:&#160;|\s)*in-tok/s", f"{FAKE_IN_RATE}&#160;in-tok/s", svg)
    svg = re.sub(r"\d+(?:\.\d+)?(?:&#160;|\s)*out-tok/s", f"{FAKE_OUT_RATE}&#160;out-tok/s", svg)
    # plan ids embed wall-clock time + random hex: 20261114T205224.196Z_42ab
    svg = re.sub(r"\d{8}T\d{6}\.\d{3}Z_[0-9a-f]{4}", "20260101T000000.000Z_0001", svg)
    return svg


FAKE_IN_RATE = 1480
FAKE_OUT_RATE = 42


async def _wait_turn(app, deadline: float) -> bool:
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        w = app._chat_worker
        if w is not None and w.is_finished and not app._agent_running:
            return True
    return False


async def drive(app, scene: dict, size: tuple[int, int], out: Path) -> None:
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press(*[("space" if c == " " else c) for c in scene["prompt"]])
        await pilot.press("enter")
        deadline = time.monotonic() + TURN_TIMEOUT
        ask = scene.get("ask")

        async def snap() -> str:
            # cursor blink is wall-clock dependent -> pin it on (byte stability)
            for w in app.query("Input, TextArea"):
                if hasattr(w, "cursor_blink"):
                    w.cursor_blink = False
                if hasattr(w, "_pause_blink"):
                    w._pause_blink(visible=w.has_focus)
            await pilot.pause(0.2)
            return app.export_screenshot(title=TITLE)

        svg = None
        if ask:
            # screenshot while the permission modal is up, then deny it
            while time.monotonic() < deadline:
                await pilot.pause(0.05)
                if type(app.screen).__name__ == "PermissionScreen":
                    break
            else:
                raise TimeoutError("permission prompt never appeared")
            await pilot.pause(0.3)
            svg = await snap()
            await pilot.press("escape")
        if not await _wait_turn(app, deadline):
            raise TimeoutError("turn did not finish")
        await pilot.pause(0.3)
        await pilot.pause()
        for key in scene.get("keys", []):
            await pilot.press(key)
            await pilot.pause(0.3)
        if svg is None:
            svg = await snap()
    svg = _title_and_normalise(svg, out.stem)
    missing = [e for e in _expect_list(scene) if e not in svg.replace("&#160;", " ")]
    if missing:
        raise AssertionError(f"expected text missing from {out.name}: {missing}")
    out.write_text(svg, encoding="utf-8")


def _expect_list(scene: dict) -> list[str]:
    e = scene.get("expect") or []
    return [e] if isinstance(e, str) else list(e)


def run_scene(name: str, out_dir: Path, size: tuple[int, int]) -> Path:
    scene = json.loads((SCENES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(PKG_DIR / "tests"))
    from _mock_llm import MockLLM  # noqa: E402  (stdlib only, no agent import)

    with FixedSandbox() as tmp, MockLLM(scene["scenario"]) as mock:
        proj = make_sandbox(tmp, mock.base_url)
        app = build_app(proj)
        target = out_dir / f"{name}.svg"
        asyncio.run(drive(app, scene, size, target))
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

    # One scene per child process: HOME/cwd/env are process-global and the
    # agent package caches paths at import time.
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
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # the app writes OSC title sequences to stdout
    try:
        p = run_scene(n, out, (w, h))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log(f"FAILED {n}: {exc}")
        return 1
    sys.stdout = real_stdout
    log(f"wrote {p}")
    print(p, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
