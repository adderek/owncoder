#!/usr/bin/env python3
"""Reproducible SVG screenshots of the Textual TUI.

Runs the real app in-process under Textual's pilot, against a scripted fake LLM
(tests/_mock_llm.py), inside a throw-away HOME + project sandbox.

    shots.py [--out DIR] [--only NAME ...] [--size 120x36] [--list]

Prose goes to stderr, written file paths to stdout. Exit 0 only if every
requested shot was written.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent.parent          # the `agent` package directory
DEFAULT_OUT = HERE.parent / "docs" / "img" / "shots"
SCENES_DIR = HERE / "scenes"
SAMPLE = HERE / "sample_project"
TURN_TIMEOUT = 60.0


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def scene_names() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("*.json"))


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

[ui]
mode = "textual"
round_summary = false
qa_summary_mode = "off"
''', encoding="utf-8")
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


async def drive(app, prompt: str, size: tuple[int, int], out: Path) -> None:
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await pilot.press(*[("space" if c == " " else c) for c in prompt])
        await pilot.press("enter")
        # Turn state: `_chat_worker` is set by _begin_chat and is a textual
        # Worker; `_agent_running` is cleared by _turn_cleanup once it ended.
        deadline = time.monotonic() + TURN_TIMEOUT
        while time.monotonic() < deadline:
            await pilot.pause(0.05)
            w = app._chat_worker
            if w is not None and w.is_finished and not app._agent_running:
                break
        else:
            if os.environ.get("SHOTS_DEBUG"):
                app.save_screenshot(filename="debug.svg", path=str(out.parent))
            raise TimeoutError("turn did not finish")
        # let queued UI events (final render, token bar) settle
        await pilot.pause(0.3)
        await pilot.pause()
        app.save_screenshot(filename=out.name, path=str(out.parent))
    # Textual derives the CSS class prefix from a content hash; pin it so
    # unrelated changes do not churn the file and inline use cannot collide.
    import re
    svg = out.read_text(encoding="utf-8")
    svg = re.sub(r"terminal-\d+", f"terminal-{out.stem}", svg)
    out.write_text(svg, encoding="utf-8")


def run_scene(name: str, out_dir: Path, size: tuple[int, int]) -> Path:
    scene = json.loads((SCENES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(PKG_DIR / "tests"))
    from _mock_llm import MockLLM  # noqa: E402  (stdlib only, no agent import)

    with tempfile.TemporaryDirectory(prefix="owncoder-demo-") as t, \
            MockLLM(scene["scenario"]) as mock:
        tmp = Path(t)
        proj = make_sandbox(tmp, mock.base_url)
        app = build_app(proj)
        target = out_dir / f"{name}.svg"
        asyncio.run(drive(app, scene["prompt"], size, target))
        os.chdir("/")  # leave the sandbox before it is deleted
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--size", default="120x36")
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
        log("--size must be WxH, e.g. 120x36")
        return 2
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # One scene per child process: HOME/cwd/env are process-global and the
    # agent package caches paths at import time.
    if len(todo) > 1 or os.environ.get("_SHOTS_CHILD") != "1":
        import subprocess
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
