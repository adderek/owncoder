"""Manage a local embeddings server via an external launcher script.

The agent does not bundle an inference server — serving models is a separate
concern (a sub-project like ollama-turboquant). This module just launches and
supervises that script so the user gets one obvious knob:

    agent embed --start [--gpu|--cpu]
    agent embed --stop
    agent embed --status

Config (RAGConfig): ``rag.embed_server_command`` — path to the launcher
script, invoked with a single argument "cpu" or "gpu"; it must start an
OpenAI-compatible embeddings server on ``config.embeddings.base_url``.
``rag.embed_server_device`` — default device when no flag is given.

The server is machine-global (one per host, not per project), so the pidfile
and log live under ~/.cache/agent/.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("~/.cache/agent").expanduser()
PID_FILE = _CACHE_DIR / "embed-server.pid"
LOG_FILE = _CACHE_DIR / "embed-server.log"

# How long to poll for the server to answer after spawning (model load time).
START_WAIT_S = 30


def _probe(base_url: str) -> bool:
    from agent.config.loader import _probe_models
    return _probe_models(base_url, "local", timeout=2) is not None


def _resolve_embed_target(config: "Config") -> tuple[str, str]:
    """Resolve the LOCAL embeddings endpoint this launcher serves → (base_url, model).

    config.embeddings.base_url is only bridged from the embeddings role pool by
    check_reachability(); the standalone `agent embed` command runs without
    that, so walk the role/pool ourselves and prefer the first local entry —
    that is the one a local launcher can bring up.
    """
    from agent.config.loader import entry_tier

    names: list[str] = []
    pin = config.model_roles.get("embeddings")
    if pin:
        names.append(pin)
    names += [n for n in config.model_pools.get("embeddings", []) if n not in names]
    entries = [e for e in (config.model_entries.get(n) for n in names) if e is not None]
    for entry in entries:
        if entry_tier(entry) == "local":
            return entry.base_url, entry.model
    if entries:
        return entries[0].base_url, entries[0].model
    return config.embeddings.base_url, config.embeddings.model


def embed_base_url(config: "Config") -> str:
    return _resolve_embed_target(config)[0]


def _read_pid() -> int | None:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return None
    return pid


def start(
    config: "Config",
    device: str | None = None,
    probe: Callable[[str], bool] | None = None,
    wait_s: float = START_WAIT_S,
) -> str:
    """Spawn the launcher script detached; wait until the endpoint answers."""
    probe = probe or _probe
    cmd = (config.rag.embed_server_command or "").strip()
    if not cmd:
        return (
            "no launcher configured — set rag.embed_server_command to a script "
            'that starts an embeddings server (invoked with "cpu" or "gpu")'
        )
    script = Path(cmd).expanduser()
    if not script.exists():
        return f"launcher not found: {script}"

    base_url = embed_base_url(config)
    if probe(base_url):
        return f"already serving at {base_url}"
    if _read_pid() is not None:
        return (
            f"launcher already running (pid {_read_pid()}) but {base_url} not "
            f"answering yet — model may still be loading, see {LOG_FILE}"
        )

    device = (device or config.rag.embed_server_device or "cpu").lower()
    if device not in ("cpu", "gpu"):
        return f"unknown device {device!r} (use cpu or gpu)"

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_FILE, "ab")
    try:
        proc = subprocess.Popen(
            [str(script), device],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(script.parent),
            start_new_session=True,  # own process group → clean stop, survives agent exit
        )
    finally:
        log.close()
    PID_FILE.write_text(str(proc.pid))
    logger.info("embed server spawned: %s %s (pid %d)", script, device, proc.pid)

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            return (
                f"launcher exited immediately (rc={proc.returncode}) — see {LOG_FILE}"
            )
        if probe(base_url):
            return f"embeddings server up: {base_url} [{device}] (pid {proc.pid}, log {LOG_FILE})"
        time.sleep(1.0)
    return (
        f"spawned (pid {proc.pid}, {device}) but {base_url} not answering after "
        f"{int(wait_s)}s — model may still be loading; check `agent embed --status` / {LOG_FILE}"
    )


def stop(config: "Config") -> str:
    pid = _read_pid()
    if pid is None:
        PID_FILE.unlink(missing_ok=True)
        return "not running (no live pid)"
    try:
        # start_new_session=True made the launcher a process-group leader; kill
        # the whole group so llama-server children die with the wrapper script.
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    for _ in range(20):
        if _read_pid() is None:
            break
        time.sleep(0.25)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    PID_FILE.unlink(missing_ok=True)
    return f"stopped (pid {pid})"


def status(config: "Config", probe: Callable[[str], bool] | None = None) -> str:
    probe = probe or _probe
    base_url, model = _resolve_embed_target(config)
    pid = _read_pid()
    up = probe(base_url)
    if up:
        who = f"pid {pid}" if pid else "externally started"
        return f"up: {base_url} ({who}, model {model})"
    if pid:
        return f"launcher alive (pid {pid}) but {base_url} not answering — loading or wedged, see {LOG_FILE}"
    return f"down: {base_url} not answering (start: agent embed --start)"
