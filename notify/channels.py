"""Notification channel implementations.

Capability tiers (what the endpoint can send back):
  display — nothing (read-only)
  choices — pick one of the offered options
  chat    — options + free text

CommandChannel: outbound only, pipe to shell command stdin.
RelayChannel: bidirectional websocket to a relay server (relay_server.py) —
incoming answers go to the broker via the on_answer callback.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from agent.config.models import NotifyChannelConfig
from agent.notify.messages import Notice, Question

logger = logging.getLogger(__name__)

CAPABILITIES = ("display", "choices", "chat")
SEND_TIMEOUT_S = 10
RELAY_QUEUE_MAX = 100
RELAY_BACKOFF_MAX_S = 60


@runtime_checkable
class Channel(Protocol):
    name: str
    capability: str

    async def send(self, msg: "Notice | Question") -> bool:
        """Deliver message. Returns False on failure (logged, never raises)."""
        ...


class CommandChannel:
    """Pipe each message to a shell command's stdin.

    Covers ntfy/signal-cli/IRC bots via small adapter commands. The message
    is written to stdin — never interpolated into the command line — so
    message content cannot inject shell syntax. The command itself comes
    from the user's config and runs with the user's own privileges.
    """

    capability = "display"  # outbound only; cannot return answers

    def __init__(self, cfg: NotifyChannelConfig) -> None:
        self.cmd = cfg.cmd
        self.format = cfg.format
        self.name = cfg.name or f"command({cfg.cmd.split()[0] if cfg.cmd else '?'})"

    async def send(self, msg: "Notice | Question") -> bool:
        payload = json.dumps(msg.to_wire()) if self.format == "json" else msg.render_text()
        try:
            proc = await asyncio.create_subprocess_shell(
                self.cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(
                    proc.communicate((payload + "\n").encode()), timeout=SEND_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                proc.kill()
                logger.warning("notify channel %s: send timed out", self.name)
                return False
            if proc.returncode != 0:
                logger.warning(
                    "notify channel %s: exit %s: %s",
                    self.name, proc.returncode, (stderr or b"").decode(errors="replace")[:200],
                )
                return False
            return True
        except OSError as exc:
            logger.warning("notify channel %s: %s", self.name, exc)
            return False


class RelayChannel:
    """Persistent websocket to a relay server (see relay_server.py).

    Outbound-only connection from the agent's machine (no listening port).
    Messages queue locally and a background task pumps them out, reconnecting
    with exponential backoff; the same task feeds incoming answer messages to
    on_answer (NotifyBroker validation). Queue overflow drops the oldest
    message with a warning — notifications are best-effort, never backpressure
    on the agent loop.
    """

    def __init__(self, cfg: NotifyChannelConfig, token: str, on_answer=None, e2e=None) -> None:
        self.url = cfg.url
        self.capability = cfg.capability
        self.name = cfg.name or f"relay({cfg.url})"
        self._token = token
        self._on_answer = on_answer
        self._e2e = e2e  # E2EBox | None; when set, payloads encrypted both ways
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=RELAY_QUEUE_MAX)
        self._task: asyncio.Task | None = None

    async def send(self, msg: "Notice | Question") -> bool:
        self._ensure_task()
        wire_dict = msg.to_wire()
        if self._e2e is not None:
            wire_dict = self._e2e.encrypt(wire_dict)
        wire = json.dumps(wire_dict)
        while True:
            try:
                self._queue.put_nowait(wire)
                return True
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                    logger.warning("notify channel %s: queue full — dropped oldest message", self.name)
                except asyncio.QueueEmpty:
                    pass

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        import websockets
        backoff = 1
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    await ws.send(json.dumps({
                        "type": "hello", "role": "agent",
                        "token": self._token, "name": self.name,
                    }))
                    backoff = 1
                    pumps = (
                        asyncio.create_task(self._pump_out(ws)),
                        asyncio.create_task(self._pump_in(ws)),
                    )
                    try:
                        done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            t.result()  # surface the pump's exception to the reconnect loop
                    finally:
                        # Both pumps must die before reconnect — a surviving
                        # _pump_out would become a duplicate queue consumer.
                        for t in pumps:
                            t.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("notify channel %s: %s — reconnecting in %ss", self.name, exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RELAY_BACKOFF_MAX_S)

    async def _pump_out(self, ws) -> None:
        while True:
            await ws.send(await self._queue.get())

    async def _pump_in(self, ws) -> None:
        async for raw in ws:
            if self._on_answer is None or isinstance(raw, bytes):
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            if self._e2e is not None:
                # E2E mode: only encrypted envelopes accepted — a plaintext
                # message here is a downgrade attempt or misconfigured client.
                if data.get("type") != "enc":
                    logger.warning("notify channel %s: plaintext message dropped (e2e enabled)", self.name)
                    continue
                inner = self._e2e.decrypt(data)
                if inner is None:
                    logger.warning("notify channel %s: undecryptable envelope dropped", self.name)
                    continue
                data = inner
            if data.get("type") == "answer":
                self._on_answer(data)


def _read_relay_token(cfg: NotifyChannelConfig) -> "str | None":
    if not cfg.token_file:
        return None
    try:
        return Path(cfg.token_file).expanduser().read_text(encoding="utf-8").strip() or None
    except OSError as exc:
        logger.warning("notify: cannot read token_file %s: %s", cfg.token_file, exc)
        return None


def build_channel(
    cfg: NotifyChannelConfig,
    on_answer: "Callable[[dict], None] | None" = None,
) -> "Channel | None":
    """Build channel from config entry. Returns None (with log) on bad config —
    a misconfigured channel must not prevent agent startup."""
    if cfg.capability not in CAPABILITIES:
        logger.warning("notify: unknown capability %r — skipping channel", cfg.capability)
        return None
    if cfg.type == "command":
        if not cfg.cmd:
            logger.warning("notify: command channel without cmd — skipping")
            return None
        if cfg.capability != "display":
            logger.warning(
                "notify: command channel is outbound-only; capability %r downgraded to display",
                cfg.capability,
            )
        return CommandChannel(cfg)
    if cfg.type == "relay":
        if not cfg.url:
            logger.warning("notify: relay channel without url — skipping")
            return None
        token = _read_relay_token(cfg)
        if token is None:
            logger.warning("notify: relay channel needs a readable token_file — skipping")
            return None
        try:
            import websockets  # noqa: F401
        except ImportError:
            logger.warning("notify: websockets not installed (pip install 'local-code-agent[notify]') — skipping relay")
            return None
        e2e = None
        if cfg.e2e_key_file:
            from agent.notify.crypto import load_box
            e2e = load_box(cfg.e2e_key_file)
            if e2e is None:
                # Fail closed: e2e was requested — never fall back to plaintext.
                logger.warning("notify: e2e key unavailable — skipping relay channel %s", cfg.url)
                return None
        return RelayChannel(cfg, token, on_answer, e2e=e2e)
    logger.warning("notify: unknown channel type %r — skipping", cfg.type)
    return None
