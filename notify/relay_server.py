"""Relay server — routes notify messages between agents and clients.

Standalone service for the user's own host:

    python -m agent.notify.relay_server --port 8970 --token-file ~/.config/agent/relay.token

Protocol (JSON, one object per websocket message):
  First message must be a hello:  {"type": "hello", "role": "agent" | "client",
                                   "token": "...", "name": "..."}
  Auth is in-band (token in hello, constant-time compare) — keeps the protocol
  identical for any websocket client (Android app, browser, wscat) and avoids
  header-API differences between websockets versions. Connections failing
  hello within 10s, or with a bad token, are closed with code 4401.

Roles & tokens:
  Single shared token (default): both roles authenticate against it and the
  hello's self-asserted "role" is trusted — simplest, fine on a trusted LAN.
  Per-role tokens (--agent-token-file + --client-token-file, different values):
  the role is decided by *which* token matches, so a holder of the client
  token can no longer impersonate an agent. Always keep the e2e key set as the
  primary defence; per-role tokens are defence-in-depth.

Routing:
  agent  → all clients (notices/questions; also stored in a replay buffer)
  client → all agents (answers; validation happens agent-side in NotifyBroker)
  New clients receive the replay buffer (last N messages) on connect, so a
  reconnecting phone sees recent history.

  Addressed delivery: a frame MAY carry a top-level "to" naming a peer's hello
  "name". When present, the frame is delivered only to opposite-role peers with
  that name (no match → dropped); when absent, it broadcasts as above. This lets
  a "main entry" agent forward curated input to one target agent (e.g. the
  current-project agent) instead of fanning out to every agent. "to" is a clear
  routing header read by the relay; under e2e it must sit OUTSIDE the encrypted
  envelope (alongside "type":"enc"), since the relay cannot read the ciphertext.
  Addressed agent→client frames are not added to the replay buffer (they target
  a specific live client, not every reconnecting one).

Abuse limits (per connection / per role):
  - max concurrent connections per role (--max-agents / --max-clients)
  - token-bucket message rate limit (--msg-rate / --msg-burst)
  - max message size (--max-msg-bytes)
  Offenders are closed with code 4429.

Security: the relay only ever forwards opaque JSON between authenticated
parties — it executes nothing. Do not expose it directly on the WAN: the token
is sent in-band. Run it behind WireGuard (bind to the wg0 address) so the wire
is encrypted and only authenticated peers reach it; e2e protects payloads from
the relay host itself.
"""
from __future__ import annotations

import argparse
import asyncio
import errno
import hmac
import json
import logging
import os
import signal
import time
from collections import deque
from pathlib import Path

from .messages import NOTIFY_PROTOCOL_VERSION

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10
CLOSE_UNAUTHORIZED = 4401
CLOSE_TOO_MANY = 4429      # connection cap or rate limit exceeded
CLOSE_BAD_VERSION = 4426   # incompatible protocol version in hello
DEFAULT_MAX_AGENTS = 8
DEFAULT_MAX_CLIENTS = 16
DEFAULT_MSG_RATE = 20.0    # sustained messages/sec per connection
DEFAULT_MSG_BURST = 40     # bucket capacity
DEFAULT_MAX_MSG_BYTES = 256 * 1024


class _TokenBucket:
    """Per-connection rate limiter. Refills at `rate`/s up to `burst`."""

    __slots__ = ("_tokens", "_rate", "_burst", "_ts")

    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate
        self._burst = float(burst)
        self._tokens = float(burst)
        self._ts = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self._tokens = min(self._burst, self._tokens + (now - self._ts) * self._rate)
        self._ts = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class RelayHub:
    """Routes messages between authenticated agents and clients.

    `token` sets a single shared token for both roles (backward compatible).
    Pass `agent_token` and/or `client_token` to use distinct per-role tokens;
    when they differ, the role is derived from the matching token rather than
    trusting the hello's self-asserted role.
    """

    def __init__(
        self,
        token: "str | None" = None,
        replay_size: int = 100,
        *,
        agent_token: "str | None" = None,
        client_token: "str | None" = None,
        max_agents: int = DEFAULT_MAX_AGENTS,
        max_clients: int = DEFAULT_MAX_CLIENTS,
        msg_rate: float = DEFAULT_MSG_RATE,
        msg_burst: int = DEFAULT_MSG_BURST,
        max_msg_bytes: int = DEFAULT_MAX_MSG_BYTES,
    ) -> None:
        self._agent_token = agent_token or token
        self._client_token = client_token or token
        if not self._agent_token or not self._client_token:
            raise ValueError("relay needs an agent token and a client token (or a shared token)")
        self._agents: set = set()
        self._clients: set = set()
        self._names: dict = {}  # ws -> hello "name" (for addressed routing)
        self._replay: deque = deque(maxlen=replay_size)
        self._roster: dict[str, dict] = {}       # name -> metadata (role, v, since)
        self._presence_version: int = 0          # bumped on every join/leave
        self._max_agents = max_agents
        self._max_clients = max_clients
        self._msg_rate = msg_rate
        self._msg_burst = msg_burst
        self._max_msg_bytes = max_msg_bytes

    async def handler(self, ws) -> None:
        auth = await self._auth(ws)
        if auth is None:
            return
        role, name, peer_v = auth
        peers, cap = (
            (self._agents, self._max_agents) if role == "agent"
            else (self._clients, self._max_clients)
        )
        if len(peers) >= cap:
            logger.warning("relay: %s connection cap (%d) reached — rejecting %s",
                           role, cap, _peer(ws))
            await ws.close(CLOSE_TOO_MANY, "too many connections")
            return
        peers.add(ws)
        if name:
            self._names[ws] = name
            self._roster[name] = {"role": role, "v": peer_v, "since": time.time()}
            self._presence_version += 1
            self._broadcast_presence()
        bucket = _TokenBucket(self._msg_rate, self._msg_burst)
        try:
            if role == "client":
                for raw in list(self._replay):
                    await ws.send(raw)
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue  # protocol is text-only
                if len(raw) > self._max_msg_bytes:
                    await ws.close(CLOSE_TOO_MANY, "message too large")
                    break
                if not bucket.allow():
                    logger.warning("relay: rate limit — closing %s %s", role, _peer(ws))
                    await ws.close(CLOSE_TOO_MANY, "rate limit")
                    break
                await self._route(role, ws, raw)
        except Exception as exc:
            logger.debug("relay: connection ended: %s", exc)
        finally:
            peers.discard(ws)
            name = self._names.pop(ws, None)
            if name is not None and self._roster.pop(name, None) is not None:
                self._presence_version += 1
                self._broadcast_presence()

    async def _auth(self, ws) -> "tuple[str, str, object] | None":
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=HELLO_TIMEOUT_S)
            hello = json.loads(raw)
        except Exception:
            await ws.close(CLOSE_UNAUTHORIZED, "hello expected")
            return None
        token = hello.get("token", "") if isinstance(hello, dict) else ""
        if (
            not isinstance(hello, dict)
            or hello.get("type") != "hello"
            or not isinstance(token, str)
        ):
            await ws.close(CLOSE_UNAUTHORIZED, "unauthorized")
            return None
        # Version negotiation: a hello MAY carry "v". Missing = legacy client,
        # accepted for backward compat. Present but a different major = reject.
        peer_v = hello.get("v")
        if peer_v is not None and peer_v != NOTIFY_PROTOCOL_VERSION:
            logger.warning("relay: protocol version %r != %d from %s",
                           peer_v, NOTIFY_PROTOCOL_VERSION, _peer(ws))
            await ws.close(CLOSE_BAD_VERSION,
                           f"protocol version mismatch (server {NOTIFY_PROTOCOL_VERSION})")
            return None
        # Always evaluate both compares (constant-time, no short-circuit leak).
        agent_ok = hmac.compare_digest(token, self._agent_token)
        client_ok = hmac.compare_digest(token, self._client_token)
        claimed = "agent" if hello.get("role") == "agent" else "client"
        if agent_ok and client_ok:
            role = claimed            # shared-token mode: trust self-asserted role
        elif agent_ok:
            role = "agent"            # per-role: role follows the matching token
        elif client_ok:
            role = "client"
        else:
            logger.warning("relay: bad token from %s", _peer(ws))
            await ws.close(CLOSE_UNAUTHORIZED, "unauthorized")
            return None
        raw_name = hello.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        return role, name, peer_v

    def _broadcast_presence(self) -> None:
        """Send current roster to all connected peers as a presence frame."""
        raw = json.dumps({
            "type": "presence",
            "v": self._presence_version,
            "peers": {
                name: meta
                for name, meta in self._roster.items()
            },
        })
        # Presence / roster is a project-discovery frame consumed by the router,
        # which connects as a CLIENT (MULTI_PROJECT_PLAN §4.4). So it must reach
        # all peers (agents + clients). Consumers that don't understand the
        # frame type are expected to skip it (versioned protocol), not crash.
        for ws in list(self._agents | self._clients):
            try:
                # fire-and-forget: schedule, don't await (called from sync contexts)
                asyncio.ensure_future(ws.send(raw))
            except Exception:
                pass

    async def _route(self, sender_role: str, sender_ws, raw: str) -> None:
        to = _routing_to(raw)
        if to is None:
            # Broadcast by role: agent → clients, client → agents.
            targets = self._clients if sender_role == "agent" else self._agents
            if sender_role == "agent":
                self._replay.append(raw)  # only broadcast agent frames are replayed
        else:
            # Addressed: deliver to any peer (either role) with the matching name,
            # except the sender. Enables agent→agent delegation as well as
            # client→named-agent and agent→named-client. Not replayed (targets one
            # live peer, not every reconnecting client).
            targets = [
                p for p in (*self._agents, *self._clients)
                if p is not sender_ws and self._names.get(p) == to
            ]
        for peer in list(targets):
            try:
                await peer.send(raw)
            except Exception:
                # Drop a dead peer from whichever pool holds it (targets may be a
                # transient list in the addressed case).
                self._agents.discard(peer)
                self._clients.discard(peer)
                dead_name = self._names.pop(peer, None)
                if dead_name is not None and self._roster.pop(dead_name, None) is not None:
                    self._presence_version += 1
                    self._broadcast_presence()


def _routing_to(raw: str) -> "str | None":
    """Clear-text "to" routing header, or None for broadcast.

    Read without trusting the rest of the frame: a malformed frame or a non-str
    "to" falls back to broadcast (the relay forwards opaque JSON either way).
    Under e2e this reads the envelope's top-level "to" — never the ciphertext.
    """
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    to = data.get("to")
    return to if isinstance(to, str) and to else None


def _peer(ws) -> str:
    try:
        addr = ws.remote_address
        return f"{addr[0]}:{addr[1]}" if addr else "?"
    except Exception:
        return "?"


# ── pidfile (shared-relay discovery / stop) ─────────────────────────────────
# The relay owns its own pidfile: written once it is bound, removed on clean
# exit. A spawned relay outlives the agent that started it, so the pidfile is
# how a later agent (or the user) finds and stops the running instance.

def pidfile_path(port: int) -> Path:
    base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    return base / f"owncoder-relay-{port}.pid"


def read_pid(port: int) -> "int | None":
    """PID of a live relay on `port` per its pidfile, or None.

    Returns None for a missing, unreadable, or stale pidfile (process gone) and
    cleans up a stale file so it cannot mislead a later reuse/stop.
    """
    path = pidfile_path(port)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if _pid_alive(pid):
        return pid
    try:
        path.unlink()  # stale — clear it
    except OSError:
        pass
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM  # exists but not ours to signal
    return True


def stop_relay(port: int, *, timeout: float = 3.0) -> bool:
    """SIGTERM the relay recorded for `port`. True if one was running and exited.

    Best-effort escalation to SIGKILL if it does not exit within `timeout`.
    """
    pid = read_pid(port)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        pidfile_path(port).unlink()
    except OSError:
        pass
    return True


def _write_pidfile(port: int) -> "Path | None":
    path = pidfile_path(port)
    try:
        path.write_text(str(os.getpid()), encoding="utf-8")
        return path
    except OSError as exc:
        logger.warning("relay: could not write pidfile %s: %s", path, exc)
        return None


def _read_token(path: "str | None", env_var: str) -> str:
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    return os.environ.get(env_var, "")


async def _serve(host: str, port: int, hub: RelayHub, max_size: int) -> None:
    import websockets
    async with websockets.serve(hub.handler, host, port, max_size=max_size):
        logger.info("relay listening on %s:%s", host, port)
        pidfile = _write_pidfile(port)
        try:
            await asyncio.Future()
        finally:
            if pidfile is not None:
                try:
                    pidfile.unlink()
                except OSError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="owncoder notify relay server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8970)
    parser.add_argument("--token-file", help="shared token file (both roles)")
    parser.add_argument("--agent-token-file", help="per-role token file for agents")
    parser.add_argument("--client-token-file", help="per-role token file for clients")
    parser.add_argument("--replay", type=int, default=100, help="messages replayed to new clients")
    parser.add_argument("--max-agents", type=int, default=DEFAULT_MAX_AGENTS)
    parser.add_argument("--max-clients", type=int, default=DEFAULT_MAX_CLIENTS)
    parser.add_argument("--msg-rate", type=float, default=DEFAULT_MSG_RATE,
                        help="sustained messages/sec per connection")
    parser.add_argument("--msg-burst", type=int, default=DEFAULT_MSG_BURST,
                        help="rate-limit bucket capacity")
    parser.add_argument("--max-msg-bytes", type=int, default=DEFAULT_MAX_MSG_BYTES)
    parser.add_argument("--stop", action="store_true",
                        help="stop a relay running on --port (via its pidfile) and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.stop:
        if stop_relay(args.port):
            logger.info("relay on port %d stopped", args.port)
        else:
            logger.info("no relay running on port %d", args.port)
        return

    shared = _read_token(args.token_file, "AGENT_RELAY_TOKEN")
    agent_token = _read_token(args.agent_token_file, "AGENT_RELAY_AGENT_TOKEN") or shared
    client_token = _read_token(args.client_token_file, "AGENT_RELAY_CLIENT_TOKEN") or shared
    if not agent_token or not client_token:
        parser.error(
            "no token: pass --token-file (shared) or both --agent-token-file and "
            "--client-token-file (or set AGENT_RELAY_TOKEN / *_AGENT_TOKEN / *_CLIENT_TOKEN)"
        )
    if agent_token == client_token:
        logger.info("relay: single shared token (self-asserted roles)")
    else:
        logger.info("relay: per-role tokens (role derived from matching token)")

    hub = RelayHub(
        replay_size=args.replay,
        agent_token=agent_token,
        client_token=client_token,
        max_agents=args.max_agents,
        max_clients=args.max_clients,
        msg_rate=args.msg_rate,
        msg_burst=args.msg_burst,
        max_msg_bytes=args.max_msg_bytes,
    )
    try:
        asyncio.run(_serve(args.host, args.port, hub, args.max_msg_bytes))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
