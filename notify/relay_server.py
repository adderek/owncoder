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

Abuse limits (per connection / per role):
  - max concurrent connections per role (--max-agents / --max-clients)
  - token-bucket message rate limit (--msg-rate / --msg-burst)
  - max message size (--max-msg-bytes)
  Offenders are closed with code 4429.

Security: the relay only ever forwards opaque JSON between authenticated
parties — it executes nothing. Run it behind TLS (reverse proxy such as
caddy/nginx, wss://) when exposed beyond localhost; the token is sent in-band.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10
CLOSE_UNAUTHORIZED = 4401
CLOSE_TOO_MANY = 4429      # connection cap or rate limit exceeded
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
        self._replay: deque = deque(maxlen=replay_size)
        self._max_agents = max_agents
        self._max_clients = max_clients
        self._msg_rate = msg_rate
        self._msg_burst = msg_burst
        self._max_msg_bytes = max_msg_bytes

    async def handler(self, ws) -> None:
        role = await self._auth(ws)
        if role is None:
            return
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
                await self._route(role, raw)
        except Exception as exc:
            logger.debug("relay: connection ended: %s", exc)
        finally:
            peers.discard(ws)

    async def _auth(self, ws) -> "str | None":
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
        return role

    async def _route(self, sender_role: str, raw: str) -> None:
        if sender_role == "agent":
            self._replay.append(raw)
            targets = self._clients
        else:
            targets = self._agents
        for peer in list(targets):
            try:
                await peer.send(raw)
            except Exception:
                targets.discard(peer)


def _peer(ws) -> str:
    try:
        addr = ws.remote_address
        return f"{addr[0]}:{addr[1]}" if addr else "?"
    except Exception:
        return "?"


def _read_token(path: "str | None", env_var: str) -> str:
    if path:
        return Path(path).expanduser().read_text(encoding="utf-8").strip()
    return os.environ.get(env_var, "")


async def _serve(host: str, port: int, hub: RelayHub, max_size: int) -> None:
    import websockets
    async with websockets.serve(hub.handler, host, port, max_size=max_size):
        logger.info("relay listening on %s:%s", host, port)
        await asyncio.Future()


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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
