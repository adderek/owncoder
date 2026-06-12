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

Routing:
  agent  → all clients (notices/questions; also stored in a replay buffer)
  client → all agents (answers; validation happens agent-side in NotifyBroker)
  New clients receive the replay buffer (last N messages) on connect, so a
  reconnecting phone sees recent history.

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
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

HELLO_TIMEOUT_S = 10
CLOSE_UNAUTHORIZED = 4401


class RelayHub:
    def __init__(self, token: str, replay_size: int = 100) -> None:
        if not token:
            raise ValueError("relay token must not be empty")
        self._token = token
        self._agents: set = set()
        self._clients: set = set()
        self._replay: deque = deque(maxlen=replay_size)

    async def handler(self, ws) -> None:
        role = await self._auth(ws)
        if role is None:
            return
        peers = self._agents if role == "agent" else self._clients
        peers.add(ws)
        try:
            if role == "client":
                for raw in list(self._replay):
                    await ws.send(raw)
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue  # protocol is text-only
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
            or not hmac.compare_digest(token, self._token)
        ):
            await ws.close(CLOSE_UNAUTHORIZED, "unauthorized")
            return None
        return "agent" if hello.get("role") == "agent" else "client"

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


def _read_token(args: argparse.Namespace) -> str:
    if args.token_file:
        return Path(args.token_file).expanduser().read_text(encoding="utf-8").strip()
    return os.environ.get("AGENT_RELAY_TOKEN", "")


async def _serve(host: str, port: int, hub: RelayHub) -> None:
    import websockets
    async with websockets.serve(hub.handler, host, port):
        logger.info("relay listening on %s:%s", host, port)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="owncoder notify relay server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8970)
    parser.add_argument("--token-file", help="file containing the shared auth token")
    parser.add_argument("--replay", type=int, default=100, help="messages replayed to new clients")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = _read_token(args)
    if not token:
        parser.error("no token: pass --token-file or set AGENT_RELAY_TOKEN")

    try:
        asyncio.run(_serve(args.host, args.port, RelayHub(token, args.replay)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
