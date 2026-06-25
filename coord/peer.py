"""Peer delegation — send a request to another named agent over the relay.

A frame addressed with a top-level "to" is routed by the relay to the peer whose
hello "name" matches (relay_server._route), including agent→agent. We reuse the
existing control protocol: a delegated request is a `chat` control frame, so the
receiving agent starts a turn for it (build_ui_server wires on_chat → the same
prompt path as a local/voice prompt).

The link is owned by the ui_server layer (RelayLink, created in
build_ui_server). It is registered here at startup so a plain tool function can
reach it without threading the agent through every call. Fail-soft: when remote
streaming is disabled (no link), delegation reports unavailable rather than
raising — the agent simply has no peers to talk to.

Scope: send-only (fire-and-forget). The peer's response comes back as its own
relay event stream, not as this call's return value; request/response
correlation is future work.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_link: Any | None = None


def set_link(link: Any | None) -> None:
    """Register (or clear) the RelayLink used to reach peer agents."""
    global _link
    _link = link


def has_link() -> bool:
    return _link is not None


def send_to_peer(name: str, text: str) -> dict:
    """Send `text` to the agent registered under hello-name `name`.

    Returns a small status dict (never raises): the caller is a tool whose
    result is shown to the model.
    """
    if not name or not text:
        return {"ok": False, "error": "both 'agent' and 'request' are required"}
    if _link is None:
        return {
            "ok": False,
            "error": "delegation unavailable: remote relay not configured "
                     "(set ui_server.remote + relay_url)",
        }
    from agent.ui_server.control_frames import build_control

    frame = build_control("chat", text=text)
    try:
        _link.send_frame(frame, to=name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("peer: send_to_peer failed")
        return {"ok": False, "error": f"send failed: {exc}"}
    return {"ok": True, "to": name, "sent": text}
