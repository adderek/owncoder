"""agent.ui_server — abstraction layer between user interfaces and the backend.

Phase 1: LocalUIServer wraps Agent in-process.
Later: transport-backed UIServer enabling remote/multi-UI/multi-agent access.

build_ui_server() returns a LocalUIServer, optionally wrapped in a RemoteBridge
that publishes the session's event stream to a relay and accepts control frames
(config.ui_server.remote). The local TUI keeps rendering in parallel.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .protocol import UIServerProtocol
from .local import LocalUIServer

if TYPE_CHECKING:
    from agent.core.agent import Agent

logger = logging.getLogger(__name__)

__all__ = ["UIServerProtocol", "LocalUIServer", "build_ui_server"]


def build_ui_server(agent: "Agent") -> Any:
    """LocalUIServer, wrapped for remote streaming when configured.

    Falls back to a plain LocalUIServer on any misconfiguration (missing url /
    token / e2e key) — a broken remote setup must never block the local UI.
    """
    inner = LocalUIServer(agent)
    cfg = getattr(agent.config, "ui_server", None)
    if cfg is None or not getattr(cfg, "remote", False):
        return inner
    if not cfg.relay_url:
        logger.warning("ui_server.remote set but relay_url empty — local only")
        return inner

    token = ""
    if cfg.relay_token_file:
        try:
            from pathlib import Path
            token = Path(cfg.relay_token_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("ui_server: cannot read relay_token_file %s: %s — local only",
                           cfg.relay_token_file, exc)
            return inner
    if not token:
        logger.warning("ui_server.remote needs a readable relay_token_file — local only")
        return inner

    e2e = None
    if cfg.e2e_key_file:
        from agent.notify.crypto import load_box
        e2e = load_box(cfg.e2e_key_file)
        if e2e is None:
            # Fail closed: e2e requested but key unavailable — do not stream plaintext.
            logger.warning("ui_server: e2e key unavailable — remote UI disabled (local only)")
            return inner

    try:
        from .relay_link import RelayLink
        from .remote_bridge import RemoteBridge
        from .control_frames import ControlDispatcher
    except ImportError as exc:
        logger.warning("ui_server: remote deps unavailable (%s) — local only", exc)
        return inner

    # Inbound control (inject/stop/answer/set) applies to the inner server.
    # Remote-initiated chat is deferred — the local input loop owns turn start.
    dispatcher = ControlDispatcher(inner)
    link = RelayLink(cfg.relay_url, token, name=cfg.name,
                     on_frame=dispatcher.handle, e2e=e2e)
    logger.info("ui_server: remote streaming to %s", cfg.relay_url)
    return RemoteBridge(inner, link.send_frame)
