"""Remote control authorization — a client token is not a licence for `set`."""
from __future__ import annotations

import asyncio

from agent.config.models import UIServerConfig
from agent.ui_server.control_frames import (
    ACTIONS,
    ControlDispatcher,
    build_control,
)


class _Server:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def inject(self, text: str) -> None:
        self.calls.append(("inject", text))

    def stop_after_iteration(self) -> None:
        self.calls.append(("stop",))

    def set_autonomy(self, arg: str) -> None:
        self.calls.append(("set_autonomy", arg))

    def set_model(self, arg: str) -> None:
        self.calls.append(("set_model", arg))


def _handle(dispatcher: ControlDispatcher, frame: str):
    return asyncio.run(dispatcher.handle(frame))


def test_denied_action_is_dropped_not_raised():
    server = _Server()
    d = ControlDispatcher(server, allowed_actions={"chat", "stop"})
    assert _handle(d, build_control("set", key="model", arg="cloud")) is None
    assert server.calls == []


def test_allowed_action_still_runs():
    server = _Server()
    d = ControlDispatcher(server, allowed_actions={"inject"})
    assert _handle(d, build_control("inject", text="hi")) is not None
    assert server.calls == [("inject", "hi")]


def test_none_allows_every_action():
    server = _Server()
    d = ControlDispatcher(server)
    _handle(d, build_control("set", key="autonomy", arg="brisk"))
    assert server.calls == [("set_autonomy", "brisk")]


def test_default_remote_set_excludes_config_mutation():
    cfg = UIServerConfig()
    assert "set" not in cfg.remote_actions
    assert {"chat", "answer", "stop", "inject", "changeset_diff"} <= set(cfg.remote_actions)


def test_every_wire_action_is_known():
    # Guards against adding an action to the dispatcher but forgetting the
    # allowlist vocabulary, which would silently deny it for remote clients.
    assert set(ACTIONS) == {"chat", "answer", "stop", "inject", "set", "changeset_diff"}
