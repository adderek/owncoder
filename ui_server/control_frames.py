"""Control frames — client → agent messages over the same wire as events.

Display events flow agent → client (RemoteBridge). Control frames flow the
other way: a remote client asks the agent to start a turn, answer a pending
question, stop, inject a mid-turn note, or change a runtime knob.

Envelope (one JSON object per frame), versioned with the ipc event protocol so
a single hello negotiation covers both directions:

    {"v": 1, "type": "control", "action": "chat",   "text": "fix the bug"}
    {"v": 1, "type": "control", "action": "answer",  "id": "q-…", "choice": "yes"}
    {"v": 1, "type": "control", "action": "stop"}
    {"v": 1, "type": "control", "action": "inject",  "text": "also update docs"}
    {"v": 1, "type": "control", "action": "set",     "key": "autonomy", "arg": "brisk"}

`ControlDispatcher.handle` applies inject/stop/set against a UIServer and routes
answers to a supplied sink. `chat` is returned for the loop owner to run (it is
the long-lived turn, not a fire-and-forget action), so the dispatcher never
blocks on it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agent.ipc.messages import EVENT_PROTOCOL_VERSION

CONTROL_TYPE = "control"

# Knob name -> UIServer setter. Mirrors the slash-command surface.
_SET_METHODS = {
    "autonomy": "set_autonomy",
    "think": "set_think_level",
    "temperature": "set_temperature",
    "max_tokens": "set_max_tokens",
    "max_iter": "set_max_iter",
    "model": "set_model",
    "plan": "set_plan",
    "notify": "set_notify",
}


@dataclass
class ControlMsg:
    action: str                 # chat | answer | stop | inject | set
    text: str = ""
    id: str = ""                # answer: the question id
    choice: str = ""            # answer: chosen option
    key: str = ""               # set: knob name
    arg: str = ""               # set: knob value


def build_control(action: str, **fields: Any) -> str:
    """Serialize a control frame. Drops empty optional fields."""
    obj = {"v": EVENT_PROTOCOL_VERSION, "type": CONTROL_TYPE, "action": action}
    obj.update({k: v for k, v in fields.items() if v not in ("", None)})
    return json.dumps(obj, ensure_ascii=False)


def is_control(obj: dict) -> bool:
    return isinstance(obj, dict) and obj.get("type") == CONTROL_TYPE


def parse_control(raw: str | dict) -> ControlMsg:
    """Rebuild a ControlMsg. Raises on wrong version / non-control / no action."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    v = obj.get("v")
    if v != EVENT_PROTOCOL_VERSION:
        raise ValueError(f"unsupported control version {v!r} "
                         f"(expected {EVENT_PROTOCOL_VERSION})")
    if not is_control(obj):
        raise ValueError(f"not a control frame: type={obj.get('type')!r}")
    action = obj.get("action")
    if not action:
        raise ValueError("control frame missing action")
    return ControlMsg(
        action=action,
        text=obj.get("text", ""),
        id=obj.get("id", ""),
        choice=obj.get("choice", ""),
        key=obj.get("key", ""),
        arg=obj.get("arg", ""),
    )


class ControlDispatcher:
    """Applies inbound control frames to a UIServer.

    `on_answer(id, choice, text)` routes an answer to whoever awaits the pending
    question (e.g. the notify broker). `on_chat(text)` is invoked for a `chat`
    action so the loop owner can start a turn; if omitted, `chat` is ignored.
    Returns the parsed ControlMsg so callers can observe what happened.
    """

    def __init__(
        self,
        server: Any,
        on_answer: Callable[[str, str, str], Any] | None = None,
        on_chat: Callable[[str], Awaitable[Any] | Any] | None = None,
    ) -> None:
        self._server = server
        self._on_answer = on_answer
        self._on_chat = on_chat

    async def handle(self, raw: str | dict) -> ControlMsg:
        msg = parse_control(raw)
        if msg.action == "inject":
            self._server.inject(msg.text)
        elif msg.action == "stop":
            self._server.stop_after_iteration()
        elif msg.action == "answer":
            if self._on_answer is not None:
                res = self._on_answer(msg.id, msg.choice, msg.text)
                if hasattr(res, "__await__"):
                    await res
        elif msg.action == "set":
            method_name = _SET_METHODS.get(msg.key)
            if method_name is not None:
                getattr(self._server, method_name)(msg.arg)
        elif msg.action == "chat":
            if self._on_chat is not None:
                res = self._on_chat(msg.text)
                if hasattr(res, "__await__"):
                    await res
        else:
            raise ValueError(f"unknown control action: {msg.action!r}")
        return msg
