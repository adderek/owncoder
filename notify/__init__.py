"""Notification channel — push progress/questions to external endpoints.

Bridges turn signals (>>>ASK, >>>BLOCKED, >>>DONE, ...) to one or more
configured channels (ntfy, signal-cli, relay server, ...) so the user can
follow and steer the agent without the full terminal UI.

Config: [notify] section (agent.toml / agent.yaml). Off by default.
"""
from agent.notify.broker import NotifyBroker
from agent.notify.messages import Answer, Notice, Question

__all__ = ["NotifyBroker", "Notice", "Question", "Answer"]
