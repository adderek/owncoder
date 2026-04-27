"""agent.ui_server — abstraction layer between user interfaces and the backend.

Phase 1: LocalUIServer wraps Agent in-process.
Later: transport-backed UIServer enabling remote/multi-UI/multi-agent access.
"""
from .protocol import UIServerProtocol
from .local import LocalUIServer

__all__ = ["UIServerProtocol", "LocalUIServer"]
