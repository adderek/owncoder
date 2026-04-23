#!/usr/bin/env python3
"""local-code-agent — entry point (shim)."""
from __future__ import annotations

from agent.cli.main import main
from agent.cli.sessions import _split_sessions
from agent.cli.logging_setup import _write_exception_dump, _setup_logging

__all__ = ["main", "_split_sessions", "_write_exception_dump", "_setup_logging"]

if __name__ == "__main__":
    main()
