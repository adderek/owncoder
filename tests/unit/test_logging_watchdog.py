"""watchdog must not log at DEBUG into agent.log — the RAG maintainer watches
the project root, so each logged inotify event on agent.log is another write."""
from __future__ import annotations

import logging

from agent.cli.logging_setup import _setup_logging


def _drop_owned_handlers():
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_owncoder", False):
            root.removeHandler(h)
            h.close()


def test_watchdog_logger_capped_at_info(tmp_path):
    logging.getLogger("watchdog").setLevel(logging.NOTSET)
    try:
        _setup_logging(agent_dir=str(tmp_path))
        assert not logging.getLogger("watchdog.observers.inotify_buffer").isEnabledFor(logging.DEBUG)
    finally:
        _drop_owned_handlers()


def test_logs_sources_can_still_enable_watchdog_debug(tmp_path):
    class _Cfg:
        level = "DEBUG"
        sources = {"watchdog": "DEBUG"}
    try:
        _setup_logging(agent_dir=str(tmp_path), logs_cfg=_Cfg())
        assert logging.getLogger("watchdog").isEnabledFor(logging.DEBUG)
    finally:
        _drop_owned_handlers()
        logging.getLogger("watchdog").setLevel(logging.NOTSET)
