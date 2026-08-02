"""Work started before the startup prompts, collected after them.

Startup asks two or three questions (which profile, start the embeddings
server, the relay is down — what now) and then does several seconds of work
that never depended on the answers: probing endpoints, importing the agent and
its UI. Doing that while the questions are on screen costs nothing and is time
the user was going to spend anyway.

Only answer-independent work belongs here. What profile is chosen decides which
endpoint is *selected*, never which ones respond, and it decides nothing about
whether a module can be imported — so probing and importing start early, while
building the LLM client waits for the answer, as it must.

Nothing here prints: a background thread writing to a terminal that is showing
a prompt produces interleaved nonsense. Failures are logged and dropped — this
is a head start, and a head start that fails just means the work happens later,
in the foreground, where it used to.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None


def _warm(config) -> None:
    from agent.config.profile_detect import prefetch_endpoints
    prefetch_endpoints(config)
    # The imports the chat path is about to need. Roughly half a second, spent
    # here rather than after the last question is answered.
    import agent.core.agent          # noqa: F401
    import agent.memory.session      # noqa: F401
    import agent.ui_server           # noqa: F401


def start(config) -> None:
    """Begin the answer-independent part of startup in the background."""
    global _thread
    if _thread is not None:
        return

    def _run() -> None:
        try:
            _warm(config)
        except Exception:
            logger.debug("startup warmup failed (work will happen inline)", exc_info=True)

    _thread = threading.Thread(target=_run, name="startup-warmup", daemon=True)
    _thread.start()


def join(timeout: float = 30.0) -> None:
    """Wait for the warmup, if one is running.

    Called before the first thing that needs it. The timeout is a backstop:
    every step in here has its own, and a warmup that somehow hangs must not
    take the session with it.
    """
    global _thread
    t = _thread
    if t is None:
        return
    t.join(timeout)
    if t.is_alive():
        logger.warning("startup warmup still running after %.0fs — continuing", timeout)
    _thread = None
