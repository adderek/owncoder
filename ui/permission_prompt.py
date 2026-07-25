"""Shared wait-for-a-modal-answer logic for the Textual permission prompt.

Kept out of ui/terminal.py because everything in there lives inside a closure
that needs a real Textual app to build, and this is the part with the failure
modes worth testing: the held tool call must never be released without an
explicit allow, and the prompt must never outlive the request that raised it.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def ask_via_modal(push_screen, screen) -> str:
    """Push *screen* and wait for its dismiss value; "" means deny.

    *push_screen* is called as ``push_screen(screen, callback)`` — i.e. Textual's
    ``App.push_screen`` — and *screen* must expose ``cancel()`` to take itself
    down unanswered.

    Every path that is not a deliberate choice yields "": the screen cannot be
    shown, the answer is None, or the caller's timeout cancels the wait. Denial
    is the only safe default, since the tool call this gates has not run yet.
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    def _resolve(choice=None) -> None:
        if not fut.done():
            fut.set_result(choice or "")

    try:
        push_screen(screen, _resolve)
    except Exception:
        # Can't ask (app shutting down, screen stack unusable) → deny now rather
        # than blocking the tool call until the caller's timeout expires.
        logger.warning("permission prompt could not be shown; denying", exc_info=True)
        return ""

    try:
        return await fut
    except asyncio.CancelledError:
        # The caller's ask timeout fired: the modal must not outlive it.
        try:
            screen.cancel()
        except Exception:
            logger.debug("permission modal teardown failed", exc_info=True)
        raise
