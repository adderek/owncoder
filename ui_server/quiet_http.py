"""ThreadingHTTPServer that logs handler errors instead of printing them.

The stdlib server writes "Exception occurred during processing of request"
plus a full traceback straight to stderr. In HTTP UI mode the terminal is a
supervision console, not the UI: a browser tab closed mid-response raises
BrokenPipeError and the traceback lands on top of the startup banner. Worse,
those frames carry request locals (prompt text, paths, session ids), so they
bypass the vault/incognito rules that gate the file log.

Routing them through the logger keeps stderr for things the operator must act
on, and lets the privacy modes decide what is written where.
"""
from __future__ import annotations

import logging
from http.server import ThreadingHTTPServer

logger = logging.getLogger(__name__)

# Client went away mid-response — routine, not worth a stack trace.
_DISCONNECT = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer whose handler exceptions go to the log."""

    def handle_error(self, request, client_address) -> None:  # noqa: D102
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, _DISCONNECT):
            logger.debug("http: client %s disconnected: %s", client_address, exc)
            return
        logger.exception("http: unhandled error serving %s", client_address)
