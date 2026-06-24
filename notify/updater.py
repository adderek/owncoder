"""Client auto-update responder.

When a relay client sends `{type:"update_query", version_code, version_name}`
on connect, the agent checks the APK it holds (a `latest.json` sidecar in the
configured dir) and, if it is newer, streams the APK back as an `update_offer`
followed by base64 `update_chunk` frames. The client reassembles, verifies the
sha256, and saves the APK for the user to install manually (no auto-install).

The APK and its sidecar are produced by the build (clients/android/docker/build.sh).
`latest.json`:  {"version_code": int, "version_name": str, "sha256": hex, "file": name}

Security: frames ride the same e2e-encrypted channel as every other message,
so a relay/MITM cannot forge an offer; the client still verifies size+sha256
before trusting the bytes, and the OS rejects any APK not signed with the
installed app's key. Serving is opt-in (set notify channel `update_apk_dir`).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path

from agent.notify.messages import UpdateChunk, UpdateOffer

logger = logging.getLogger(__name__)

# Raw bytes per chunk. After base64 (~1.33x) and the e2e envelope's own base64
# the on-wire frame is ~1.8x the raw size (see channels.RELAY_MAX_FRAME_BYTES),
# so 96 KiB raw → ~180 KiB frame, safely under the relay's 256 KiB cap.
CHUNK_BYTES = 96 * 1024
# Pace below the relay's default 20 msg/s rate limit so a large APK does not
# trip the limiter and get the agent connection closed (code 4429).
INTER_CHUNK_DELAY_S = 0.07
SIDECAR = "latest.json"


class UpdateResponder:
    """Serves the newest local APK to clients asking with an older version."""

    def __init__(self, apk_dir: str) -> None:
        self._dir = Path(apk_dir).expanduser()

    def _latest(self) -> "dict | None":
        """Read+validate the latest.json sidecar. None if missing/malformed."""
        sidecar = self._dir / SIDECAR
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.debug("update: no usable %s: %s", sidecar, exc)
            return None
        if not isinstance(meta, dict):
            return None
        try:
            int(meta["version_code"])
            str(meta["version_name"])
            str(meta["sha256"])
            str(meta["file"])
        except (KeyError, TypeError, ValueError):
            logger.warning("update: %s missing required fields", sidecar)
            return None
        return meta

    async def handle(self, channel, query: dict) -> None:
        """Reply to one update_query, streaming the APK if we have a newer one."""
        meta = self._latest()
        if meta is None:
            return
        try:
            client_code = int(query.get("version_code", 0))
        except (TypeError, ValueError):
            client_code = 0
        if int(meta["version_code"]) <= client_code:
            return  # client already on this build or newer
        apk = self._dir / str(meta["file"])
        try:
            data = apk.read_bytes()
        except OSError as exc:
            logger.warning("update: cannot read APK %s: %s", apk, exc)
            return
        digest = hashlib.sha256(data).hexdigest()
        if digest != meta["sha256"]:
            logger.warning("update: %s sha256 mismatch vs %s — refusing to serve", apk, SIDECAR)
            return
        chunks = (len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES
        name = str(meta["version_name"])
        await channel.send_raw(UpdateOffer(
            version_code=int(meta["version_code"]), version_name=name,
            size=len(data), sha256=digest, chunks=chunks,
        ).to_wire())
        logger.info("update: offering %s (%d bytes, %d chunks) to client v%d",
                    name, len(data), chunks, client_code)
        for seq in range(chunks):
            start = seq * CHUNK_BYTES
            blob = data[start:start + CHUNK_BYTES]
            await channel.send_raw(UpdateChunk(
                id=name, seq=seq, last=(seq == chunks - 1),
                data=base64.b64encode(blob).decode("ascii"),
            ).to_wire())
            await asyncio.sleep(INTER_CHUNK_DELAY_S)
