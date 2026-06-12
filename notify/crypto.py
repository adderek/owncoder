"""End-to-end encryption for the notify relay path.

The relay server authenticates connections with the relay token but must not
be able to read message contents. Agent and client share a second secret (the
e2e key file) that the relay never sees; every wire message is encrypted with
it, so the relay only forwards opaque envelopes:

    {"type": "enc", "v": 1, "n": "<b64 12-byte nonce>", "c": "<b64 ciphertext>"}

Scheme: key = HKDF-SHA256(secret, salt="owncoder-notify", info="e2e-v1"),
AES-256-GCM per message with a random nonce, AAD pins the protocol version.
GCM authentication means tampered or wrong-key envelopes fail to decrypt and
are dropped. Replay of old answers is already blocked by single-use question
ids in the broker.

Generate a key:  openssl rand -base64 32 > ~/.config/agent/notify-e2e.key
(any non-empty secret works — HKDF stretches it — but use a random one).
"""
from __future__ import annotations

import base64
import json
import logging
import os

logger = logging.getLogger(__name__)

_HKDF_SALT = b"owncoder-notify"
_HKDF_INFO = b"e2e-v1"
_AAD = b"owncoder-notify-v1"
ENC_VERSION = 1


class E2EBox:
    """Symmetric AEAD box derived from a shared secret string."""

    def __init__(self, secret: str) -> None:
        if not secret:
            raise ValueError("e2e secret must not be empty")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.hashes import SHA256
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        key = HKDF(
            algorithm=SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO,
        ).derive(secret.encode("utf-8"))
        self._aead = AESGCM(key)

    def encrypt(self, message: dict) -> dict:
        nonce = os.urandom(12)
        ct = self._aead.encrypt(nonce, json.dumps(message).encode("utf-8"), _AAD)
        return {
            "type": "enc",
            "v": ENC_VERSION,
            "n": base64.b64encode(nonce).decode("ascii"),
            "c": base64.b64encode(ct).decode("ascii"),
        }

    def decrypt(self, envelope: dict) -> "dict | None":
        """Returns inner message, or None on any failure (tamper, wrong key,
        malformed envelope). Failures are dropped, never raised."""
        try:
            if envelope.get("type") != "enc" or envelope.get("v") != ENC_VERSION:
                return None
            nonce = base64.b64decode(envelope["n"])
            ct = base64.b64decode(envelope["c"])
            inner = json.loads(self._aead.decrypt(nonce, ct, _AAD))
            return inner if isinstance(inner, dict) else None
        except Exception:
            return None


def load_box(key_file: str) -> "E2EBox | None":
    """Build E2EBox from a key file. Returns None (with log) when the key file
    is unreadable/empty or cryptography is not installed — callers must treat
    that as 'channel disabled', never as 'continue in plaintext'."""
    from pathlib import Path
    try:
        secret = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("notify: cannot read e2e_key_file %s: %s", key_file, exc)
        return None
    if not secret:
        logger.warning("notify: e2e_key_file %s is empty", key_file)
        return None
    try:
        return E2EBox(secret)
    except ImportError:
        logger.warning("notify: cryptography not installed (pip install 'local-code-agent[notify]')")
        return None
