"""Off-the-record session modes — what the agent is allowed to leave on disk.

Four modes, set once per session and read by every store that writes:

    standard    normal. Everything persists in the clear (today's behaviour).
    incognito   nothing persists. Session, Q/A log, side-logs, facts, memory.db,
                checkpoint journal, notes and the log file are all suppressed.
    private     incognito **plus** a hard requirement that every LLM endpoint is
                local (enforced in core/agent.py and core/model_tier.py) — the
                transcript never leaves the machine either.
    vault       everything persists, encrypted at rest. Requires a passphrase;
                the key exists only in this process's memory.

``incognito`` used to mean "session.json is not written", while the Q/A log,
side-logs, facts rounds, memory.db and checkpoint pre-image blobs kept writing
the same conversation to disk beside it. Every one of those paths now asks this
module first, which is why the gate lives here and not in each store.

Crypto: Argon2id (via ``cryptography``, needs OpenSSL 3.2+) derives a 256-bit
key from the passphrase; AES-256-GCM seals each file. Two on-disk shapes:

    blob    whole-file payload — one nonce, one tag. Rewritten in full.
    frames  append-only log — each record is length-prefixed and sealed with its
            own nonce, so appending never rewrites (or re-nonces) what is there.

Both carry a plaintext header naming the KDF parameters and salt, so a vault
file is self-describing and stays readable after a parameter change.

Not hidden: file *names*, sizes and mtimes. An observer with disk access can
still see that a session happened at a given time and roughly how long it was —
only the contents are protected. Losing the passphrase loses the data; there is
no recovery path and that is the point.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import struct
import threading
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

MODES = ("standard", "incognito", "private", "vault")

#: Suffix appended to a logical path to get its sealed sibling.
ENC_SUFFIX = ".enc"

_MAGIC = b"OCV1\n"

# Argon2id parameters. 64 MiB / 3 passes / 4 lanes is the interactive profile:
# ~0.1s on this class of machine, which is cheap enough to run at every unlock
# and expensive enough that a stolen disk is not brute-forced from a wordlist.
_KDF = {"kdf": "argon2id", "m": 65536, "t": 3, "p": 4}
_VERIFIER = b"ocvault-verify-v1"


# ---------------------------------------------------------------------------
# Mode + key state (process-global, like the session it describes)
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_mode: str = "standard"
_key: bytes | None = None
_salt: bytes | None = None


def set_mode(mode: str) -> None:
    """Record the active session mode. Called by ``Agent.set_session_mode``."""
    global _mode
    with _lock:
        _mode = mode if mode in MODES else "standard"


def mode() -> str:
    with _lock:
        return _mode


def persist_allowed() -> bool:
    """May anything be written at all?

    False for incognito and private, and false for a *locked* vault: with no key
    there is no way to seal, and writing the plaintext instead would turn a
    forgotten unlock into exactly the disclosure this module exists to prevent.
    """
    with _lock:
        if _mode in ("incognito", "private"):
            return False
        return not (_mode == "vault" and _key is None)


def encrypting() -> bool:
    """True when writes must be sealed — vault mode with a key in hand."""
    with _lock:
        return _mode == "vault" and _key is not None


def locked() -> bool:
    """True in vault mode before ``unlock()`` — reads and writes cannot proceed."""
    with _lock:
        return _mode == "vault" and _key is None


def lock() -> None:
    """Forget the key. The mode stays; nothing more can be read or written.

    Databases are sealed and their plaintext working copies removed first —
    after this returns there is no way back in without the passphrase.
    """
    global _key
    try:
        discard_working_copies()
    except Exception:
        logger.warning("vault: sealing databases on lock failed", exc_info=True)
    with _lock:
        _key = None


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

class VaultError(RuntimeError):
    """Vault could not be opened — missing dependency, bad passphrase, corrupt file."""


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:  # optional dep, same as notify e2e / credpool
        raise VaultError(
            "vault mode needs the 'cryptography' package: pip install 'cryptography>=44'"
        ) from exc
    return AESGCM(key)


def _derive(passphrase: str, salt: bytes) -> bytes:
    try:
        from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
    except ImportError as exc:
        raise VaultError(
            "vault mode needs cryptography>=44 built against OpenSSL 3.2+ for Argon2id"
        ) from exc
    kdf = Argon2id(
        salt=salt, length=32,
        iterations=_KDF["t"], lanes=_KDF["p"], memory_cost=_KDF["m"],
    )
    return kdf.derive(passphrase.encode("utf-8"))


def header_path(agent_dir: Path) -> Path:
    """Where the salt + verifier live. Holds no plaintext of the conversation."""
    return Path(agent_dir) / "vault" / "vault.json"


def unlock(passphrase: str, agent_dir: Path) -> None:
    """Derive and cache the key for this process.

    First call for a project writes the salt and a verifier token; later calls
    check the passphrase against that verifier, so a typo fails immediately
    instead of producing a directory of files that cannot be decrypted.
    """
    global _key, _salt
    if not passphrase:
        raise VaultError("empty passphrase")

    path = header_path(Path(agent_dir))
    existing: dict[str, Any] | None = None
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VaultError(f"vault header unreadable: {path}") from exc

    if existing:
        salt = base64.b64decode(existing["salt"])
        params = {k: existing.get(k, _KDF[k]) for k in ("kdf", "m", "t", "p")}
        if params["kdf"] != "argon2id":
            raise VaultError(f"unsupported vault kdf: {params['kdf']}")
        key = _derive_with(passphrase, salt, params)
        token = base64.b64decode(existing["verifier"])
        try:
            _aesgcm(key).decrypt(token[:12], token[12:], _MAGIC)
        except Exception as exc:
            raise VaultError("wrong passphrase") from exc
    else:
        salt = secrets.token_bytes(16)
        key = _derive(passphrase, salt)
        nonce = secrets.token_bytes(12)
        token = nonce + _aesgcm(key).encrypt(nonce, _VERIFIER, _MAGIC)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_private(path, json.dumps(
            {**_KDF, "salt": base64.b64encode(salt).decode(),
             "verifier": base64.b64encode(token).decode()},
            indent=2,
        ).encode("utf-8"))

    with _lock:
        _key, _salt = key, salt


def _derive_with(passphrase: str, salt: bytes, params: dict) -> bytes:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
    kdf = Argon2id(salt=salt, length=32, iterations=int(params["t"]),
                   lanes=int(params["p"]), memory_cost=int(params["m"]))
    return kdf.derive(passphrase.encode("utf-8"))


def _write_private(path: Path, data: bytes) -> None:
    """Write 0600 from creation — never world-readable, not even briefly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _require_key() -> bytes:
    with _lock:
        if _key is None:
            raise VaultError("vault is locked — unlock it first")
        return _key


# ---------------------------------------------------------------------------
# Sealed file format
# ---------------------------------------------------------------------------

def sealed_path(path: Path | str) -> Path:
    """Logical path → the path actually written in vault mode."""
    p = Path(path)
    return p if p.name.endswith(ENC_SUFFIX) else p.with_name(p.name + ENC_SUFFIX)


def _header_bytes(kind: str) -> bytes:
    with _lock:
        salt = _salt or b""
    head = {**_KDF, "cipher": "aes-256-gcm", "kind": kind,
            "salt": base64.b64encode(salt).decode()}
    return _MAGIC + json.dumps(head, separators=(",", ":")).encode("utf-8") + b"\n"


def _parse_header(fh) -> dict:
    magic = fh.read(len(_MAGIC))
    if magic != _MAGIC:
        raise VaultError("not a vault file")
    line = b""
    while not line.endswith(b"\n"):
        ch = fh.read(1)
        if not ch:
            raise VaultError("truncated vault header")
        line += ch
    return json.loads(line.decode("utf-8"))


def _aad(kind: str, index: int) -> bytes:
    """Bind each record to its file kind and position, so frames cannot be
    reordered, duplicated or moved between files without the tag failing."""
    return _MAGIC + kind.encode() + b"|" + str(index).encode()


def seal_bytes(payload: bytes) -> bytes:
    """One whole-file sealed blob, header included."""
    key = _require_key()
    nonce = secrets.token_bytes(12)
    return _header_bytes("blob") + nonce + _aesgcm(key).encrypt(nonce, payload, _aad("blob", 0))


def open_bytes(raw: bytes) -> bytes:
    """Inverse of :func:`seal_bytes`."""
    key = _require_key()
    import io
    fh = io.BytesIO(raw)
    head = _parse_header(fh)
    if head.get("kind") != "blob":
        raise VaultError(f"expected a blob vault file, got {head.get('kind')!r}")
    body = fh.read()
    try:
        return _aesgcm(key).decrypt(body[:12], body[12:], _aad("blob", 0))
    except Exception as exc:
        raise VaultError("cannot decrypt (wrong key or tampered file)") from exc


def _seal_frame(index: int, payload: bytes) -> bytes:
    key = _require_key()
    nonce = secrets.token_bytes(12)
    ct = _aesgcm(key).encrypt(nonce, payload, _aad("frames", index))
    return struct.pack(">I", len(ct)) + nonce + ct


def _iter_frames(path: Path) -> Iterator[bytes]:
    """Yield each frame's plaintext in order.

    A truncated tail (killed mid-append) ends iteration instead of raising: the
    records already written stay readable, which is the whole point of framing.
    """
    key = _require_key()
    with open(path, "rb") as fh:
        head = _parse_header(fh)
        if head.get("kind") != "frames":
            raise VaultError(f"expected a frames vault file, got {head.get('kind')!r}")
        index = 0
        while True:
            size_raw = fh.read(4)
            if len(size_raw) < 4:
                return
            (size,) = struct.unpack(">I", size_raw)
            rest = fh.read(12 + size)
            if len(rest) < 12 + size:
                logger.warning("vault: truncated frame %d in %s", index, path.name)
                return
            try:
                yield _aesgcm(key).decrypt(rest[:12], rest[12:], _aad("frames", index))
            except Exception as exc:
                raise VaultError(f"cannot decrypt frame {index} of {path.name}") from exc
            index += 1


# ---------------------------------------------------------------------------
# IO facade — what the stores call instead of Path.read_text / write_text
# ---------------------------------------------------------------------------
#
# Every function is a no-op (write) or a miss (read) when persistence is off, so
# a caller never has to branch on the mode itself.

def write_text(path: Path | str, text: str) -> bool:
    """Write *text* at the logical *path*. Returns whether anything was written."""
    if not persist_allowed():
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if encrypting():
        _write_private(sealed_path(p), seal_bytes(text.encode("utf-8")))
        # A previous standard-mode file at the same logical path would still be
        # readable; the sealed copy replaces it rather than shadowing it.
        _unlink_quietly(p)
    else:
        p.write_text(text, encoding="utf-8")
    return True


def read_text(path: Path | str) -> str | None:
    """Read the logical *path*, sealed or not. None when absent or unreadable."""
    p = Path(path)
    enc = sealed_path(p)
    if enc.exists():
        if locked():
            return None
        try:
            return open_bytes(enc.read_bytes()).decode("utf-8")
        except (VaultError, OSError):
            logger.warning("vault: cannot read %s", enc.name)
            return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def write_json(path: Path | str, data: Any, indent: int | None = 2) -> bool:
    return write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))


def read_json(path: Path | str) -> Any | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def append_jsonl(path: Path | str, record: Any) -> bool:
    """Append one record to a logical JSONL path."""
    if not persist_allowed():
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    if encrypting():
        enc = sealed_path(p)
        if not enc.exists():
            _write_private(enc, _header_bytes("frames"))
        index, end = _frames_end(enc)
        with open(enc, "r+b") as fh:
            fh.truncate(end)
            fh.seek(end)
            fh.write(_seal_frame(index, line.encode("utf-8")))
    else:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return True


def iter_jsonl(path: Path | str) -> Iterator[Any]:
    """Yield decoded records from a logical JSONL path, sealed or not.

    Undecodable lines are skipped: a half-written plaintext tail is normal for
    an append log and must not take out the records before it.
    """
    p = Path(path)
    enc = sealed_path(p)
    if enc.exists():
        if locked():
            return
        try:
            for payload in _iter_frames(enc):
                try:
                    yield json.loads(payload.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        except (VaultError, OSError):
            logger.warning("vault: cannot read frames from %s", enc.name)
        return
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def rewrite_jsonl(path: Path | str, records: list) -> bool:
    """Replace a logical JSONL file wholesale (trim / prune paths)."""
    if not persist_allowed():
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if encrypting():
        enc = sealed_path(p)
        buf = bytearray(_header_bytes("frames"))
        for i, rec in enumerate(records):
            buf += _seal_frame(i, json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        _write_private(enc, bytes(buf))
        _unlink_quietly(p)
    else:
        p.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )
    return True


def exists(path: Path | str) -> bool:
    """Does the logical path exist in either shape?"""
    p = Path(path)
    return p.exists() or sealed_path(p).exists()


def unlink(path: Path | str) -> None:
    """Remove both shapes of a logical path."""
    p = Path(path)
    _unlink_quietly(p)
    _unlink_quietly(sealed_path(p))


def glob(directory: Path | str, pattern: str) -> list[Path]:
    """Glob *logical* names in *directory* — sealed siblings answer to the
    plaintext pattern, so callers keep matching ``round-*.json``."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out: dict[str, Path] = {}
    for p in d.glob(pattern):
        out[p.name] = p
    for p in d.glob(pattern + ENC_SUFFIX):
        out.setdefault(p.name[: -len(ENC_SUFFIX)], d / p.name[: -len(ENC_SUFFIX)])
    return sorted(out.values())


def rglob(directory: Path | str, pattern: str) -> list[Path]:
    """Recursive :func:`glob`, returning logical paths."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out: dict[str, Path] = {}
    for p in d.rglob(pattern):
        out[str(p)] = p
    for p in d.rglob(pattern + ENC_SUFFIX):
        logical = Path(str(p)[: -len(ENC_SUFFIX)])
        out.setdefault(str(logical), logical)
    return sorted(out.values())


def _frames_end(enc: Path) -> tuple[int, int]:
    """(frame count, byte offset just past the last intact frame).

    Walks the length prefixes only — no decryption, so appending stays cheap on
    a long log. The offset is what makes a killed-mid-append file self-healing:
    the next append truncates the partial tail rather than writing a frame after
    it, which would put every later frame at the wrong index.
    """
    try:
        with open(enc, "rb") as fh:
            _parse_header(fh)
            size_total = os.fstat(fh.fileno()).st_size
            n, good = 0, fh.tell()
            while True:
                size_raw = fh.read(4)
                if len(size_raw) < 4:
                    return n, good
                (size,) = struct.unpack(">I", size_raw)
                end = fh.seek(12 + size, os.SEEK_CUR)
                if end > size_total:
                    return n, good
                n, good = n + 1, end
    except (OSError, VaultError):
        return 0, 0


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# SQLite backing (memory.db and friends)
# ---------------------------------------------------------------------------
#
# SQLite cannot be sealed in place without SQLCipher, so the database file is
# moved off the disk instead:
#
#   incognito/private  a shared-cache in-memory database. Every thread's
#                      connection reaches the same one, and it dies with the
#                      process.
#   vault              a working copy under $XDG_RUNTIME_DIR (tmpfs, 0700), with
#                      the sealed image at <logical>.enc. Unsealed on open,
#                      resealed after each write. The plaintext copy lives in RAM
#                      and is unlinked on exit — it is never on persistent
#                      storage, but a swapped-out tmpfs page is a caveat worth
#                      knowing about.

_sqlite_work: dict[str, Path] = {}   # logical db path → tmpfs working copy


def _runtime_dir() -> Path | None:
    base = os.environ.get("XDG_RUNTIME_DIR")
    if not base or not Path(base).is_dir():
        return None
    d = Path(base) / "owncoder-vault"
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    return d


_memory_keepalive: dict[str, Any] = {}   # dsn → an open connection


def _memory_dsn(logical: Path) -> str:
    """A shared-cache in-memory database standing in for *logical*.

    Distinct name per logical path so two stores don't collide, shared cache so
    per-thread connections reach the same database — and one connection is held
    open here for the life of the process, because a shared-cache memory
    database is destroyed the moment its last connection closes.
    """
    dsn = f"file:ocmem_{_digest(str(logical))}?mode=memory&cache=shared"
    if dsn not in _memory_keepalive:
        import sqlite3
        _memory_keepalive[dsn] = sqlite3.connect(dsn, uri=True, check_same_thread=False)
    return dsn


def sqlite_target(logical: Path | str) -> tuple[str, bool]:
    """Where a store should actually open *logical*. Returns (dsn, uri_flag)."""
    logical = Path(logical)
    if persist_allowed() and not encrypting():
        return str(logical), False

    if not persist_allowed():
        return _memory_dsn(logical), True

    work = _sqlite_work.get(str(logical))
    if work is None:
        rt = _runtime_dir()
        if rt is None:
            logger.warning("vault: no XDG_RUNTIME_DIR — %s stays in memory only "
                           "and will not be sealed", logical.name)
            return _memory_dsn(logical), True
        work = rt / (_digest(str(logical)) + ".db")
        enc = sealed_path(logical)
        if enc.exists() and not work.exists():
            try:
                _write_private(work, open_bytes(enc.read_bytes()))
            except VaultError:
                logger.warning("vault: cannot open sealed database %s", enc.name)
        _sqlite_work[str(logical)] = work
    return str(work), False


def reseal_sqlite(logical: Path | str) -> None:
    """Re-encrypt a database's working copy. No-op outside vault mode."""
    if not encrypting():
        return
    work = _sqlite_work.get(str(Path(logical)))
    if work is None or not work.exists():
        return
    try:
        _write_private(sealed_path(Path(logical)), seal_bytes(work.read_bytes()))
    except (VaultError, OSError):
        logger.warning("vault: could not seal %s", Path(logical).name)


def reseal_all() -> None:
    for logical in list(_sqlite_work):
        reseal_sqlite(logical)


def discard_working_copies() -> None:
    """Seal, then remove the tmpfs plaintext. Called on exit and on lock."""
    reseal_all()
    for logical, work in list(_sqlite_work.items()):
        _unlink_quietly(work)
        for suffix in ("-wal", "-shm"):
            _unlink_quietly(Path(str(work) + suffix))
        _sqlite_work.pop(logical, None)


def _digest(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------

def prompt_and_unlock(agent_dir: Path | str, ask=None) -> None:
    """Ask for the passphrase (twice, when creating a new vault) and unlock.

    *ask* defaults to ``getpass`` so the passphrase never reaches the terminal,
    the scrollback, or the shell history; a UI can pass its own prompt.
    Raises :class:`VaultError` if the vault stays locked.
    """
    if ask is None:
        import getpass
        ask = getpass.getpass
    first_time = not header_path(Path(agent_dir)).exists()
    passphrase = ask("Vault passphrase: ")
    if first_time:
        if ask("Confirm passphrase: ") != passphrase:
            raise VaultError("passphrases did not match")
        if len(passphrase) < 8:
            raise VaultError("passphrase must be at least 8 characters")
    unlock(passphrase, Path(agent_dir))


import atexit

atexit.register(lambda: discard_working_copies() if encrypting() else None)


def describe() -> str:
    """One line for /status and the mode banner."""
    m = mode()
    if m == "standard":
        return "standard: sessions, logs and memory persist in the clear"
    if m == "incognito":
        return "incognito: nothing is written to disk — session, logs, notes, memory"
    if m == "private":
        return "private: incognito, plus every LLM call is pinned to a local endpoint"
    if locked():
        return "vault: LOCKED — unlock with a passphrase before this session can persist"
    return "vault: everything persists, encrypted at rest (AES-256-GCM, Argon2id)"
