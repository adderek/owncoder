"""``expect_rev`` — compare-and-swap for file writes.

Several agents (and the user) can edit one worktree at the same time. The edit
journal in core/checkpoint.py already *detects* that after the fact; this
prevents it: the agent states which revision of a file it means to edit, and the
write is refused when the file is no longer at that revision.

A revision id is the sha256 of the file's text as the tools layer sees it —
``checkpoint.sha_text`` — so it needs no registry, is verifiable by anyone from
the bytes alone, and survives a restart. Short handles (``r7``) exist only
because models carry three tokens more reliably than sixty-four hex characters:
they are minted at read time, resolved to a sha at the tool boundary, and never
stored durably. Their scope is this process (an agent and its in-process
subagents); a handle from anywhere else is *unknown* and is rejected rather than
being quietly downgraded to "edit whatever is on disk".

Enforcement is staged via ``[tools.revisions] mode``:

    off       the argument is ignored entirely
    warn      a missing expect_rev is fine; a *wrong* one still refuses
    require   every mutating call must carry an expect_rev

A supplied-but-wrong pin refuses in every mode including ``warn``: the agent
made an explicit assertion, and honouring a false assertion is worse than
having none.

Timing: the check runs immediately before the write, using the same content the
write is derived from where the caller has it. That closes the window the agent
can see (read → think → write) but not the last few syscalls; the cross-process
write lock that closes the rest is Step 2 of docs/TODO-2.md.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

MODES = ("off", "warn", "require")

# Shortest accepted sha prefix. Below this, collisions across a repo's files
# stop being hypothetical and a pin would start matching the wrong content.
MIN_PREFIX = 7

_HANDLE_RE = re.compile(r"^r\d+$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")

# Prefixes reserved for Steps 3+ (subtree / commit pinning). Named here so they
# are rejected explicitly instead of falling through to "unparseable".
_RESERVED_PREFIXES = ("tree:", "git:")

_mode = "off"
_handles_on = True

# handle -> (path, sha) and its inverse, so re-reading an unchanged file
# re-uses the handle the agent already has rather than minting a new one.
_by_handle: dict[str, tuple[str, str]] = {}
_by_pin: dict[tuple[str, str], str] = {}
_counter = 0

# Handles are cheap but not free; a long session reading thousands of files
# should not grow this without bound. Oldest are dropped first — an expired
# handle is reported as unknown, which is a rejection, never a silent write.
_MAX_HANDLES = 512


def reset() -> None:
    """Back to defaults with an empty handle registry (tests, new session)."""
    global _mode, _handles_on, _counter
    _mode = "off"
    _handles_on = True
    _counter = 0
    _by_handle.clear()
    _by_pin.clear()


def setup(config) -> None:
    """Read ``[tools.revisions]``. Never raises — a bad value falls back to off."""
    global _mode, _handles_on
    reset()
    section = getattr(getattr(config, "tools", None), "revisions", None)
    if section is None:
        return
    raw = str(getattr(section, "mode", "off") or "off").strip().lower()
    if raw not in MODES:
        logger.warning("revisions: unknown mode %r; treating as 'off'", raw)
        raw = "off"
    _mode = raw
    _handles_on = bool(getattr(section, "handles", True))


def mode() -> str:
    return _mode


def enabled() -> bool:
    return _mode != "off"


def handles_enabled() -> bool:
    return _handles_on


def short(sha: str | None) -> str:
    """A sha as it appears in a message: first 8 hex chars."""
    return (sha or "?")[:8]


def sha_of_text(text: str) -> str:
    from agent.core.checkpoint import sha_text
    return sha_text(text)


def sha_of_file(fpath: Path) -> str | None:
    """Current revision of *fpath*, or None when it does not exist / is unreadable.

    Decoded exactly the way every other hash in the pipeline is (utf-8 with
    ``errors="replace"``), so a rev minted at read time compares equal to the
    ``after_sha`` the same file's write recorded.
    """
    try:
        if not fpath.is_file():
            return None
        return sha_of_text(fpath.read_bytes().decode("utf-8", errors="replace"))
    except OSError:
        return None


def mint(path: str, sha: str) -> str | None:
    """Register ``(path, sha)`` and return its short handle. None when off."""
    global _counter
    if not _handles_on or not sha:
        return None
    existing = _by_pin.get((path, sha))
    if existing is not None:
        return existing
    _counter += 1
    handle = f"r{_counter}"
    _by_handle[handle] = (path, sha)
    _by_pin[(path, sha)] = handle
    while len(_by_handle) > _MAX_HANDLES:
        oldest, pin = next(iter(_by_handle.items()))
        _by_handle.pop(oldest, None)
        _by_pin.pop(pin, None)
    return handle


def annotate_read(result: dict, path: str, fpath: Path,
                  content: str | None = None) -> dict:
    """Add ``rev`` (and a handle) to a read result so the agent can quote it back.

    Pass *content* when the caller has already read the whole file — the rev is
    always the whole file's, even for a windowed read. A no-op when revisions are
    off, so nothing is spent on tokens the agent has no use for. Mutates and
    returns *result*.
    """
    if not enabled() or result.get("error"):
        return result
    sha = sha_of_text(content) if content is not None else sha_of_file(fpath)
    if sha is None:
        return result
    result["rev"] = f"sha256:{sha}"
    handle = mint(path, sha)
    if handle:
        result["rev_handle"] = handle
    return result


def _last_writer(path: str, sha: str | None) -> str | None:
    """Which actor left *path* at *sha*, per the edit journal. None if unknown."""
    if not sha:
        return None
    try:
        from agent.core import checkpoint
        for entry in reversed(checkpoint.journal_entries()):
            if entry.get("path") == path and entry.get("after_sha") == sha:
                return entry.get("actor")
    except Exception:
        logger.debug("revisions: could not attribute %s", path, exc_info=True)
    return None


def _reject(path: str, message: str, *, expected: str | None,
            current: str | None, actor: str | None = None) -> dict:
    """The one refusal shape. Carries the current rev so a retry is correct."""
    err: dict = {
        "error": message,
        "kind": "rev_mismatch",
        "path": path,
        "expected_rev": expected,
        "current_rev": f"sha256:{current}" if current else None,
    }
    if actor:
        err["current_actor"] = actor
    if current:
        handle = mint(path, current)
        if handle:
            err["current_rev_handle"] = handle
    return err


def _describe(path: str, current: str | None) -> str:
    if current is None:
        return f"{path} does not exist"
    return f"{path} is at {short(current)}…"


def check(path: str, fpath: Path, expect_rev: str | None,
          *, content: str | None = None) -> dict | None:
    """Validate a pin before a write. Returns an error dict, or None to proceed.

    *content* is the file's current text when the caller has already read it,
    saving a second read; omit it and the file is hashed from disk.
    """
    if _mode == "off":
        return None

    if content is not None:
        current = sha_of_text(content)
    else:
        current = sha_of_file(fpath)

    raw = (expect_rev or "").strip()
    if not raw:
        if _mode != "require":
            return None
        hint = "new" if current is None else f"sha256:{current}"
        return _reject(
            path,
            f"expect_rev is required ([tools.revisions] mode = \"require\"). "
            f"{_describe(path, current)} — pass expect_rev=\"{hint}\".",
            expected=None, current=current,
        )

    lowered = raw.lower()

    if lowered == "any":
        # Explicit opt-out. Recorded on the journal entry by the caller so a
        # later reader can tell "did not pin" from "was not asked to".
        return None

    if lowered == "new":
        if current is None:
            return None
        return _reject(
            path,
            f"expect_rev=\"new\" but {path} already exists (at {short(current)}…). "
            f"Read it before editing, or pass its rev to overwrite it.",
            expected="new", current=current,
            actor=_last_writer(path, current),
        )

    for prefix in _RESERVED_PREFIXES:
        if lowered.startswith(prefix):
            return _reject(
                path,
                f"expect_rev {raw!r} uses the reserved {prefix!r} form, which is "
                f"not implemented yet. Use a file sha256 or \"any\".",
                expected=raw, current=current,
            )

    if lowered.startswith("sha256:"):
        wanted = lowered[len("sha256:"):]
    elif _HANDLE_RE.match(lowered):
        pin = _by_handle.get(lowered)
        if pin is None:
            return _reject(
                path,
                f"expect_rev {raw!r} is not a revision this agent handed out "
                f"(handles do not survive a restart and are not shared between "
                f"agent processes). Re-read {path} and use the rev it returns.",
                expected=raw, current=current,
            )
        held_path, wanted = pin
        if held_path != path:
            # A sha names bytes, not identity: two identical files share one, and
            # a rename preserves it. A pin is always (path, rev), so a handle
            # minted for another file must not authorise this write.
            return _reject(
                path,
                f"expect_rev {raw!r} was minted for {held_path}, not {path}. "
                f"Re-read {path} and use its own rev.",
                expected=raw, current=current,
            )
    elif _HEX_RE.match(lowered) and len(lowered) >= MIN_PREFIX:
        wanted = lowered
    elif _HEX_RE.match(lowered):
        return _reject(
            path,
            f"expect_rev {raw!r} is too short — give at least {MIN_PREFIX} hex "
            f"characters, or the full \"sha256:…\" form.",
            expected=raw, current=current,
        )
    else:
        return _reject(
            path,
            f"expect_rev {raw!r} is not a revision. Use \"sha256:<hex>\", a hex "
            f"prefix of {MIN_PREFIX}+ characters, a handle like \"r7\", \"new\", "
            f"or \"any\" to opt out.",
            expected=raw, current=current,
        )

    if not _HEX_RE.match(wanted) or len(wanted) < MIN_PREFIX:
        return _reject(
            path,
            f"expect_rev {raw!r} is not a usable sha256 (need {MIN_PREFIX}+ hex "
            f"characters).",
            expected=raw, current=current,
        )

    if current is None:
        return _reject(
            path,
            f"expect_rev {short(wanted)}… was given but {path} does not exist. "
            f"Use expect_rev=\"new\" to create it.",
            expected=raw, current=None,
        )

    if current.startswith(wanted):
        return None

    actor = _last_writer(path, current)
    by = f" (edited by {actor})" if actor else ""
    return _reject(
        path,
        f"{path} is at {short(current)}…, you expected {short(wanted)}…{by}. "
        f"Re-read before editing.",
        expected=raw, current=current, actor=actor,
    )


def journal_note(expect_rev: str | None) -> str | None:
    """What to record on the journal entry for a write, or None to record nothing.

    Only the *shape* of the assertion is kept — "any" (an explicit opt-out) or
    "pinned" — because the sha itself is already the entry's ``before_sha``.
    """
    if not enabled():
        return None
    raw = (expect_rev or "").strip().lower()
    if raw == "any":
        return "any"
    return "pinned" if raw else None


# Shared wording for the three tools' schemas, so the agent is told the same
# thing whichever one it reaches for.
ARG_DESCRIPTION = (
    "Revision this write assumes the file is at, from read_file's `rev`/"
    "`rev_handle` (\"sha256:…\", a 7+ char hex prefix, or a handle like \"r7\"). "
    "\"new\" asserts the file does not exist yet; \"any\" opts out. "
    "A wrong value refuses the write instead of overwriting somebody else's edit."
)
