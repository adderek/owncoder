"""Canonical locations for diagnostic artefacts under the agent directory.

Crash reports, exception dumps, failure reports and recovery records all carry
tracebacks with locals — prompts, tool arguments, file contents — and can hold
secrets. They are *diagnostics*: written by the harness, read by the harness and
the user, never policy or state. Keeping them in one ``<agent_dir>/diagnostics/``
subtree separates them from the files the sandbox must protect (core rules, path
grants, permissions, checkpoints) and gives retention/redaction one place to live.

Layout::

    <agent_dir>/diagnostics/crashes/     crash-<ts>.txt
    <agent_dir>/diagnostics/exceptions/  exception-<ts>.dump
    <agent_dir>/diagnostics/failures/    <ts>-<kind>-*.json + index.jsonl
    <agent_dir>/diagnostics/recovery/    <session_id>.json

Writers always use ``resolve()`` (the new path). Readers use ``read_dir()``,
which falls back to the legacy top-level directory so data written by an older
version is still found. ``migrate_legacy()`` moves it over; reads never rename
anything, so a reader can't pull a directory out from under a live writer.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUBDIR = "diagnostics"
CRASHES = "crashes"
EXCEPTIONS = "exceptions"
FAILURES = "failures"
RECOVERY = "recovery"
STREAMS = (CRASHES, EXCEPTIONS, FAILURES, RECOVERY)


def diagnostics_dir(base: Path) -> Path:
    """Root of the diagnostics subtree (``<agent_dir>/diagnostics``)."""
    return base / SUBDIR


def resolve(base: Path, name: str) -> Path:
    """Canonical write location for the *name* diagnostic stream."""
    return base / SUBDIR / name


def read_dir(base: Path, name: str) -> Path:
    """Location to read the *name* stream from: canonical, else legacy."""
    new = resolve(base, name)
    if new.exists():
        return new
    legacy = base / name
    return legacy if legacy.exists() else new


def migrate_legacy(base: Path) -> None:
    """Move legacy top-level diagnostics into ``diagnostics/`` (best-effort)."""
    for name in STREAMS:
        legacy = base / name
        new = resolve(base, name)
        if legacy.exists() and not new.exists():
            _move(legacy, new)
    _migrate_exception_dumps(base)


def _move(legacy: Path, new: Path) -> None:
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(new)
        logger.info("diagnostics: moved %s -> %s", legacy, new)
    except OSError as e:
        logger.warning("diagnostics: could not move %s to %s: %s", legacy, new, e)


def exception_dump_dir(base: Path) -> Path:
    """Write location for ``exception-*.dump`` files."""
    return resolve(base, EXCEPTIONS)


def _migrate_exception_dumps(base: Path) -> None:
    legacy = sorted(base.glob("exception-*.dump"))
    if not legacy:
        return
    new = exception_dump_dir(base)
    try:
        new.mkdir(parents=True, exist_ok=True)
        for p in legacy:
            p.rename(new / p.name)
    except OSError as e:
        logger.warning("diagnostics: could not move exception dumps: %s", e)
