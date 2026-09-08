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

Writers always use ``resolve()`` (the new path). Readers use ``read_dirs()``,
which returns *every* existing location, canonical first: ``migrate_legacy()``
is best-effort, so a legacy directory can survive next to the canonical one and
a reader that looked at only one of them would silently lose records. Reads
never rename anything, so a reader can't pull a directory out from under a live
writer.
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
    """Canonical location if it exists, else the legacy one, else canonical."""
    dirs = read_dirs(base, name)
    return dirs[0] if dirs else resolve(base, name)


def read_dirs(base: Path, name: str) -> list[Path]:
    """Every existing location for the *name* stream, canonical first.

    Both may exist: migration merges per file and is best-effort, so a legacy
    directory can outlive the canonical one. Readers must union them —
    otherwise records written by an older version vanish the moment anything
    creates the canonical directory.
    """
    out: list[Path] = []
    for d in (resolve(base, name), base / name):
        if d.is_dir() and d not in out:
            out.append(d)
    return out


def migrate_legacy(base: Path) -> None:
    """Merge legacy top-level diagnostics into ``diagnostics/`` (best-effort).

    Per *file*, not per directory: one failed rename must not strand the rest
    of the stream, and a pre-existing canonical directory must not block
    migration the way a directory-level ``rename`` would.
    """
    for name in STREAMS:
        _merge_dir(base / name, resolve(base, name))
    _migrate_exception_dumps(base)


def _merge_dir(legacy: Path, new: Path) -> None:
    if not legacy.is_dir():
        return
    try:
        new.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("diagnostics: could not create %s: %s", new, e)
        return
    for p in sorted(legacy.iterdir()):
        target = new / p.name
        if target.exists():
            continue
        try:
            p.rename(target)
        except OSError as e:
            logger.warning("diagnostics: could not move %s to %s: %s", p, target, e)
    try:
        legacy.rmdir()  # succeeds only when every file moved
    except OSError:
        pass
    else:
        logger.info("diagnostics: merged %s -> %s", legacy, new)


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
    except OSError as e:
        logger.warning("diagnostics: could not create %s: %s", new, e)
        return
    for p in legacy:
        target = new / p.name
        if target.exists():
            continue
        try:
            p.rename(target)
        except OSError as e:
            logger.warning("diagnostics: could not move %s to %s: %s", p, target, e)
