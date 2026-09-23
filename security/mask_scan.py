"""The per-command tree walk behind the sandbox's secret masks and read-only binds.

Every sandboxed command needs the concrete paths to mask (secret files) and to
bind read-only (policy files outside the directory-level binds), and those can
only be found by walking the project tree. This module does that walk once per
command for every glob set, and records what it cost.

Limits are time and output size, not file count. A file count says nothing
about how long the walk takes or whether bwrap can take the result, and a
large-but-ordinary repository would hit any fixed number. The two real limits:

* ``timeout_s`` — the walk runs before every command, so it is bounded in
  wall time. Running out is an incomplete scan (see runner._truncated).
* ``max_matches`` — every match becomes three bwrap arguments, and bubblewrap
  refuses more than BWRAP_MAX_ARGS in total. Counted across all glob sets,
  because they share that one argument list.

Matching is exactly the fs gate's rule — a glob matches the root-relative path
or the bare filename, with fnmatch semantics — compiled once per set instead of
calling fnmatch per file per glob. Symlinks are listed the way os.walk lists
them (links to files and broken links; links to directories are neither listed
nor followed), which an external lister such as ripgrep does not do.

Nothing here caches configuration: callers pass the current values on every
call, so a setting changed at runtime applies to the next command.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

def prune_dirs() -> frozenset[str]:
    """Directories the walk never descends into.

    VCS and dependency trees dominate the file count and are not where the
    threat model's secrets or policy files live. The list is `path_policy`'s
    hidden set — the same one the indexer and grep prune — so "large and
    churns" is stated once. Directories whose *access* is restricted are not
    in it: a pruned `.ssh` would be an unmasked `.ssh`.
    """
    from . import path_policy
    return path_policy.hidden_dir_names()

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_MATCHES = 2000
# bubblewrap: "Exceeded maximum number of arguments 9000" (argc, whole command
# line). Measured on 0.12.0: 2990 `--ro-bind SRC DEST` triples plus a minimal
# command run, 2994 do not.
BWRAP_MAX_ARGS = 9000
# Leaves room under BWRAP_MAX_ARGS for the fixed binds, the protected paths,
# the interpreter paths and the command's own argv.
MAX_MATCHES_CEILING = 2900

# How often (files) the walk looks at the clock. A directory boundary is
# checked too, so a tree of many small directories is bounded as well.
_CLOCK_EVERY = 512
_WARN_FRACTION = 0.5
_WARN_INTERVAL_S = 300.0


@dataclass
class ScanStats:
    """What one scan saw. Kept for status display; see last_stats()."""
    root: str
    files: int = 0
    dirs: int = 0
    matches: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_matches: int = DEFAULT_MAX_MATCHES
    incomplete: str | None = None
    finished_at: float = 0.0          # time.time()

    @property
    def total_matches(self) -> int:
        return sum(self.matches.values())

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_matches"] = self.total_matches
        return d


@dataclass
class ScanResult:
    matches: dict[str, list[Path]]
    stats: ScanStats


_lock = threading.Lock()
_last: ScanStats | None = None
_last_warn_at = 0.0


def last_stats() -> dict | None:
    """The most recent scan's stats, or None before the first command."""
    with _lock:
        return None if _last is None else _last.as_dict()


def effective_max_matches(configured) -> int:
    """Clamp a configured match limit into what bwrap can actually take."""
    try:
        n = int(configured)
    except (TypeError, ValueError):
        n = DEFAULT_MAX_MATCHES
    return max(1, min(n, MAX_MATCHES_CEILING))


def effective_timeout(configured) -> float:
    try:
        t = float(configured)
    except (TypeError, ValueError):
        t = DEFAULT_TIMEOUT_S
    return max(0.0, t)


def _compile(globs: list[str]) -> re.Pattern | None:
    if not globs:
        return None
    return re.compile("|".join(f"(?:{fnmatch.translate(g)})" for g in globs))


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + os.sep)


def scan(
    root: Path,
    sets: dict[str, tuple[list[str], list[Path]]],
    *,
    timeout_s: float,
    max_matches: int,
    prune: "frozenset[str] | set[str] | None" = None,
    dir_sets: "frozenset[str] | set[str]" = frozenset(),
) -> ScanResult:
    """Walk *root* once and match every glob set in *sets*.

    *sets* maps a name to ``(globs, skip_dirs)``; a set is not matched inside
    its own skip_dirs (the caller covers those wholesale), and a subtree every
    set skips is not descended at all. Sets named in *dir_sets* also match
    directory names — checked before pruning, so a nested `.git` is found even
    though the walk never descends into it. Returns partial matches with
    ``stats.incomplete`` set when the time budget or the match limit ran out.
    """
    root_s = str(root)
    compiled = {
        name: (_compile(globs), [str(d) for d in skips])
        for name, (globs, skips) in sets.items()
    }
    compiled = {name: v for name, v in compiled.items() if v[0] is not None}
    matches: dict[str, list[Path]] = {name: [] for name in sets}
    stats = ScanStats(root=root_s, timeout_s=timeout_s, max_matches=max_matches,
                      matches={name: 0 for name in sets})
    start = time.monotonic()
    deadline = start + timeout_s
    total = 0

    if compiled:
        since_clock = 0
        pruned = prune_dirs() if prune is None else prune
    for dirpath, dirnames, filenames in os.walk(root_s):
            if time.monotonic() >= deadline:
                stats.incomplete = f"scan ran out of its {timeout_s:g}s time budget"
                break
            active = [
                (name, rx) for name, (rx, skips) in compiled.items()
                if not any(_under(dirpath, s) for s in skips)
            ]
            if not active:
                dirnames[:] = []
                continue
            stats.dirs += 1
            relbase = "" if dirpath == root_s else dirpath[len(root_s) + 1:] + "/"
            dir_active = [(name, rx) for name, rx in active if name in dir_sets]
            if dir_active:
                keep = []
                for dn in dirnames:
                    rel = relbase + dn
                    hit = False
                    for name, rx in dir_active:
                        if rx.match(dn) or rx.match(rel):
                            matches[name].append(Path(dirpath, dn))
                            total += 1
                            hit = True
                    if not hit:
                        keep.append(dn)     # a matched dir is bound whole
                dirnames[:] = keep
                if total > max_matches:
                    stats.incomplete = f"scan hit the {max_matches}-match limit"
                    break
            dirnames[:] = [d for d in dirnames if d not in pruned]
            for fn in filenames:
                stats.files += 1
                since_clock += 1
                if since_clock >= _CLOCK_EVERY:
                    since_clock = 0
                    if time.monotonic() >= deadline:
                        stats.incomplete = f"scan ran out of its {timeout_s:g}s time budget"
                        break
                rel = relbase + fn
                for name, rx in active:
                    if rx.match(fn) or rx.match(rel):
                        matches[name].append(Path(dirpath, fn))
                        total += 1
                if total > max_matches:
                    stats.incomplete = f"scan hit the {max_matches}-match limit"
                    break
            if stats.incomplete:
                break

    stats.elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    stats.finished_at = time.time()
    stats.matches = {name: len(v) for name, v in matches.items()}
    _record(stats)
    return ScanResult(matches=matches, stats=stats)


def _record(stats: ScanStats) -> None:
    global _last, _last_warn_at
    logger.debug(
        "mask scan %s: %d files, %d dirs, matches %s, %.1f ms%s",
        stats.root, stats.files, stats.dirs, stats.matches, stats.elapsed_ms,
        f" — incomplete: {stats.incomplete}" if stats.incomplete else "",
    )
    near_time = stats.elapsed_ms / 1000 >= stats.timeout_s * _WARN_FRACTION
    near_matches = stats.total_matches >= stats.max_matches * _WARN_FRACTION
    with _lock:
        _last = stats
        warn = (not stats.incomplete and (near_time or near_matches)
                and stats.finished_at - _last_warn_at >= _WARN_INTERVAL_S)
        if warn:
            _last_warn_at = stats.finished_at
    if warn:
        logger.warning(
            "sandbox mask scan under %s is near its limits: %.0f ms of %gs "
            "budget, %d of %d matches (%d files). Past a limit every command "
            "is refused — see security.mask_scan_timeout_s / "
            "security.mask_scan_max_matches, or /sandbox.",
            stats.root, stats.elapsed_ms, stats.timeout_s,
            stats.total_matches, stats.max_matches, stats.files,
        )
