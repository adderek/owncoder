"""Project snapshot injected at session start.

In a tiny project the exploration rounds cost more than the project: every round
re-sends the whole context, so ten locate/read round trips over three small files
spend many times those files' size. Loading every text file up front removes the
rounds; the prompt cache keeps the repeated prefix cheap.

Bigger projects get a map instead (paths, line counts, landmarks), so the first
read can go straight to a range.

The snapshot is a system message marked `_preload_marker`: merged into the system
prompt on the wire (a stable, cacheable prefix), dropped at compaction
(memory/compactor.py), and never updated — the text says so, and says to read a
file again after editing it.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PRELOAD_MARKER = "_preload_marker"

_PER_FILE_MAX_CHARS = 20_000
_MAP_ENTRIES = 8
_MAP_MAX_CHARS = 6_000

# What the full snapshot contained: relative path → sha of the text shown. The
# snapshot is never updated, and a model anchoring edits on it after the file
# changed loops on anchor_not_found (eval add-function: 20 requests vs 5 without
# preload). The edit tools use these to say so at the moment it matters.
_snapshot: dict[str, str] = {}
_snapshot_root: Path | None = None
_noted: set[str] = set()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _rel(path: str) -> str | None:
    if _snapshot_root is None or not path:
        return None
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(_snapshot_root))
        except ValueError:
            return None
    return os.path.normpath(path)


def mark_edited(path: str) -> str:
    """Note for an edit result when *path* was in the snapshot; once per file."""
    rel = _rel(path)
    if rel is None or rel not in _snapshot or rel in _noted:
        return ""
    _noted.add(rel)
    return (f"{rel} changed: its PROJECT SNAPSHOT copy is now stale. `current` in "
            f"this result is the file now — quote later anchors from it, not the snapshot.")


def stale_hint(path: str, current_text: str) -> str:
    """Extra anchor_not_found detail when the file no longer matches its snapshot."""
    rel = _rel(path)
    if rel is None or rel not in _snapshot or _snapshot[rel] == _sha(current_text):
        return ""
    return (" The PROJECT SNAPSHOT copy of this file is stale (the file changed "
            "since): read_file it and quote the anchor from that.")


def _text_files(config) -> list[tuple[str, str]] | None:
    """(relative path, text) of every readable text file, or None when the
    project is too big to list or the listing fails."""
    from agent.tools.files.read import list_files
    from agent.tools.rules import get_rules
    from agent.tools._common import read_deny_globs, is_read_protected

    tools = config.tools
    listing = list_files(".", max_results=int(tools.preload_map_max_files))
    if listing.get("truncated") or "files" not in listing:
        return None
    root = Path(tools.working_dir)
    rules = get_rules()
    deny = read_deny_globs()
    out: list[tuple[str, str]] = []
    for entry in listing["files"]:
        rel = entry["path"]
        # Same gates read_file applies: never preload what it would refuse.
        if not rules.check_read(rel)[0] or is_read_protected(rel, deny):
            continue
        try:
            raw = (root / rel).read_bytes()
        except OSError:
            continue
        if b"\0" in raw[:4096]:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        out.append((rel, text))
    return out


def _render_full(files: list[tuple[str, str]], total: int) -> str:
    parts = [
        f"[PROJECT SNAPSHOT · {len(files)} files · {total} chars · taken at session start]\n"
        "Every text file of this project, numbered like read_file, as it was when the "
        "session started — not updated. After you edit a file, the edit result's "
        "`current` shows its text now: use that, not this copy. Dropped at compaction."
    ]
    for rel, text in files:
        lines = text.splitlines()
        numbered = "\n".join(f"{i + 1}:{line}" for i, line in enumerate(lines))
        parts.append(f"=== {rel} · {len(lines)} lines ===\n{numbered}")
    return "\n\n".join(parts)


def _render_map(files: list[tuple[str, str]]) -> str:
    from agent.tools.files.outline import outline
    head = (f"[PROJECT MAP · {len(files)} files · taken at session start — "
            f"read_file the range you need]")
    lines: list[str] = []
    used = 0
    for n, (rel, text) in enumerate(files):
        entries = outline(text, max_entries=_MAP_ENTRIES, filename=Path(rel).name)
        marks = ", ".join(f"{e['line']}:{e['name']}" for e in entries if e["kind"] != "...")
        line = f"{rel} · {len(text.splitlines())} lines" + (f" · {marks}" if marks else "")
        if used + len(line) > _MAP_MAX_CHARS:
            lines.append(f"(+{len(files) - n} more files — list_files)")
            break
        lines.append(line)
        used += len(line) + 1
    return head + "\n" + "\n".join(lines)


def build_preload(config) -> str:
    """The snapshot text for this project, or "" for none."""
    tools = config.tools
    mode = str(getattr(tools, "preload_mode", "off") or "off").lower()
    if mode not in ("map", "full", "auto"):
        return ""
    try:
        files = _text_files(config)
    except Exception:
        logger.debug("preload: listing failed", exc_info=True)
        return ""
    if not files:
        return ""

    from agent.core.context_budget import effective_ctx_window
    from agent.core.context_state import estimate_tokens
    window = effective_ctx_window(config)
    total = sum(len(text) for _, text in files)
    fits = (window >= tools.preload_min_window
            and total <= tools.preload_max_chars
            and estimate_tokens(total) <= window * tools.preload_window_fraction
            and all(len(text) <= _PER_FILE_MAX_CHARS for _, text in files))
    if mode in ("full", "auto"):
        mode = "full" if fits else "map"

    global _snapshot_root
    _snapshot.clear()
    _noted.clear()
    _snapshot_root = Path(tools.working_dir).resolve()
    if mode != "full":
        return _render_map(files)
    _snapshot.update({rel: _sha(text) for rel, text in files})
    return _render_full(files, total)
