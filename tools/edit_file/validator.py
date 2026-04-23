from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .matcher import (
    _count_lines, _find_exact, _find_loose_v2, _range_to_offsets,
    _candidate, _MAX_CANDIDATES,
)


@dataclass
class _ValidatedChunk:
    chunk_index: int
    path: str
    fpath: Path
    original: str
    start: int
    end: int
    replacement: str
    removed_lines: int
    added_lines: int


def _validate_chunk(
    idx: int,
    chunk: dict,
    file_cache: dict[str, tuple[Path, str]],
    resolve_fn,
    edit_cfg,
    match_override: str | None,
) -> tuple[_ValidatedChunk | None, dict | None]:
    from agent.tools.rules import get_rules

    def err(kind: str, detail: str, **extra) -> dict:
        return {"chunk_index": idx, "kind": kind, "detail": detail, **extra}

    for field_name in ("path", "anchor", "replacement"):
        if field_name not in chunk:
            return None, err("bad_input", f"missing required field: {field_name}")

    path = chunk["path"]
    anchor = chunk["anchor"]
    replacement = chunk["replacement"]

    if (
        not isinstance(path, str)
        or not isinstance(anchor, str)
        or not isinstance(replacement, str)
    ):
        return None, err("bad_input", "path/anchor/replacement must be strings")
    if not anchor:
        return None, err("bad_input", "anchor must be non-empty")

    try:
        fpath = resolve_fn(path)
    except ValueError as e:
        return None, err("bad_input", str(e))

    rules = get_rules()
    try:
        from agent.tools.files import _working_dir
        rel = str(fpath.relative_to(_working_dir()))
    except Exception:
        rel = path

    allowed, msg = rules.check_write(rel)
    if not allowed:
        return None, err("readonly", msg or f"cannot write: {path}")

    if not fpath.exists():
        return None, err("file_not_found", f"file does not exist: {path}")

    if path in file_cache:
        _, original = file_cache[path]
    else:
        original = fpath.read_text(encoding="utf-8", errors="replace")
        file_cache[path] = (fpath, original)

    total_lines = original.count("\n") + (0 if original.endswith("\n") else 1)
    if not original:
        total_lines = 0

    anchor_lines = _count_lines(anchor)
    repl_lines = _count_lines(replacement)
    cap = edit_cfg.max_chunk_lines
    if cap > 0 and (anchor_lines > cap or repl_lines > cap):
        return None, err(
            "chunk_too_large",
            f"anchor={anchor_lines} replacement={repl_lines} lines exceeds max_chunk_lines={cap}",
            anchor_lines=anchor_lines,
            replacement_lines=repl_lines,
            limit=cap,
        )
    frac = edit_cfg.max_file_fraction
    if frac > 0 and total_lines >= 20:
        if anchor_lines / total_lines > frac:
            return None, err(
                "fraction_exceeded",
                f"anchor spans {anchor_lines}/{total_lines} lines (> {frac:.0%}); refusing whole-file replace",
                anchor_lines=anchor_lines,
                total_lines=total_lines,
                limit=frac,
            )

    sha = chunk.get("anchor_sha256")
    if sha:
        actual = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
        if actual.lower() != str(sha).lower():
            return None, err("anchor_sha_mismatch", "anchor_sha256 does not match the anchor you provided; re-read the file", expected=sha, actual=actual)

    tol = max(0, edit_cfg.line_delta_tolerance)
    if "expect_removed" in chunk and chunk["expect_removed"] is not None:
        try:
            exp = int(chunk["expect_removed"])
        except (TypeError, ValueError):
            return None, err("bad_input", "expect_removed must be int")
        if abs(exp - anchor_lines) > tol:
            return None, err("delta_exceeds_tolerance", f"expect_removed={exp} but anchor has {anchor_lines} lines (tolerance ±{tol})", expected=exp, actual=anchor_lines, tolerance=tol)
    if "expect_added" in chunk and chunk["expect_added"] is not None:
        try:
            exp = int(chunk["expect_added"])
        except (TypeError, ValueError):
            return None, err("bad_input", "expect_added must be int")
        if abs(exp - repl_lines) > tol:
            return None, err("delta_exceeds_tolerance", f"expect_added={exp} but replacement has {repl_lines} lines (tolerance ±{tol})", expected=exp, actual=repl_lines, tolerance=tol)

    lo, hi = 0, len(original)
    if chunk.get("range_hint") is not None:
        rh = chunk["range_hint"]
        if (
            not isinstance(rh, (list, tuple))
            or len(rh) != 2
            or not all(isinstance(x, int) for x in rh)
        ):
            return None, err("range_hint_invalid", "range_hint must be [int, int]")
        sl, el = rh
        if sl < 1 or el < sl or el > max(total_lines, 1):
            return None, err("range_hint_invalid", f"range_hint [{sl}, {el}] invalid for file with {total_lines} lines", total_lines=total_lines)
        lo, hi = _range_to_offsets(original, sl, el)

    mode = (match_override or edit_cfg.match or "exact").lower()
    if mode == "model":
        mode = "exact"

    spans = _find_exact(original, anchor, lo, hi)
    match_mode = "exact"
    if not spans and mode == "loose":
        spans = _find_loose_v2(original, anchor, lo, hi)
        match_mode = "loose"

    if not spans:
        return None, err(
            "anchor_not_found",
            "anchor not present in file (exact search%s). Re-read the file and re-quote."
            % (" + loose fallback" if mode == "loose" else ""),
        )
    if len(spans) > 1:
        cands = [_candidate(original, s, e, i) for i, (s, e) in enumerate(spans[:_MAX_CANDIDATES])]
        return None, err("anchor_ambiguous", f"anchor matched {len(spans)} locations; add range_hint or extend anchor for uniqueness", match_count=len(spans), candidates=cands)

    s, e = spans[0]
    return _ValidatedChunk(
        chunk_index=idx,
        path=path,
        fpath=fpath,
        original=original,
        start=s,
        end=e,
        replacement=replacement,
        removed_lines=anchor_lines,
        added_lines=repl_lines,
    ), None
