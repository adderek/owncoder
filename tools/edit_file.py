"""edit_file — anchored-chunk edit tool.

The only edit tool exposed to the LLM. Pattern the model must learn:
    read_file → quote exact anchor → edit_file

See PHASE 0 spec in `edit_file.md` and the docstring of `_build_schema` for
the input shape. Core rules:

- Exact byte match by default; loose (whitespace-collapsing) fallback only
  when `edit.match` opts in. Anchor must match exactly once.
- `range_hint` is 1-indexed inclusive; anchor must lie entirely inside it.
- Hard caps `max_chunk_lines` and `max_file_fraction` make whole-file replace
  structurally impossible.
- Atomic by default: validate all chunks, then splice bottom-up per file.
  On any failure write nothing.
- Stable error taxonomy: file_not_found, readonly, anchor_not_found,
  anchor_ambiguous, anchor_sha_mismatch, range_hint_invalid,
  delta_exceeds_tolerance, chunk_too_large, fraction_exceeded,
  chunks_overlap, atomic_rollback.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from agent.tools import register
from agent.tools.rules import get_rules


# Bound blast radius on "everything went wrong" outputs.
_MAX_CANDIDATES = 8
_CTX_LINES = 3


# ── Helpers ────────────────────────────────────────────────────────────────


def _count_lines(s: str) -> int:
    if not s:
        return 0
    n = s.count("\n")
    return n if s.endswith("\n") else n + 1


def _find_exact(hay: str, needle: str, lo: int = 0, hi: int | None = None) -> list[tuple[int, int]]:
    if not needle:
        return []
    hi = len(hay) if hi is None else hi
    out: list[tuple[int, int]] = []
    i = lo
    while True:
        j = hay.find(needle, i, hi)
        if j == -1:
            return out
        out.append((j, j + len(needle)))
        i = j + 1


def _find_loose_v2(hay: str, needle: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Whitespace-collapsing fallback. Returns spans in the original text."""
    import re
    stripped = needle.strip("\n")
    if not stripped:
        return []
    parts = re.split(r"\s+", stripped)
    pattern = r"\s+".join(re.escape(p) for p in parts if p)
    if not pattern:
        return []
    window = hay[lo:hi]
    return [(m.start() + lo, m.end() + lo) for m in re.finditer(pattern, window)]


def _line_of_offset(text: str, off: int) -> int:
    """1-indexed line number containing byte offset `off`."""
    return text.count("\n", 0, off) + 1


def _range_to_offsets(text: str, start_line: int, end_line: int) -> tuple[int, int]:
    """Convert inclusive 1-indexed line range to byte offsets [lo, hi)."""
    # lo = start of start_line; hi = end of end_line (including its trailing \n).
    lo = 0
    for _ in range(start_line - 1):
        nl = text.find("\n", lo)
        if nl == -1:
            return lo, len(text)
        lo = nl + 1
    hi = lo
    for _ in range(end_line - start_line + 1):
        nl = text.find("\n", hi)
        if nl == -1:
            return lo, len(text)
        hi = nl + 1
    return lo, hi


def _candidate(text: str, start: int, end: int, idx: int) -> dict:
    line_no = _line_of_offset(text, start)
    before_lines = text[:start].splitlines()[-_CTX_LINES:]
    after_lines = text[end:].splitlines()[:_CTX_LINES]
    return {
        "index": idx,
        "line": line_no,
        "before": "\n".join(before_lines),
        "match": text[start:end],
        "after": "\n".join(after_lines),
    }


# ── Validation pipeline ────────────────────────────────────────────────────


@dataclass
class _ValidatedChunk:
    chunk_index: int
    path: str               # original path as passed (for reporting)
    fpath: Path             # resolved
    original: str           # file contents at validation time
    start: int              # match byte offset (inclusive)
    end: int                # match byte offset (exclusive)
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
    """Validate one chunk. Returns (validated, None) or (None, error_dict)."""
    from agent.tools.rules import get_rules

    def err(kind: str, detail: str, **extra) -> dict:
        return {"chunk_index": idx, "kind": kind, "detail": detail, **extra}

    # Required fields
    for field_name in ("path", "anchor", "replacement"):
        if field_name not in chunk:
            return None, err("bad_input", f"missing required field: {field_name}")

    path = chunk["path"]
    anchor = chunk["anchor"]
    replacement = chunk["replacement"]

    if not isinstance(path, str) or not isinstance(anchor, str) or not isinstance(replacement, str):
        return None, err("bad_input", "path/anchor/replacement must be strings")
    if not anchor:
        return None, err("bad_input", "anchor must be non-empty")

    # Resolve + rule check
    try:
        fpath = resolve_fn(path)
    except ValueError as e:
        return None, err("bad_input", str(e))

    rules = get_rules()
    # relative path for rule checks
    try:
        from agent.tools.files import _working_dir  # lazy import to avoid cycle at module load
        rel = str(fpath.relative_to(_working_dir()))
    except Exception:
        rel = path

    allowed, msg = rules.check_write(rel)
    if not allowed:
        return None, err("readonly", msg or f"cannot write: {path}")

    if not fpath.exists():
        return None, err("file_not_found", f"file does not exist: {path}")

    # Cache file contents per call
    if path in file_cache:
        _, original = file_cache[path]
    else:
        original = fpath.read_text(encoding="utf-8", errors="replace")
        file_cache[path] = (fpath, original)

    total_lines = original.count("\n") + (0 if original.endswith("\n") else 1)
    if not original:
        total_lines = 0

    # Size caps
    anchor_lines = _count_lines(anchor)
    repl_lines = _count_lines(replacement)
    cap = edit_cfg.max_chunk_lines
    if cap > 0 and (anchor_lines > cap or repl_lines > cap):
        return None, err(
            "chunk_too_large",
            f"anchor={anchor_lines} replacement={repl_lines} lines exceeds max_chunk_lines={cap}",
            anchor_lines=anchor_lines, replacement_lines=repl_lines, limit=cap,
        )
    frac = edit_cfg.max_file_fraction
    if frac > 0 and total_lines >= 20:
        if anchor_lines / total_lines > frac:
            return None, err(
                "fraction_exceeded",
                f"anchor spans {anchor_lines}/{total_lines} lines (> {frac:.0%}); refusing whole-file replace",
                anchor_lines=anchor_lines, total_lines=total_lines, limit=frac,
            )

    # anchor_sha256 integrity
    sha = chunk.get("anchor_sha256")
    if sha:
        actual = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
        if actual.lower() != str(sha).lower():
            return None, err(
                "anchor_sha_mismatch",
                "anchor_sha256 does not match the anchor you provided; re-read the file",
                expected=sha, actual=actual,
            )

    # Delta expectation checks
    tol = max(0, edit_cfg.line_delta_tolerance)
    if "expect_removed" in chunk and chunk["expect_removed"] is not None:
        try:
            exp = int(chunk["expect_removed"])
        except (TypeError, ValueError):
            return None, err("bad_input", "expect_removed must be int")
        if abs(exp - anchor_lines) > tol:
            return None, err(
                "delta_exceeds_tolerance",
                f"expect_removed={exp} but anchor has {anchor_lines} lines (tolerance ±{tol})",
                expected=exp, actual=anchor_lines, tolerance=tol,
            )
    if "expect_added" in chunk and chunk["expect_added"] is not None:
        try:
            exp = int(chunk["expect_added"])
        except (TypeError, ValueError):
            return None, err("bad_input", "expect_added must be int")
        if abs(exp - repl_lines) > tol:
            return None, err(
                "delta_exceeds_tolerance",
                f"expect_added={exp} but replacement has {repl_lines} lines (tolerance ±{tol})",
                expected=exp, actual=repl_lines, tolerance=tol,
            )

    # range_hint
    lo, hi = 0, len(original)
    if chunk.get("range_hint") is not None:
        rh = chunk["range_hint"]
        if (not isinstance(rh, (list, tuple)) or len(rh) != 2
                or not all(isinstance(x, int) for x in rh)):
            return None, err("range_hint_invalid", "range_hint must be [int, int]")
        sl, el = rh
        if sl < 1 or el < sl or el > max(total_lines, 1):
            return None, err(
                "range_hint_invalid",
                f"range_hint [{sl}, {el}] invalid for file with {total_lines} lines",
                total_lines=total_lines,
            )
        lo, hi = _range_to_offsets(original, sl, el)

    # Match resolution
    mode = (match_override or edit_cfg.match or "exact").lower()
    if mode == "model":
        mode = "exact"  # safety default if config says "model" but no override

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
        return None, err(
            "anchor_ambiguous",
            f"anchor matched {len(spans)} locations; add range_hint or extend anchor for uniqueness",
            match_count=len(spans),
            candidates=cands,
        )

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


# ── Public tool ────────────────────────────────────────────────────────────


def _build_schema() -> dict:
    """Build the tool's JSON schema, hiding knobs unless config opts in."""
    rules = get_rules()
    ec = rules.config.edit

    chunk_props = {
        "path": {"type": "string", "description": "File path to edit."},
        "anchor": {
            "type": "string",
            "description": "Exact text currently in the file. Must match once.",
        },
        "replacement": {
            "type": "string",
            "description": "New text to insert in place of the anchor. Use \"\" to delete.",
        },
        "range_hint": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2, "maxItems": 2,
            "description": "Optional [start_line, end_line], 1-indexed inclusive. Anchor must lie entirely inside.",
        },
        "anchor_sha256": {
            "type": "string",
            "description": "Optional sha256 (hex, lowercase) of the anchor bytes for integrity check.",
        },
        "expect_removed": {
            "type": "integer",
            "description": "Optional self-check: number of lines the anchor spans.",
        },
        "expect_added": {
            "type": "integer",
            "description": "Optional self-check: number of lines in replacement.",
        },
    }

    props: dict = {
        "chunks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "anchor", "replacement"],
                "properties": chunk_props,
            },
            "description": "One or more anchored edits. All applied atomically (default).",
        },
    }

    if ec.match == "model":
        props["match"] = {
            "type": "string",
            "enum": ["exact", "loose"],
            "description": "Match mode: 'exact' (default) or 'loose' (whitespace-tolerant fallback).",
        }
    if ec.on_chunk_fail == "model":
        props["on_chunk_fail"] = {
            "type": "string",
            "enum": ["abort", "skip"],
            "description": "'abort' (default) fails atomically; 'skip' applies good chunks only.",
        }

    return {
        "description": (
            "Edit an existing file by replacing one or more anchored regions. "
            "Always read_file first, then quote the anchor EXACTLY (whitespace matters). "
            "Anchor must match exactly once; use range_hint to disambiguate. "
            "Fails loudly on any mismatch — no silent changes. "
            "Use write_file only to create a new file or fully overwrite one. "
            "Example: chunks=[{'path': 'foo.py', 'anchor': 'def bar():', 'replacement': 'def bar(x):'}]"
        ),
        "parameters": {
            "type": "object",
            "required": ["chunks"],
            "properties": props,
        },
    }


def _register_edit_file() -> None:
    """Register edit_file with a schema reflecting current config."""
    schema = _build_schema()
    register("edit_file", schema)(edit_file)


def edit_file(chunks: list[dict], match: str | None = None, on_chunk_fail: str | None = None) -> dict:
    from agent.tools.files import _resolve, _log_edit, _undo_stack

    rules = get_rules()
    ec = rules.config.edit

    if not isinstance(chunks, list) or not chunks:
        return {"error": "bad_input", "errors": [{"chunk_index": -1, "kind": "bad_input",
                                                  "detail": "chunks must be a non-empty list"}]}

    attempted_paths = list(set(ch.get("path") for ch in chunks if isinstance(ch, dict) and "path" in ch))

    # Gate model-supplied overrides by config
    match_override = match if ec.match == "model" else None
    fail_mode = on_chunk_fail if ec.on_chunk_fail == "model" else None
    fail_mode = (fail_mode or ec.on_chunk_fail or "abort").lower()
    if fail_mode not in ("abort", "skip"):
        fail_mode = "abort"

    file_cache: dict[str, tuple[Path, str]] = {}
    validated: list[_ValidatedChunk] = []
    errors: list[dict] = []

    for i, ch in enumerate(chunks):
        if not isinstance(ch, dict):
            errors.append({"chunk_index": i, "kind": "bad_input", "detail": "chunk must be an object"})
            continue
        v, e = _validate_chunk(i, ch, file_cache, _resolve, ec, match_override)
        if e is not None:
            errors.append(e)
        else:
            validated.append(v)  # type: ignore[arg-type]

    # Overlap check per file among validated chunks
    by_file: dict[str, list[_ValidatedChunk]] = {}
    for v in validated:
        by_file.setdefault(v.path, []).append(v)
    for path, vs in by_file.items():
        vs_sorted = sorted(vs, key=lambda c: c.start)
        for a, b in zip(vs_sorted, vs_sorted[1:]):
            if b.start < a.end:
                errors.append({
                    "chunk_index": b.chunk_index,
                    "kind": "chunks_overlap",
                    "detail": f"chunk {b.chunk_index} overlaps chunk {a.chunk_index} in {path}",
                    "overlaps_with": a.chunk_index,
                })
                # Demote both from validated set
                if b in validated:
                    validated.remove(b)

    # Decide commit
    if errors and fail_mode == "abort":
        _log_edit("edit_file", "<multi>", "atomic_rollback",
                  chunk_count=len(chunks),
                  paths=attempted_paths,
                  error_kinds=[e["kind"] for e in errors])
        return {"error": "atomic_rollback", "errors": errors}

    # Apply: per-file, descending by start offset
    applied: list[dict] = []
    by_file2: dict[str, list[_ValidatedChunk]] = {}
    for v in validated:
        by_file2.setdefault(v.path, []).append(v)

    for path, vs in by_file2.items():
        vs_sorted = sorted(vs, key=lambda c: c.start, reverse=True)
        fpath = vs_sorted[0].fpath
        content = vs_sorted[0].original
        _undo_stack[path] = content
        for v in vs_sorted:
            content = content[:v.start] + v.replacement + content[v.end:]
            applied.append({
                "path": path,
                "chunk_index": v.chunk_index,
                "removed_lines": v.removed_lines,
                "added_lines": v.added_lines,
            })
        # One write size check on the final content
        size_ok, size_msg = rules.check_write_size(content)
        if not size_ok:
            # Roll back in-memory: skip this file
            errors.append({"chunk_index": vs[0].chunk_index, "kind": "write_size_exceeded",
                           "detail": size_msg or "write size limit exceeded", "path": path})
            # Remove applied entries for this file
            applied = [a for a in applied if a["path"] != path]
            del _undo_stack[path]
            continue
        if rules.config.dry_run:
            continue
        fpath.write_text(content, encoding="utf-8")

    outcome = "ok" if not errors else "skip_partial"
    _log_edit("edit_file", "<multi>", outcome,
              chunk_count=len(chunks),
              paths=attempted_paths,
              applied=len(applied),
              error_kinds=[e["kind"] for e in errors] or None)

    result: dict = {"ok": True, "applied": applied}
    if errors:
        result["skipped"] = errors
        result["outcome"] = "skip_partial"
    if rules.config.dry_run:
        result["dry_run"] = True
    return result


# Registration is performed from `load_all_tools` after rules are loaded,
# so the schema reflects the effective `[edit]` config.
