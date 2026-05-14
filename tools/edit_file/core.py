from __future__ import annotations

from pathlib import Path

from agent.tools.rules import get_rules
from .validator import _ValidatedChunk, _validate_chunk
from .schema import _build_schema, _register_edit_file  # noqa: F401


def edit_file(
    chunks: list[dict] | None = None,
    path: str | None = None,
    anchor: str | None = None,
    replacement: str | None = None,
    match_mode: str | None = None,
    on_chunk_fail: str | None = None,
) -> dict:
    from agent.tools.files import _resolve, _log_edit, _undo_stack

    rules = get_rules()
    ec = rules.config.edit

    # Accept flat path+anchor+replacement as alternative to chunks
    if not chunks and path and anchor and replacement is not None:
        chunks = [{"path": path, "anchor": anchor, "replacement": replacement}]

    # Detect empty/missing args — models often call edit_file({}) for new files
    if not isinstance(chunks, list) or not chunks:
        return {
            "error": "empty_args",
            "hint": "edit_file modifies EXISTING files. To CREATE a new file use write_file(path, content). "
                    "edit_file requires one or more anchored edits: "
                    "chunks=[{'path': '...', 'anchor': '...', 'replacement': '...'}] "
                    "or flat path + anchor + replacement.",
        }
    
    # Apply top-level range_hint into chunks that lack one.
    # (path and replacement are now explicit params, handled by the flat-args path above)
    # range_hint is left for a future signature expansion.
    top_range_hint = None  # placeholder for future kwargs.pop("range_hint", None)

    attempted_paths = list(
        set(ch.get("path") for ch in chunks if isinstance(ch, dict) and "path" in ch)
    )

    match_override = match_mode if ec.match == "model" else None
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
                if b in validated:
                    validated.remove(b)

    if errors and fail_mode == "abort":
        _log_edit("edit_file", "<multi>", "atomic_rollback", chunk_count=len(chunks), paths=attempted_paths, error_kinds=[e["kind"] for e in errors])
        return {"error": "atomic_rollback", "errors": errors}

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
            content = content[: v.start] + v.replacement + content[v.end :]
            entry: dict = {"path": path, "chunk_index": v.chunk_index, "removed_lines": v.removed_lines, "added_lines": v.added_lines}
            if v.auto_unescaped:
                entry["auto_unescaped"] = True
            applied.append(entry)
        size_ok, size_msg = rules.check_write_size(content)
        if not size_ok:
            errors.append({"chunk_index": vs[0].chunk_index, "kind": "write_size_exceeded", "detail": size_msg or "write size limit exceeded", "path": path})
            applied = [a for a in applied if a["path"] != path]
            del _undo_stack[path]
            continue
        if rules.config.dry_run:
            continue
        fpath.write_text(content, encoding="utf-8")

    outcome = "ok" if not errors else "skip_partial"
    _log_edit("edit_file", "<multi>", outcome, chunk_count=len(chunks), paths=attempted_paths, applied=len(applied), error_kinds=[e["kind"] for e in errors] or None)

    result: dict = {"ok": True, "applied": applied}
    if errors:
        result["skipped"] = errors
        result["outcome"] = "skip_partial"
    if rules.config.dry_run:
        result["dry_run"] = True
    return result
