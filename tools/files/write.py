from __future__ import annotations

from agent.core import revisions
from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack, _log_edit, _read_text, _write_text


@register(
    "write_file",
    {
        "description": "Write new file (creates parent dirs). New files only — use edit_file to modify existing.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "New file content. Escape double quotes inside the string as \\\""},
                "expect_rev": {"type": "string", "description": revisions.ARG_DESCRIPTION},
            },
            "required": ["path", "content"],
        },
    },
)
def write_file(path: str, content: str, expect_rev: str | None = None,
               _confirmed: bool = False) -> dict:
    import difflib

    fpath = _resolve(path)

    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    is_new = not fpath.exists()
    allowed, msg = rules.check_write(rel, is_new=is_new)
    if not allowed:
        return {"error": msg or f"Cannot write to: {path}"}
    size_ok, size_msg = rules.check_write_size(content)
    if not size_ok:
        return {"error": size_msg}
    rev_error = revisions.check(path, fpath, expect_rev)
    if rev_error is not None:
        _log_edit("write_file", path, "rev_mismatch", expect_rev=expect_rev)
        return rev_error
    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_write": f"{len(content)} bytes"}
    if is_new and rules.config.confirm_create and not _confirmed:
        return {"error": f"Creating new files requires confirmation: {path}", "requires_confirm": True}

    fpath.parent.mkdir(parents=True, exist_ok=True)

    replaced_lines = 0
    if fpath.exists():
        original = _read_text(fpath)
        replaced_lines = len(original.splitlines())
        _undo_stack[path] = original
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
                n=3,
            )
        )
        diff_summary = "".join(diff_lines[:60])
        if len(diff_lines) > 60:
            diff_summary += f"\n... ({len(diff_lines) - 60} more diff lines)"
    else:
        diff_summary = f"(new file, {len(content.splitlines())} lines)"

    # _write_text goes through security.fs.safe_open when the harness is live:
    # every path component is opened O_NOFOLLOW, so a symlink planted between
    # _resolve and the write (by a concurrent sandboxed command) fails it.
    _write_text(fpath, content)

    if not is_new and fpath.suffix == ".py":
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            _log_edit("write_file", path, "syntax_error")
            return {"error": f"File written but has syntax error: {e}. Use undo_file to revert.", "path": path}

    # Count the new file only now that it was actually written (not on the
    # earlier check_write gate, which also fires for dry-run/aborted writes).
    if is_new:
        rules.note_file_created()

    _log_edit("write_file", path, "ok", expect_rev=expect_rev)
    result = {"ok": path, "diff": diff_summary}
    if replaced_lines:
        # How much existing content this call replaced wholesale — read by the
        # tool-hint layer to suggest edit_file for large in-place rewrites.
        result["replaced_lines"] = replaced_lines
    return result
