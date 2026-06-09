from __future__ import annotations

from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack


@register(
    "write_file",
    {
        "description": "Write new file (creates parent dirs). New files only — use edit_file to modify existing.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write"},
                "content": {"type": "string", "description": "New file content. Escape double quotes inside the string as \\\""},
            },
            "required": ["path", "content"],
        },
    },
)
def write_file(path: str, content: str) -> dict:
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
    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_write": f"{len(content)} bytes"}
    if is_new and rules.config.confirm_create:
        return {"error": f"Creating new files requires confirmation: {path}", "requires_confirm": True}

    fpath.parent.mkdir(parents=True, exist_ok=True)

    if fpath.exists():
        original = fpath.read_text(encoding="utf-8", errors="replace")
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

    # Route through security.fs.safe_open so the final component gets
    # O_NOFOLLOW — a symlink planted at fpath between _resolve and the
    # write must not redirect the write. Fall back to write_text only when
    # the harness is not configured (bare ToolsConfig fixtures).
    try:
        from agent.security import policy as _sec_policy, fs as _sec_fs
        from . import paths as _paths
        # Mirror _resolve's gate: only route through safe_open when the
        # files-layer config is set *and* the security harness is active.
        # Otherwise sec_policy.root may point at a stale root from a prior
        # fixture and safe_resolve would spuriously reject the write.
        if _paths._config is not None and _sec_policy.is_configured():
            with _sec_fs.safe_open(fpath, "w") as f:
                f.write(content)
        else:
            fpath.write_text(content, encoding="utf-8")
    except ImportError:
        fpath.write_text(content, encoding="utf-8")

    if not is_new and fpath.suffix == ".py":
        try:
            import ast
            ast.parse(content)
        except SyntaxError as e:
            return {"error": f"File written but has syntax error: {e}. Use undo_file to revert.", "path": path}

    return {"ok": path, "diff": diff_summary}
