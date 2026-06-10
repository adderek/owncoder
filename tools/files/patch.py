from __future__ import annotations

import os
import tempfile
from pathlib import Path

from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack, _log_edit


def _apply_unified_diff(original: str, patch: str) -> str:
    import subprocess

    with tempfile.NamedTemporaryFile(mode="w", suffix=".orig", delete=False, encoding="utf-8") as f:
        f.write(original)
        orig_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as f:
        f.write(patch)
        patch_path = f.name

    try:
        try:
            result = subprocess.run(
                ["patch", "--dry-run", orig_path, patch_path],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("patch dry-run timed out")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "patch dry-run failed")

        try:
            result = subprocess.run(
                ["patch", orig_path, patch_path],
                capture_output=True, text=True, timeout=15,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("patch apply timed out")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())

        return Path(orig_path).read_text(encoding="utf-8")
    finally:
        for _p in (orig_path, patch_path, orig_path + ".orig"):
            try:
                if os.path.exists(_p):
                    os.unlink(_p)
            except OSError:
                pass


def patch_file(path: str, unified_diff: str) -> dict:
    fpath = _resolve(path)

    rules = get_rules()
    rel = str(fpath.relative_to(_working_dir()))
    allowed, msg = rules.check_write(rel)
    if not allowed:
        return {"error": msg or f"Cannot write to: {path}"}
    lines_ok, lines_msg = rules.check_patch_lines(unified_diff)
    if not lines_ok:
        return {"error": lines_msg}
    if rules.config.dry_run:
        return {"dry_run": True, "path": path, "would_patch": f"{unified_diff.count(chr(10)) + 1} lines"}

    if not fpath.exists():
        return {"error": f"File not found: {path}"}

    original = fpath.read_text(encoding="utf-8", errors="replace")
    _undo_stack[path] = original

    try:
        patched = _apply_unified_diff(original, unified_diff)
    except Exception as e:
        error_msg = str(e)
        if "patch dry-run failed" in error_msg:
            error_msg = "The provided unified diff does not match the file content (dry-run failed). Check your hunk headers, line numbers, and context lines."
        elif "malformed patch" in error_msg:
            error_msg = "The provided diff is malformed. Ensure it is a valid unified diff with correct hunk headers (@@ -L,C +L,C @@)."
        return {"error": f"Patch failed: {error_msg}"}

    fpath.write_text(patched, encoding="utf-8")
    _log_edit("patch_file", path, "ok")
    return {"ok": path}
