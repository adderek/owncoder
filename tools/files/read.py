from __future__ import annotations

from pathlib import Path

from agent.tools import register
from agent.tools.rules import get_rules
from .paths import _resolve, _working_dir, _undo_stack

# Default window (lines) served for an unbounded read of a large file. The
# loop-guard auto-advance in core/turn.py pages through files in this same
# unit — keep the two in sync via this constant.
READ_WINDOW_LINES = 200

# Landmarks listed alongside a truncated read. Enough to cover a big module,
# small enough that the map never competes with the content for attention.
_OUTLINE_ENTRIES = 40


def _read_limits() -> tuple[int, int, float, int]:
    """(max_lines, max_chars, headroom_fraction, hard_max_chars) for unbounded reads."""
    from agent.config.models import ToolsConfig as _D
    try:
        from .paths import _config as _files_config
        t = _files_config.tools if _files_config else None
    except Exception:
        t = None
    return (
        int(getattr(t, "read_full_max_lines", _D.read_full_max_lines)),
        max(1, int(getattr(t, "read_full_max_chars", _D.read_full_max_chars))),
        float(getattr(t, "read_full_headroom_fraction", _D.read_full_headroom_fraction)),
        int(getattr(t, "read_full_hard_max_chars", _D.read_full_hard_max_chars)),
    )


def _fits_whole(chars: int, total: int, snap) -> bool:
    """Serve an unbounded read whole? Small files always; bigger ones only while
    the context is mostly empty, so reading the whole file once is cheaper than
    the ranged reads and greps that would otherwise add up to more than it."""
    max_lines, max_chars, fraction, hard_max = _read_limits()
    if total <= max_lines and chars <= max_chars:
        return True
    if snap is None or snap.window <= 0 or fraction <= 0 or chars > hard_max:
        return False
    from agent.core import context_state
    return context_state.estimate_tokens(chars) <= snap.headroom * fraction


def _window(lines: list[str], start: int, end: int, max_chars: int,
            char_offset: int = 0) -> tuple[int, str, tuple[int, int] | None]:
    """Numbered lines[start:end] (0-based, end exclusive), bounded by characters.
    *char_offset* skips the beginning of the first line (continuing a cut line).
    Returns (1-based number of the last line shown, text, cut), where cut is
    (column the line was cut at, full line length) or None."""
    out: list[str] = []
    used = 0
    last = start
    for i in range(start, end):
        line = lines[i]
        col = char_offset if i == start else 0
        piece = line[col:]
        room = max_chars - used
        if room <= 0:
            break
        last = i + 1
        if len(piece) > room:
            out.append(f"{i + 1}:{piece[:room]} [... line cut at char {col + room} of {len(line)}]")
            return last, "\n".join(out), (col + room, len(line))
        out.append(f"{i + 1}:{piece}")
        used += len(piece) + 1
    return last, "\n".join(out), None


def _head_window(lines: list[str], max_lines: int, max_chars: int) -> tuple[int, str, tuple[int, int] | None]:
    """First lines of a file, bounded by line count AND characters."""
    return _window(lines, 0, min(len(lines), max_lines), max_chars)


def _range_max_chars() -> int:
    from agent.config.models import ToolsConfig as _D
    try:
        from .paths import _config as _files_config
        t = _files_config.tools if _files_config else None
    except Exception:
        t = None
    return max(1, int(getattr(t, "read_range_max_chars", _D.read_range_max_chars)))


def _continue_note(last: int, total: int, cut: tuple[int, int] | None) -> str:
    """Where to pick up after a truncated read, in the tool's own parameters."""
    nxt = f"Next unread line: {last + 1} of {total}." if last < total else ""
    if cut:
        return (f"Line {last} cut at char {cut[0]} of {cut[1]}: "
                f"start_line={last}, char_offset={cut[0]} continues it. " + nxt).rstrip()
    return nxt


def _format_size(bytes_val: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.0f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.0f}TB"


def _with_rev(result: dict, path: str, fpath: Path, text: str) -> dict:
    """Tag a read result with the file's revision, for quoting back as expect_rev."""
    from agent.core import revisions
    return revisions.annotate_read(result, path, fpath, content=text)


def _count_lines_fast(fpath: Path) -> int:
    """Count lines without reading full file into memory."""
    with open(fpath, "rb") as f:
        return sum(1 for _ in f)


def _build_gitignore_spec(base):
    try:
        import pathspec  # type: ignore
    except ImportError:
        return None
    gi = base / ".gitignore"
    if not gi.exists():
        return None
    lines = [
        line.strip()
        for line in gi.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return pathspec.PathSpec.from_lines("gitignore", lines)


@register(
    "read_file",
    {
        "description": (
            "Read file contents, optionally a line range. "
            "No range: small file served whole (read it once, don't slice it); "
            "large file → head + outline, then read the range you need. "
            "Range auto-clamps; end_of_file=true signals end of file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)"},
                "char_offset": {"type": "integer", "description": (
                    "Skip this many characters of start_line — continues a line a "
                    "truncated read cut (the note names the value)")},
            },
            "required": ["path"],
        },
    },
)
def read_file(path: str, start_line: int | None = None, end_line: int | None = None,
              char_offset: int | None = None) -> dict:
    fpath = _resolve(path)
    if char_offset and start_line is None:
        start_line = 1

    rel = str(fpath.relative_to(_working_dir()))
    allowed, _ = get_rules().check_read(rel)
    if not allowed:
        return {"error": f"File not found: {path}"}

    if not fpath.exists():
        return {"error": f"File not found: {path}", "resolved": str(fpath)}
    if not fpath.is_file():
        return {"error": f"Not a file: {path}", "resolved": str(fpath)}

    try:
        text = fpath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"error": str(e)}

    lines = text.splitlines()
    total = len(lines)
    filesize = fpath.stat().st_size

    def _make_header(sl_show: int, el_show: int, clamped: bool = False, past_eof: bool = False) -> str:
        size_str = f" · {_format_size(filesize)}"
        if past_eof:
            return (
                f"[{fpath.name} · {total} lines{size_str} · "
                f"END OF FILE — requested line {sl_show} beyond file's {total} lines. "
                f"Last {el_show - sl_show + 1} lines shown for reference below.]"
            )
        if clamped:
            return (
                f"[{fpath.name} · {total} lines{size_str} · "
                f"showing lines {sl_show}-{el_show}"
                f" · start_line={el_show + 1} for more · "
                f"note: requested up to line {end_line} but file has {total}]"
            )
        if sl_show == 1 and el_show >= total:
            return f"[{fpath.name} · {total} lines{size_str}]"
        return (
            f"[{fpath.name} · {total} lines{size_str} · "
            f"showing lines {sl_show}-{el_show} · "
            f"start_line={el_show + 1} for more]"
        )

    # Repeatedly reading one whole file is symbol-hunting, not reading. Past the
    # configured limit the unbounded read serves the map only; ranged reads keep
    # working, so the model can still fetch the lines the outline names.
    wall = 0
    try:
        from .paths import _config as _files_config
        wall = int(getattr(_files_config.tools, "outline_only_after_reads", 0)) if _files_config else 0
    except Exception:
        wall = 0
    if wall and start_line is None and end_line is None:
        from agent.core.tool_hints import read_count
        seen = read_count(path)
        if seen >= wall:
            from .outline import outline as _outline, format_outline as _fmt
            entries = _outline(text, max_entries=_OUTLINE_ENTRIES, filename=fpath.name)
            return _with_rev({
                "content": (
                    f"[{fpath.name} · {total} lines · read {seen}× this session — "
                    f"outline only. Read a range (start_line/end_line) for content, "
                    f"or call find_symbol/grep_code to locate what you are looking for.]\n"
                    + (_fmt(entries) if entries else "(no landmarks found)")
                ),
                "metadata": {"total_lines": total, "file_size": filesize,
                             "outline": entries, "outline_only": True},
            }, path, fpath, text)

    # Price the read against the remaining headroom. A file that does not fit
    # is worse than useless: it lands, pushes the turn over the compaction
    # threshold, and is summarised away — after which the model reads it again.
    snap = None
    try:
        from agent.core import context_state
        snap = context_state.current()
    except Exception:
        snap = None
    if (snap is not None and snap.window > 0 and start_line is None and end_line is None
            and not snap.would_survive(context_state.estimate_tokens(filesize))):
        from .outline import outline as _outline, format_outline as _fmt
        cost = context_state.estimate_tokens(filesize)
        entries = _outline(text, max_entries=_OUTLINE_ENTRIES, filename=fpath.name)
        window_end, head, cut = _head_window(lines, READ_WINDOW_LINES, _read_limits()[1])
        return _with_rev({
            "content": (
                f"[{fpath.name} · {total} lines · TRUNCATED: ~{cost} tokens, but only "
                f"~{snap.headroom} tokens of context remain before compaction. Serving lines "
                f"1-{window_end} and the outline instead — reading it whole would be summarised "
                f"away and lost. {_continue_note(window_end, total, cut)} "
                f"Read the range you need, or call find_symbol.]\n" + head
                + (("\n\n[outline of the whole file]\n" + _fmt(entries)) if entries else "")
            ),
            "truncated": True,
            "metadata": {"total_lines": total, "file_size": filesize, "outline": entries,
                         "budget_limited": True, "estimated_tokens": cost,
                         "headroom_tokens": snap.headroom,
                         "served_lines": window_end, "next_start_line": window_end + 1},
        }, path, fpath, text)

    if start_line is None and end_line is None and not _fits_whole(len(text), total, snap):
        max_lines, max_chars = _read_limits()[:2]
        served, numbered, cut = _head_window(lines, READ_WINDOW_LINES, max_chars)
        # Only the first window is served, so hand over a map of the rest:
        # without it the model pages blindly (read 1-200, 300-400, 400-450...)
        # hunting for a landmark whose line number we already know.
        from .outline import outline as _outline, format_outline as _fmt
        entries = _outline(text, max_entries=_OUTLINE_ENTRIES, filename=fpath.name)
        # Say why and where to continue: a bare "lines 1-12" on a 50-line file
        # reads as a short file, and a cut single line looked complete.
        over = [f"{total} lines > {max_lines}"] if total > max_lines else []
        if len(text) > max_chars:
            over.append(f"{len(text)} chars > {max_chars}")
        body = (
            f"[{fpath.name} · {total} lines · {_format_size(filesize)} · TRUNCATED: "
            f"too big to serve whole ({', '.join(over)}); showing lines 1-{served}. "
            f"{_continue_note(served, total, cut)} "
            f"Read the range you need with start_line/end_line.]\n" + numbered
        )
        if entries:
            body += (
                "\n\n[outline of the whole file — read the range you need, "
                "do not page through it]\n" + _fmt(entries)
            )
        return _with_rev({
            "content": body,
            "truncated": True,
            "metadata": {"total_lines": total, "file_size": filesize,
                         "served_lines": served, "next_start_line": served + 1,
                         "outline": entries},
        }, path, fpath, text)

    past_eof = False
    clamped = False

    if start_line is not None and start_line > total:
        # Model requested past EOF — show last 50 lines as reference
        sl = max(0, total - 50)
        el = total
        past_eof = True
    else:
        sl = max(0, (start_line or 1) - 1)
        clamped = end_line is not None and end_line > total
        el = min(end_line if end_line else total, total)

    # Ranged reads are capped by characters too: asking for "line 1" of a
    # minified bundle must not return 50k tokens.
    range_max = _range_max_chars()
    offset = 0 if past_eof else max(0, int(char_offset or 0))
    last, numbered, cut = _window(lines, sl, el, range_max, offset)
    short = cut is not None or last < el
    header = _make_header(sl + 1, last if short else el, clamped and not short, past_eof)
    if short:
        header = (header[:-1] + f" · TRUNCATED: range exceeds {range_max} chars. "
                  + _continue_note(last, total, cut) + "]")

    result: dict = {
        "content": header + "\n" + numbered,
        "metadata": {"total_lines": total, "file_size": filesize},
    }
    if short:
        result["truncated"] = True
        result["metadata"].update(served_lines=last, next_start_line=last + 1)
    if past_eof:
        result["end_of_file"] = True

    if fpath.suffix == ".py":
        try:
            import ast
            ast.parse(text)
        except SyntaxError as e:
            if path in _undo_stack:
                result["warning"] = (
                    f"File has syntax error: {e}. It was recently modified. Use undo_file to revert."
                )

    return _with_rev(result, path, fpath, text)


@register(
    "list_files",
    {
        "description": (
            "List files in directory, .gitignore-aware. Returns relative paths + sizes. "
            "Capped at max_results (default 500) — narrow with pattern (e.g. 'src/**/*.py'). "
            "include_lines=true adds line counts (slower)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list (default: working dir)", "default": "."},
                "pattern": {"type": "string", "description": "Glob pattern", "default": "**/*"},
                "ignore_patterns": {"type": "array", "items": {"type": "string"}, "description": "Patterns to ignore"},
                "max_results": {"type": "integer", "description": "Max entries to return (default 500). On overflow, a directory-grouped summary is returned instead of paths."},
                "include_lines": {"type": "boolean", "description": "Include line count per file (default: false)", "default": False},
            },
            "required": [],
        },
    },
)
def list_files(
    path: str = ".",
    pattern: str = "**/*",
    ignore_patterns: list[str] | None = None,
    max_results: int = 500,
    include_lines: bool = False,
) -> dict:
    import fnmatch

    base = _resolve(path)
    if not base.is_dir():
        return {"error": f"Not a directory: {path}"}

    default_ignore = {".git", "__pycache__", "node_modules", "*.pyc", "build", "dist", ".agent", "graphify-out"}
    all_ignore = default_ignore | set(ignore_patterns or [])

    gitignore_spec = _build_gitignore_spec(base)

    cap = max(1, int(max_results))
    rules = get_rules()
    results: list[dict] = []
    dir_counts: dict[str, int] = {}
    total_kept = 0

    for fpath in sorted(base.glob(pattern)):
        if not fpath.is_file():
            continue
        rel = str(fpath.relative_to(base))
        parts = Path(rel).parts

        skip = False
        for part in parts:
            for pat in all_ignore:
                if fnmatch.fnmatch(part, pat):
                    skip = True
                    break
            if skip:
                break

        if not skip and gitignore_spec is not None:
            skip = gitignore_spec.match_file(rel)

        if not skip and rules.ignore.matches(rel):
            skip = True

        if skip:
            continue

        total_kept += 1
        top = parts[0] if len(parts) > 1 else "."
        dir_counts[top] = dir_counts.get(top, 0) + 1
        if len(results) < cap:
            stat = fpath.stat()
            entry = {"path": rel, "size": stat.st_size}
            if include_lines:
                entry["lines"] = _count_lines_fast(fpath)
            results.append(entry)

    if total_kept > cap:
        summary = sorted(
            ({"dir": d, "count": n} for d, n in dir_counts.items()),
            key=lambda x: -x["count"],
        )
        return {
            "truncated": True,
            "total": total_kept,
            "returned": 0,
            "by_top_dir": summary,
            "hint": (
                f"{total_kept} files matched (cap={cap}). "
                "Re-call with a narrower `pattern` (e.g. 'src/**/*.py') or a deeper `path`, "
                "or raise `max_results` if you really need everything."
            ),
        }

    return {"files": results, "count": len(results)}
