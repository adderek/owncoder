"""Mid-turn progress guards: repeated-read auto-advance, repeated-edit hints,
and the loop-guard stop note.

These are the parts of the turn loop that watch *tool results* for signs the
model has stopped making progress and rewrite the result to unstick it. They are
pure functions over one tool call and the turn's counter dicts (which they mutate
in place), so they can be tested without running a turn.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


# Tool names that mutate files on disk — a successful call marks the turn
# "dirty" for the post-edit verify hook (see run_turn / VerifyConfig).
MUTATING_TOOLS = {"edit_file", "write_file", "patch_file", "replace_text", "replace_symbol", "undo_file"}


def loop_guard_stop_note(summary: str, triggered: list[tuple[str, str, int, str]]) -> str:
    """Build a tool-specific recovery message for loop-guard stops that persists in history."""
    tool_names = {n for n, _, _, _ in triggered}
    parts: list[str] = [f"[loop guard: stopped after repeated tool calls ({summary})."]

    if "edit_file" in tool_names:
        paths = []
        for n, _, _, args_json in triggered:
            if n == "edit_file":
                try:
                    p = json.loads(args_json).get("path", "")
                    if p and p not in paths:
                        paths.append(p)
                except Exception:
                    pass
        if paths:
            parts.append(f" The anchor used for {', '.join(paths)} was not found repeatedly.")
            parts.append(" To recover: re-read the file to get current content and correct anchors, then retry the edit.")
        else:
            parts.append(" edit_file anchor was not found repeatedly. Re-read the target file for fresh anchors before retrying.")

    if "read_file" in tool_names:
        parts.append(" The same file range was read repeatedly without progress. Use search_files to locate the target or specify a different start_line/end_line range.")

    if not tool_names & {"edit_file", "read_file"}:
        parts.append(" Redirect: describe what you are trying to accomplish or ask the user for guidance.")

    parts.append("]")
    return "".join(parts)


def patch_read_file_result(tc, result: str, read_path_counts: dict,
                           warn_threshold: int, stop_threshold: int,
                           read_advance: dict | None = None) -> tuple[str, str | None]:
    """Track repeated read_file of the same range; auto-advance, warn, then stop.

    Mutates *read_path_counts* (and *read_advance*) in place. On a repeated read of
    the same range the identical window is replaced with the *next* slice of the
    file so an instruction-ignoring model is forced to make progress instead of
    re-reading the same head. Returns (result, stop_note): stop_note is non-None
    only at the hard ceiling, when even auto-advance failed to unstick the model.
    """
    try:
        a = json.loads(tc.function.arguments or "{}")
        rpath = str(a.get("path", ""))
        if not rpath:
            return result, None
        # Key on (path, start_line, end_line) so reading different sections of
        # the same large file doesn't trigger the guard.
        rkey = (rpath, a.get("start_line"), a.get("end_line"))
        read_path_counts[rkey] = read_path_counts.get(rkey, 0) + 1
        count = read_path_counts[rkey]

        # Auto-advance: the model re-read the SAME range without acting on it.
        # Serving the identical window again just feeds the loop, so return the
        # NEXT lines of the file instead. read_file already advertises
        # "read offset=N for more"; this enforces it behaviourally.
        auto_advanced = False
        if read_advance is not None and count >= 2:
            try:
                from agent.tools.files.read import read_file as _rf, READ_WINDOW_LINES
                try:
                    total = int(json.loads(result).get("metadata", {}).get("total_lines") or 0)
                except Exception:
                    total = 0
                win = READ_WINDOW_LINES
                # The advance cursor is keyed per path (not per range like the
                # counts): once a file is stuck, every repeat pages forward
                # regardless of which range the model keeps asking for.
                nxt = read_advance.get(rpath)
                if nxt is None:
                    sl, el = a.get("start_line"), a.get("end_line")
                    base_end = el if el else ((sl + win - 1) if sl else win)
                    nxt = (base_end or win) + 1
                if total and nxt > total:
                    note = (
                        f"[loop guard: all {total} lines of '{rpath}' have now been shown "
                        f"across {count} reads. Stop re-reading — make your change with "
                        f"edit_file, or use search_files for a specific anchor.]"
                    )
                    result = json.dumps({"content": note, "end_of_file": True,
                                         "metadata": {"total_lines": total}})
                    auto_advanced = True
                else:
                    adv = _rf(rpath, start_line=nxt, end_line=nxt + win - 1)
                    if isinstance(adv, dict) and not adv.get("error"):
                        adv["_auto_advanced"] = (
                            f"[loop-guard] You re-read '{rpath}' without acting, so this is the "
                            f"NEXT block (lines {nxt}+) — the offset advances on each repeat. "
                            f"Use search_files or edit_file once you have the anchor; do not "
                            f"re-request the same range."
                        )
                        read_advance[rpath] = nxt + win
                        result = json.dumps(adv)
                        auto_advanced = True
            except Exception:
                pass

        if count >= stop_threshold:
            logger.warning("loop_guard: read_file path '%s' range %s-%s count %d >= stop threshold",
                           rpath, a.get("start_line"), a.get("end_line"), count)
            return result, (
                f"[loop guard: '{rpath}' same range read {count}× this turn without progress. "
                f"Stop re-reading — use search_files to find a specific anchor, "
                f"or report what you need and ask the user for guidance.]"
            )
        # When auto-advance replaced the result it already carries its own
        # note about a *different* range — adding the "same range read N×"
        # warning on top would contradict it and confuse weak models.
        if count >= warn_threshold and not auto_advanced:
            try:
                r_parsed = json.loads(result)
            except Exception:
                r_parsed = {}
            if isinstance(r_parsed, dict):
                r_parsed["_loop_warning"] = (
                    f"[loop-guard] '{rpath}' same range read {count}× this turn. "
                    "If previous reads didn't give you the anchor, use search_files or "
                    "specify start_line/end_line to target a different section. "
                    "Do not re-read the same range again."
                )
                result = json.dumps(r_parsed)
    except Exception:
        pass
    return result, None


def patch_edit_file_result(tc, result: str, read_path_counts: dict,
                           edit_file_fails: dict, fail_threshold: int,
                           read_advance: dict | None = None) -> str:
    """On successful edit clear that file's read counters; on repeated
    anchor_not_found inject a structure hint. Mutates both count dicts in place."""
    try:
        e_parsed = json.loads(result)
        if not isinstance(e_parsed, dict):
            return result
        a = json.loads(tc.function.arguments or "{}")
        e_path = str(a.get("path", "") or "")
        if not e_parsed.get("error"):
            # Successful edit — drop read-path counters for this file so reads
            # after the edit don't accumulate against the guard, and reset the
            # auto-advance offset so a fresh read starts at the top of the file.
            if e_path:
                for k in [k for k in read_path_counts if k[0] == e_path]:
                    del read_path_counts[k]
                if read_advance is not None:
                    read_advance.pop(e_path, None)
            return result
        if e_parsed.get("error") == "atomic_rollback":
            for e_chunk in e_parsed.get("errors", []):
                if isinstance(e_chunk, dict) and e_chunk.get("kind") == "anchor_not_found":
                    fail_key = f"{e_path}:{e_chunk.get('chunk_index', 0)}"
                    edit_file_fails[fail_key] = edit_file_fails.get(fail_key, 0) + 1
                    if edit_file_fails[fail_key] >= fail_threshold:
                        structure = e_chunk.get("file_structure") or []
                        def_names = [s["name"] for s in structure if s["kind"] == "def"]
                        class_names = [s["name"] for s in structure if s["kind"] == "class"]
                        hint_parts = []
                        if class_names:
                            hint_parts.append(f"classes: {', '.join(class_names[:5])}")
                        if def_names:
                            hint_parts.append(f"methods: {', '.join(def_names[:10])}")
                        hint = (
                            f"[loop-guard] The anchor you used is not in this file "
                            f"({edit_file_fails[fail_key]}×). "
                        )
                        if hint_parts:
                            hint += "File has " + "; ".join(hint_parts) + ". "
                        hint += "Search the file with search_files or read different sections to find the right anchor."
                        e_parsed["_error_hint"] = hint
                        result = json.dumps(e_parsed)
                    break  # only process first anchor_not_found per call
    except Exception:
        pass
    return result


def observe_tool_calls(loop_detector, tool_calls) -> list[tuple[str, str, int, str]]:
    """Feed a batch of tool calls to the loop detector.

    Returns the list of (tool_name, signature, count, arguments_json) entries that
    tripped either the per-signature repeat threshold or the per-name call cap.
    Empty list means the batch looks like progress.
    """
    from .loop_detector import LoopDetector

    triggered: list[tuple[str, str, int, str]] = []
    for tc in tool_calls:
        sig = LoopDetector.signature(tc.function.name, tc.function.arguments)
        cnt = loop_detector.observe(sig)
        if loop_detector.triggered(sig, cnt):
            triggered.append((tc.function.name, sig, cnt, tc.function.arguments or "{}"))
        name_cnt = loop_detector.observe_name(tc.function.name)
        if loop_detector.name_capped(tc.function.name, name_cnt):
            triggered.append((tc.function.name, f"name:{tc.function.name}", name_cnt, tc.function.arguments or "{}"))
    return triggered
