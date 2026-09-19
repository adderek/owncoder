from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .streaming import _is_narrating_tool_use, _has_pseudo_tool_tag

logger = logging.getLogger(__name__)

_EXTRACT_SHRINK_RATIO = 0.25
# Fraction of lines that may start with markdown bullets before we refuse the write.
_PROSE_BULLET_THRESHOLD = 0.40

_FILE_RE = re.compile(
    r"(?<![\w.\-/])(/?[a-zA-Z0-9./\-_]+\.(?:sh|bash|py|js|mjs|cjs|ts|jsx|tsx|go|rs|java|kt|c|cpp|h|hpp|rb|toml|yaml|yml|json|md|txt|html|htm|css))\b"
)

# URLs and HTML/markdown link targets contain path-like text that _FILE_RE
# happily matches (e.g. "cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"
# out of a <script src=...> tag). Blank them out before scanning for filenames.
_URLISH_RE = re.compile(
    r"""(?:[a-zA-Z][a-zA-Z0-9+.\-]*://\S+)"""
    r"""|(?:\b(?:src|href|url|action|from|import)\s*=\s*\\?["'][^"'\\]*)""",
    re.IGNORECASE,
)

# "example.com/x/y.js" — a hostname, not a path we should ever write to.
_HOSTNAME_HEAD_RE = re.compile(
    r"^(?:www\.)?[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)+/", re.IGNORECASE
)


def _mask_urls(text: str) -> str:
    """Replace URL/attribute-value spans with spaces, preserving offsets."""
    return _URLISH_RE.sub(lambda m: " " * len(m.group(0)), text)


def _find_filenames(text: str) -> list[str]:
    """_FILE_RE hits in *text*, with URL-borne and hostname-headed ones dropped."""
    return [c for c in _FILE_RE.findall(_mask_urls(text)) if not _HOSTNAME_HEAD_RE.match(c)]


def _merge_consecutive_assistants(messages: list[dict]) -> list[dict]:
    """Merge ALL consecutive assistant messages anywhere in the list.
    
    Some APIs (Qwen 1M, DeepSeek) reject any pair of consecutive assistant
    messages, not just at the end. This scans the full list and merges every
    adjacent pair of assistant messages.
    
    Handles three cases:
    1. Neither message has tool_calls — merge content.
    2. One message has tool_calls — merge content into the one with tool_calls,
       preserving tool_calls and reasoning_content from that message.
    3. Both have tool_calls — merge content and combine both tool_calls lists.
    """
    if len(messages) < 2:
        return messages
    out: list[dict] = []
    for m in messages:
        if out and out[-1].get("role") == "assistant" and m.get("role") == "assistant":
            a, b = out[-1], m
            a_tc = a.get("tool_calls")
            b_tc = b.get("tool_calls")

            a_content = a.get("content") or ""
            b_content = b.get("content") or ""
            merged_content = a_content + ("\n\n" if a_content and b_content else "") + b_content

            # Start from `a` so internal metadata survives the merge. This runs on
            # BOTH API messages (reasoning under `reasoning_content`, no _-keys) and
            # persisted history (reasoning under `_reasoning_content`, plus side-log
            # refs in `_tool_refs`); building a fresh dict here dropped both, losing
            # tool-result links and prior reasoning across turns.
            merged: dict = {**a, "role": "assistant", "content": merged_content}
            merged.pop("tool_calls", None)
            if a_tc and b_tc:
                merged["tool_calls"] = a_tc + b_tc
            elif a_tc:
                merged["tool_calls"] = a_tc
            elif b_tc:
                merged["tool_calls"] = b_tc

            # Combine side-log references from both messages.
            refs = (a.get("_tool_refs") or []) + (b.get("_tool_refs") or [])
            if refs:
                merged["_tool_refs"] = refs

            # Reasoning may live under either key; with tool_calls keep the longer,
            # otherwise concatenate (matches the prior per-case behaviour).
            for key in ("reasoning_content", "_reasoning_content"):
                ra = a.get(key) or ""
                rb = b.get(key) or ""
                if a_tc or b_tc:
                    rc = ra if len(ra) >= len(rb) else rb
                else:
                    rc = ra + rb
                if rc:
                    merged[key] = rc
                else:
                    merged.pop(key, None)

            out[-1] = merged
        else:
            out.append(m)
    return out


def _short_repr(v, limit: int = 40) -> str:
    """repr() capped at `limit` chars without splitting string quotes.

    A blindly sliced repr like `'https://x.pl/wiadomosc/202` (closing quote
    lost) lands in the compaction summary lines; models imitate whatever those
    look like, and _parse_agent_exec_args then splices the next `key=` into the
    value. Strings are shortened before repr so quoting stays balanced, and the
    `…` marker makes truncation visible — and rejectable — in imitations.
    """
    r = repr(v)
    if len(r) <= limit:
        return r
    if isinstance(v, str):
        return repr(v[: max(1, limit - 4)] + "…")
    return r[: limit - 1] + "…"


def _summary_safe(v) -> str:
    """One-line, tag-free rendering of a value for a compaction summary.

    `<` is the only character that has to go: left alone, a result containing
    one revives the very tag-shaped text these summaries exist to avoid, and
    the unexecuted-tag detectors would fire on replayed history. Newlines are
    folded so one collapsed call stays one line, which is what
    _TOOL_SUMMARY_RE and the http replay both key on.
    """
    return " ".join(str(v).split()).replace("<", "&lt;")


def _tool_summary_line(name: str, args: str, result: str) -> str:
    """Render one executed tool call for the compacted history.

    Deliberately NOT `<agent_exec .../>` syntax. Models imitate whatever shape
    they see in history, and the tag form was imitable as a call: a copied tag
    carried `…`-truncated args, so _parse_agent_exec_args rejected every value,
    nothing executed, and the fabricated result the model wrote inside the tag
    read as fact. This form cannot be mistaken for a call by any parser — and
    an imitation of it is caught by _has_fake_tool_summary instead.
    """
    return f"[tool] {name}({_summary_safe(args)}) → {_summary_safe(result)}".rstrip()


def _collapse_tool_rounds(
    messages: list[dict],
    result_preview: int = 200,
    side_log=None,
    turn_id: int | None = None,
) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_calls = m["tool_calls"]
            j = i + 1
            result_msgs: list[dict] = []
            while j < len(messages) and messages[j].get("role") == "tool":
                result_msgs.append(messages[j])
                j += 1

            exec_parts: list[str] = []
            refs: list[int] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_name = tc.get("function", {}).get("name", "?")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    t_args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    t_arg_str = ", ".join(f"{k}={_short_repr(v)}" for k, v in list(t_args.items())[:2])
                except Exception:
                    t_arg_str = str(args_raw)[:60]

                raw_result = ""
                result_content = ""
                for r in result_msgs:
                    if r.get("tool_call_id") == tc.get("id"):
                        raw = r.get("content", "")
                        raw_result = raw
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                if "error" in parsed:
                                    result_content = f"ERROR: {parsed['error']}"
                                elif "truncated" in parsed:
                                    result_content = f"(truncated, {parsed.get('original_length', '?')} chars)"
                                else:
                                    result_content = str(list(parsed.keys()))[:result_preview]
                            else:
                                result_content = str(parsed)[:result_preview]
                        except Exception:
                            result_content = raw[:result_preview]
                        break

                exec_parts.append(_tool_summary_line(tc_name, t_arg_str, result_content))

                if side_log is not None:
                    try:
                        # run_turn already logged this call at execution time
                        # (with ok/duration_ms). Collapsing the same round again
                        # — which happens on every compaction pass — must reuse
                        # that row, not append a poorer duplicate: the session's
                        # tool_calls.jsonl was otherwise ~50% duplicates.
                        seq = None
                        getter = getattr(side_log, "seq_for_call_id", None)
                        if getter is not None:
                            seq = getter("tool_calls.jsonl", tc.get("id"))
                        if seq is None:
                            seq = side_log.append("tool_calls.jsonl", {
                                "turn": turn_id,
                                "tool_call_id": tc.get("id"),
                                "tool": tc_name,
                                "arguments": t_args,
                                "result": raw_result,
                            })
                        refs.append(seq)
                    except Exception as e:
                        logger.warning("side_log append failed: %s", e)

            summary = "\n".join(exec_parts)

            if m.get("content") and str(m["content"]).strip():
                # Combine text content and tool summary into ONE assistant message
                # (avoids consecutive assistant messages that strict APIs reject)
                combined_content = m["content"].rstrip() + "\n\n" + summary
                combined: dict = {"role": "assistant", "content": combined_content}
                if rc := m.get("_reasoning_content"):
                    combined["_reasoning_content"] = rc
                if refs:
                    combined["_tool_refs"] = refs
                out.append(combined)
            else:
                summary_msg: dict = {"role": "assistant", "content": summary}
                if refs:
                    summary_msg["_tool_refs"] = refs
                out.append(summary_msg)
            i = j
        else:
            out.append(m)
            i += 1
    return out


def _truncate_large_messages(messages: list[dict], token_budget: int) -> list[dict]:
    from agent._tokens import count_tokens_approx

    result = [m.copy() for m in messages]
    for _ in range(10):
        total = sum(count_tokens_approx(m.get("content") or "") for m in result)
        if total <= token_budget:
            break
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(result):
            if m.get("role") in ("system",):
                continue
            content = m.get("content") or ""
            toks = count_tokens_approx(content)
            if toks > longest_len:
                longest_len = toks
                longest_idx = i
        if longest_idx < 0 or longest_len < 100:
            break
        content = result[longest_idx].get("content") or ""
        keep_chars = max(200, len(content) // 4)
        result[longest_idx] = {
            **result[longest_idx],
            "content": content[:keep_chars] + "\n\n[... truncated to fit context window ...]",
        }
    return result


def _build_extracted_summary(filename: str, code: str, outcome: str, err: str | None, existing_len: int, side_log, turn_id: int | None) -> dict:
    if outcome == "ok":
        arrow = "ok"
    elif outcome == "refused_shrink":
        arrow = f"refused (would shrink {existing_len}→{len(code)} chars)"
    elif outcome == "refused_policy":
        arrow = f"refused by policy: {err}"
    else:
        arrow = f"ERROR: {err}"

    safe_path = filename.replace("\\", "\\\\").replace("'", "\\'")
    # Tool name stays a plain \w+ identifier so the line matches
    # _TOOL_SUMMARY_RE like any other collapsed round (http replay unfolds it,
    # and an imitation of it is detected rather than silently believed).
    summary_text = _tool_summary_line("write_file", f"path='{safe_path}'", f"{arrow} (extracted from narration)")
    summary_msg: dict = {"role": "assistant", "content": summary_text}

    if side_log is not None:
        try:
            seq = side_log.append("tool_calls.jsonl", {
                "turn": turn_id,
                "tool_call_id": None,
                "tool": "write_file (extracted)",
                "arguments": {"path": filename, "content": code},
                "result": {"outcome": outcome, "existing_len": existing_len, "error": err},
                "source": "narration_fallback",
            })
            summary_msg["_tool_refs"] = [seq]
        except Exception as e:
            logger.warning("side_log append failed (extracted fallback): %s", e)

    return summary_msg


def _is_prose_not_code(filename: str, code: str) -> bool:
    """Return True if `code` looks like planning prose rather than source code."""
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return False
    # High ratio of markdown bullet lines → prose
    bullet_lines = sum(1 for l in lines if l.lstrip().startswith(("* ", "- ", "• ")))
    if bullet_lines / len(lines) >= _PROSE_BULLET_THRESHOLD:
        return True
    # Python-specific: attempt to parse; refuse if syntax error AND no valid
    # Python statement could be constructed (i.e. it's not real code at all).
    ext = Path(filename).suffix.lower()
    if ext == ".py":
        try:
            import ast as _ast
            _ast.parse(code)
        except SyntaxError:
            # Still might be partial code — only refuse if bullets are dominant.
            # A SyntaxError alone is insufficient (e.g. f-strings with complex exprs).
            if bullet_lines > 0:
                return True
    return False


def _apply_code_from_history(
    messages: list[dict],
    on_tool_call,
    side_log=None,
    turn_id: int | None = None,
) -> tuple[str, dict | None] | None:
    """UNGATED narration fallback (no permissions/hooks/classifier) — kept for
    tests of the extraction rules. The turn loop uses
    ``apply_code_from_history_gated``."""
    prep = _prepare_extracted_write(messages, side_log=side_log, turn_id=turn_id)
    if prep is None or prep[0] == "done":
        return None if prep is None else prep[1]
    _, filename, code, existing_len = prep
    return _commit_extracted_write(filename, code, existing_len, on_tool_call,
                                   side_log=side_log, turn_id=turn_id)


async def apply_code_from_history_gated(
    messages: list[dict],
    on_tool_call,
    config,
    side_log=None,
    turn_id: int | None = None,
) -> tuple[str, dict | None] | None:
    """Narration fallback through the same gates as a real write_file call.

    The extraction rules (prose, XML tags, missing parent, shrink) run first,
    so the user is never asked to approve a write that would be refused anyway;
    then permissions → pre-tool hooks → action classifier (pre_tool_gates);
    only then the file is written. A blocked write is reported to the model
    like any refused call, not silently dropped.
    """
    prep = _prepare_extracted_write(messages, side_log=side_log, turn_id=turn_id)
    if prep is None or prep[0] == "done":
        return None if prep is None else prep[1]
    _, filename, code, existing_len = prep
    from agent.core.tool_calls import pre_tool_gates
    blocked, _note = await pre_tool_gates("write_file", {"path": filename, "content": code}, config)
    if blocked is not None:
        reason = str(blocked.get("error") or "blocked by policy")
        logger.warning("[extract] write to %s blocked by gate: %s", filename, reason)
        human = (f"Refused to write `{filename}` from narration: {reason}. "
                 f"Nothing was written.")
        summary = _build_extracted_summary(filename, code, "refused_policy", err=reason,
                                           existing_len=existing_len, side_log=side_log,
                                           turn_id=turn_id)
        return human, summary
    return _commit_extracted_write(filename, code, existing_len, on_tool_call,
                                   side_log=side_log, turn_id=turn_id)


def _prepare_extracted_write(messages: list[dict], side_log=None, turn_id: int | None = None):
    """Extraction + refusal rules, no disk write.

    → None (nothing to apply) | ("done", (human, summary)) for a refusal already
    decided here | ("write", filename, code, existing_len).
    """
    # A message that wrote tool calls as XML tags (<write_file path=... content=...>)
    # cannot be mined safely: the "code" lives inside an attribute value, so any
    # block we pull out is a truncated, backslash-escaped fragment. Let the turn
    # loop nudge for a real tool call instead of writing that fragment to disk.
    last_assistant = next(
        (m.get("content") or "" for m in reversed(messages)
         if m.get("role") == "assistant" and (m.get("content") or "").strip()),
        "",
    )
    if _has_pseudo_tool_tag(last_assistant):
        logger.warning("[extract] refused: response wrote tool calls as XML tags, not real calls")
        return None

    result = extract_last_code_block(messages)
    if not result:
        return None
    filename, code = result

    outcome: str
    human: str

    # Reject prose masquerading as code before touching disk.
    if _is_prose_not_code(filename, code):
        logger.warning(
            "[extract] refused write to %s: content looks like prose/planning text, not code",
            filename,
        )
        return None

    p = Path(filename)
    if not p.is_absolute():
        try:
            from agent.tools.files.paths import _working_dir
            p = _working_dir() / p
        except Exception:
            p = Path.cwd() / p
    # Narration fallback exists to recover an edit the model described instead of
    # calling; it must never mint a new directory tree. A path whose parent does
    # not exist is almost always a misparse (a URL, a made-up path) rather than a
    # file the model meant to write.
    if not p.exists() and not p.parent.is_dir():
        logger.warning(
            "[extract] refused write to %s: parent directory does not exist "
            "(narration fallback does not create directory trees)", filename,
        )
        return None
    existing_len = 0
    if p.exists() and p.is_file():
        try:
            existing = p.read_text(encoding="utf-8")
            existing_len = len(existing)
            if len(code) < existing_len * _EXTRACT_SHRINK_RATIO:
                logger.warning(
                    "[extract] refused overwrite of %s: extracted %d chars would shrink existing %d chars (<%.0f%%)",
                    filename, len(code), existing_len, _EXTRACT_SHRINK_RATIO * 100,
                )
                outcome = "refused_shrink"
                human = (
                    f"Refused to overwrite `{filename}` from an extracted snippet "
                    f"({len(code)} chars) — file has {existing_len} chars. "
                    f"Call edit_file or write_file explicitly if this is intended."
                )
                summary = _build_extracted_summary(filename, code, outcome, err=None, existing_len=existing_len, side_log=side_log, turn_id=turn_id)
                return "done", (human, summary)
        except Exception:
            pass
    return "write", filename, code, existing_len


def _commit_extracted_write(filename: str, code: str, existing_len: int, on_tool_call,
                            side_log=None, turn_id: int | None = None) -> tuple[str, dict]:
    err: str | None = None
    from agent.tools.files import write_file
    if on_tool_call:
        on_tool_call("write_file (extracted)", filename)
    r = write_file(filename, code)
    if "error" in r:
        outcome = "error"
        err = str(r["error"])
        human = f"Failed to apply: {r['error']}"
    else:
        outcome = "ok"
        human = f"Applied changes to `{filename}`."

    summary = _build_extracted_summary(filename, code, outcome, err=err, existing_len=existing_len, side_log=side_log, turn_id=turn_id)
    return human, summary


def extract_last_code_block(messages: list[dict]) -> tuple[str, str] | None:
    content = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            content = m["content"]
            break
    if not content:
        return None

    fenced_matches = list(re.finditer(r"```(?:\w*)\n(.*?)```", content, re.DOTALL))
    for m in fenced_matches:
        code = m.group(1).strip()
        pre = content[max(0, m.start() - 200):m.start()]
        post = content[m.end():m.end() + 80]
        candidates = _find_filenames(pre) + _find_filenames(post)
        if candidates:
            pre_hits = _find_filenames(pre)
            filename = pre_hits[-1] if pre_hits else candidates[0]
            if code:
                return filename, code

    indented: list[str] = []
    block_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("    ") or line.startswith("\t"):
            block_lines.append(line.lstrip())
        else:
            if len(block_lines) >= 2:
                indented.append("\n".join(block_lines))
            block_lines = []
    if len(block_lines) >= 2:
        indented.append("\n".join(block_lines))

    if not indented:
        logger.debug(f"[extract] no code blocks with a nearby filename in: {content[:120]!r}")
        return None

    code = max(indented, key=len).strip()
    # Prefer the filename mentioned closest *before* the block, matching the
    # fenced branch — the first name in the message is often unrelated context.
    block_start = content.find(code.splitlines()[0].strip()) if code.splitlines() else -1
    pre_hits = _find_filenames(content[:block_start]) if block_start > 0 else []
    same_msg_hits = pre_hits or _find_filenames(content)
    if not same_msg_hits:
        logger.debug(f"[extract] indented code found ({len(code)} chars) but no filename in same message")
        return None
    return (pre_hits[-1] if pre_hits else same_msg_hits[0]), code
