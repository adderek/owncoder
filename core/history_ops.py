from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .streaming import _is_narrating_tool_use

logger = logging.getLogger(__name__)

_EXTRACT_SHRINK_RATIO = 0.25

_FILE_RE = re.compile(
    r"\b([a-zA-Z0-9./\-_]+\.(?:sh|bash|py|js|mjs|cjs|ts|jsx|tsx|go|rs|java|kt|c|cpp|h|hpp|rb|toml|yaml|yml|json|md|txt))\b"
)


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

            if a_tc or b_tc:
                merged: dict = {"role": "assistant", "content": merged_content}
                if a_tc and b_tc:
                    merged["tool_calls"] = a_tc + b_tc
                elif a_tc:
                    merged["tool_calls"] = a_tc
                else:
                    merged["tool_calls"] = b_tc
                rc_a = a.get("reasoning_content") or ""
                rc_b = b.get("reasoning_content") or ""
                rc = rc_a if len(rc_a) >= len(rc_b) else rc_b
                if rc:
                    merged["reasoning_content"] = rc
                out[-1] = merged
            else:
                merged: dict = {"role": "assistant", "content": merged_content}
                rc = (a.get("reasoning_content") or "") + (b.get("reasoning_content") or "")
                if rc:
                    merged["reasoning_content"] = rc
                out[-1] = merged
        else:
            out.append(m)
    return out


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
                    t_arg_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(t_args.items())[:2])
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

                safe_args = t_arg_str.replace('"', '&quot;').replace('>', '&gt;').replace('<', '&lt;')
                safe_result = result_content.replace('"', '&quot;').replace('>', '&gt;').replace('<', '&lt;')
                exec_parts.append(f'<agent_exec tool="{tc_name}" args="{safe_args}">{safe_result}</agent_exec>')

                if side_log is not None:
                    try:
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
    else:
        arrow = f"ERROR: {err}"

    safe_path = filename.replace('"', '&quot;').replace('>', '&gt;').replace('<', '&lt;')
    summary_text = f"<agent_exec tool=\"write_file (extracted)\" args=\"path={safe_path}\">{arrow}</agent_exec>"
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


def _apply_code_from_history(
    messages: list[dict],
    on_tool_call,
    side_log=None,
    turn_id: int | None = None,
) -> tuple[str, dict | None] | None:
    result = extract_last_code_block(messages)
    if not result:
        return None
    filename, code = result

    outcome: str
    human: str
    err: str | None = None

    p = Path(filename)
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
                return human, summary
        except Exception:
            pass

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
        candidates = _FILE_RE.findall(pre) + _FILE_RE.findall(post)
        if candidates:
            pre_hits = _FILE_RE.findall(pre)
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
    same_msg_hits = _FILE_RE.findall(content)
    if not same_msg_hits:
        logger.debug(f"[extract] indented code found ({len(code)} chars) but no filename in same message")
        return None
    return same_msg_hits[0], code
