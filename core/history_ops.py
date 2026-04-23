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


def _merge_trailing_assistants(api_messages: list[dict]) -> list[dict]:
    if len(api_messages) < 2:
        return api_messages
    out = list(api_messages)
    while len(out) >= 2:
        a, b = out[-2], out[-1]
        if (
            a.get("role") == "assistant"
            and b.get("role") == "assistant"
            and not a.get("tool_calls")
            and not b.get("tool_calls")
        ):
            merged = {"role": "assistant", "content": (a.get("content") or "") + (b.get("content") or "")}
            out = out[:-2] + [merged]
        else:
            break
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

            if m.get("content") and str(m["content"]).strip():
                out.append({"role": "assistant", "content": m["content"]})

            parts: list[str] = []
            refs: list[int] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("function", {}).get("name", "?")
                args_raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    arg_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:2])
                except Exception:
                    args = args_raw
                    arg_str = str(args_raw)[:60]

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

                parts.append(f"{name}({arg_str}) → {result_content}")

                if side_log is not None:
                    try:
                        seq = side_log.append("tool_calls.jsonl", {
                            "turn": turn_id,
                            "tool_call_id": tc.get("id"),
                            "tool": name,
                            "arguments": args,
                            "result": raw_result,
                        })
                        refs.append(seq)
                    except Exception as e:
                        logger.warning("side_log append failed: %s", e)

            summary = "[tools: " + " | ".join(parts) + "]"
            summary_msg: dict = {"role": "system", "content": summary}
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

    summary_text = f"[tools: write_file (extracted)(path={filename!r}) → {arrow}]"
    summary_msg: dict = {"role": "system", "content": summary_text}

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
