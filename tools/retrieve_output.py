"""`retrieve_output` tool — fetch full or partial output from truncated tool calls.

When execute_tool truncates a large result, the full output is stored
in the OutputStore. This tool lets the agent retrieve any part of it
without re-running the original tool.
"""
from __future__ import annotations

from agent.tools import register

_config = None


def setup(config) -> None:
    global _config
    _config = config


def _get_store():
    from agent.core.output_store import get_store
    return get_store()


def _looks_like_placeholder(call_id: str) -> bool:
    """Detect hallucinated/planning-text call IDs."""
    if len(call_id) > 80:
        return True
    low = call_id.lower()
    return any(marker in low for marker in ("not provided", "...", "use read_file", "(", "will use", "placeholder"))


@register(
    "retrieve_output",
    {
        "description": (
            "Retrieve stored output from a tool call whose result was truncated. "
            "Use when a previous tool result was too large and you need to see "
            "more of its content. You can get the full result or a specific range. "
            "The 'call_id' is shown in the truncated result envelope."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "call_id": {
                    "type": "string",
                    "description": "Call ID from the truncated result envelope.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full", "head", "tail", "range", "lines"],
                    "description": (
                        "'full' = complete stored output. "
                        "'head' = first N chars (default head_chars). "
                        "'tail' = last N chars (default tail_chars). "
                        "'range' = character range (use start/end). "
                        "'lines' = line range (use start_line/end_line, 0-indexed, end exclusive)."
                    ),
                },
                "start": {
                    "type": "integer",
                    "description": "Start character position (for mode='range').",
                },
                "end": {
                    "type": "integer",
                    "description": "End character position, exclusive (for mode='range').",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line, 0-indexed (for mode='lines').",
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line, 0-indexed, exclusive (for mode='lines').",
                },
            },
            "required": ["call_id", "mode"],
        },
    },
)
def retrieve_output(
    call_id: str,
    mode: str = "full",
    start: int | None = None,
    end: int | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    if _looks_like_placeholder(call_id):
        return {
            "error": (
                "call_id looks like a placeholder, not a real output-store ID. "
                "retrieve_output only works with a call_id from a '[output truncated: call_id=...]' "
                "envelope in a previous tool result. Do not invent or guess call_ids."
            )
        }
    try:
        store = _get_store()
    except RuntimeError as e:
        return {"error": str(e)}

    if mode == "full":
        content = store.get(call_id)
        if content is None:
            return {"error": f"call_id '{call_id}' not found in output store (may have expired)."}
        info = store.info(call_id)
        return {"call_id": call_id, "content": content, "chars": info["chars"] if info else len(content)}

    elif mode == "head":
        n = (store.head_chars if end is None else end)
        content = store.get_range(call_id, 0, n)
        if content is None:
            return {"error": f"call_id '{call_id}' not found."}
        return {"call_id": call_id, "content": content}

    elif mode == "tail":
        content = store.get(call_id)
        if content is None:
            return {"error": f"call_id '{call_id}' not found."}
        n = store.tail_chars
        tail = content[-n:]
        return {"call_id": call_id, "content": tail}

    elif mode == "range":
        s = start or 0
        content = store.get_range(call_id, s, end)
        if content is None:
            return {"error": f"call_id '{call_id}' not found."}
        return {"call_id": call_id, "content": content, "start": s, "end": end}

    elif mode == "lines":
        s = start_line or 0
        content = store.get_lines(call_id, s, end_line)
        if content is None:
            return {"error": f"call_id '{call_id}' not found."}
        return {"call_id": call_id, "content": content, "start_line": s, "end_line": end_line}

    else:
        return {"error": f"Unknown mode '{mode}'. Use full, head, tail, range, or lines."}
