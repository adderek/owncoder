"""agent.py — backward-compat re-exports for public API.

All implementation lives in agent/core/. This module re-exports
everything that external callers and tests previously imported from here.
"""
from __future__ import annotations

# Re-exports from agent/core/ — all public API stays at agent.agent.*
from agent.core.agent import Agent
from agent.core.loop_detector import LoopDetector
from agent.core.turn import run_turn, _post_turn_capture_and_summarize
from agent.core.tool_calls import (
    execute_tool,
    _FakeToolCall,
    _parse_raw_tool_calls,
    _tool_result_message,
    _tool_result_char_limit,
    _extract_json_objects,
    _TOOL_WRAP_TAGS,
    _TAG_RE,
    _DECODER,
)
from agent.core.streaming import (
    _stream_response,
    _strip_tool_blocks,
    _is_narrating_tool_use,
    _NARRATION_PHRASES,
)
from agent.core.history_ops import (
    _merge_trailing_assistants,
    _collapse_tool_rounds,
    _truncate_large_messages,
    _apply_code_from_history,
    _build_extracted_summary,
    extract_last_code_block,
    _FILE_RE,
    _EXTRACT_SHRINK_RATIO,
)
from agent.core.prompts import (
    _build_system_prompt,
    _inject_think_hint,
    _build_call_kwargs,
    _log_llm_request,
    THINK_LEVELS,
    _THINK_HINTS,
    _REASONING_EFFORT,
    SYSTEM_PROMPT_PATH,
    GUIDELINES_DIR,
    _PREAMBLE_CACHE,
)
