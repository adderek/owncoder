"""Per-turn setup: which tool schemas the model is offered, and the final
message-shape fixups applied before every API call.

Split out of core/turn.py — both are pure(ish) transforms with no dependency on
the turn's mutable state, and both are where model-specific quirks accumulate.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, TYPE_CHECKING

from agent.memory.compactor import _count_tokens_approx

from .history_ops import _merge_consecutive_assistants

if TYPE_CHECKING:
    from agent.config import Config

logger = logging.getLogger(__name__)


def select_tools(all_schemas: list[dict], config: "Config",
                 excluded_tools: set[str] | None = None,
                 ) -> tuple[list[dict], Callable[[], list[dict]] | None, bool]:
    """Narrow the full tool catalog to what this turn should offer.

    Returns ``(tools, refresh, compaction_on)``. *refresh* is non-None only under
    progressive tool disclosure: the turn loop calls it at the top of each
    iteration to re-expose whatever the model activated via find_tools. The
    active set is reset here, so it starts empty every turn and the model
    re-discovers what THIS turn needs instead of inheriting the last one's.
    """
    tools = all_schemas
    if excluded_tools:
        tools = [t for t in tools if t.get("function", {}).get("name") not in excluded_tools]
    # Typed turn-signal tools (ask_user/mark_done/blocked/…) are only offered
    # when turn signals are enabled; otherwise drop them from the schema.
    _ts_cfg = getattr(config, "turn_signals", None)
    if _ts_cfg is not None and not getattr(_ts_cfg, "enabled", True):
        from agent.tools.turn_signals import SIGNAL_TOOL_NAMES
        tools = [t for t in tools if t.get("function", {}).get("name") not in SIGNAL_TOOL_NAMES]
    compaction_on = config.tool_compaction.enabled
    if compaction_on:
        from agent.tool_compactor import inject_purpose_into_schemas
        tools = inject_purpose_into_schemas(tools)

    _discovery_on = bool(getattr(getattr(config, "tool_discovery", None), "enabled", False))
    if not _discovery_on:
        # find_tools is meaningless without the catalog → never offer it.
        tools = [t for t in tools if t.get("function", {}).get("name") != "find_tools"]
        return tools, None, compaction_on

    from agent.core import tool_discovery as _td
    _td.reset_active()
    _all_tools = tools

    def _refresh() -> list[dict]:
        return _td.select_schemas(_all_tools, _td.active_names(), config)

    tools = _refresh()
    _full_tok = _count_tokens_approx([{"content": json.dumps(_all_tools)}])
    _core_tok = _count_tokens_approx([{"content": json.dumps(tools)}])
    logger.info(
        "tool_discovery: %d/%d tool schemas exposed (~%d of ~%d tokens, saving ~%d)",
        len(tools), len(_all_tools), _core_tok, _full_tok, _full_tok - _core_tok,
    )
    return tools, _refresh, compaction_on


def normalize_api_messages(messages: list[dict]) -> list[dict]:
    """Strip internal keys and apply model-quirk fixups to produce API-ready messages.

    Pure transform (no side effects): drops _-prefixed keys, surfaces stored
    reasoning, merges consecutive assistants, merges leading system messages,
    strips a trailing prefill assistant, and fills reasoning_content for
    thinking-mode sessions.
    """
    def _to_api_msg(m: dict) -> dict:
        result = {k: v for k, v in m.items() if not k.startswith("_")}
        if rc := m.get("_reasoning_content"):
            result["reasoning_content"] = rc
        return result

    api_messages = [_to_api_msg(m) for m in messages]
    api_messages = _merge_consecutive_assistants(api_messages)
    # Merge ALL system messages into a single leading one — some models (e.g.
    # Qwen3.6 with --jinja) raise a Jinja exception if any system message has
    # loop.first=False, meaning only the very first message may be a system msg.
    # System msgs can end up mid-conversation (e.g. scheduled/bg-job status
    # notes appended by agent.py between turns), not just as a leading run, so
    # this must scan the whole list rather than only the leading prefix.
    sys_msgs = [m for m in api_messages if m.get("role") == "system"]
    rest = [m for m in api_messages if m.get("role") != "system"]
    if sys_msgs:
        merged_content = "\n\n".join(m["content"] for m in sys_msgs if m.get("content"))
        api_messages = [{**sys_msgs[0], "content": merged_content}] + rest
    # Trailing assistant without tool_calls = unintentional prefill; reject by
    # most APIs (and always incompatible with enable_thinking). Strip it.
    if api_messages and api_messages[-1].get("role") == "assistant" and not api_messages[-1].get("tool_calls"):
        logger.warning("run_turn: stripping trailing assistant message (prefill) before API call")
        api_messages = api_messages[:-1]
    # DeepSeek / reasoning models require reasoning_content on ALL assistant
    # messages in a thinking-mode session. Fill absent ones with "".
    if any(m.get("role") == "assistant" and m.get("reasoning_content") for m in api_messages):
        api_messages = [
            {**m, "reasoning_content": m.get("reasoning_content", "")}
            if m.get("role") == "assistant" and "reasoning_content" not in m
            else m
            for m in api_messages
        ]
    return api_messages
