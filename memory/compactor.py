"""Two-stage, incremental session compaction.

Stage 1 — *Deep Analysis* — expands the transcript into a detailed
"knowledge draft" (Tier-2 source facts). High token budget, no brevity
pressure. Accepts the previous round's knowledge draft so facts accumulate
across rounds instead of being re-derived (and gradually lost) each time.

Stage 2 — *Refined Synthesis* — compresses the draft into the artefacts
that actually land in the live context: a short `<summary>` for the A view,
a `<q>` restatement of the user's outstanding intent, and a structured
`<facts>` JSON. Low token budget.

Both stages are model calls, but the Stage-1 draft is persisted to disk via
`FactsStore` so the agent can recall elided specifics through the
`recall_facts` tool without having to reconstruct them from nothing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from agent.config import Config
    from agent.memory.facts_store import FactsStore, FactsRound


class CompactionError(Exception):
    """Raised when stage-2 synthesis cannot produce usable output."""


_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# Back-compat: existing tests import COMPACTION_PROMPT. Keep the name but
# point it at the new Stage-2 prompt, since that's the one whose output
# format (<facts>/<summary>) still matches what _parse_compaction_output
# parses.
def _read_prompt(name: str) -> str:
    p = _PROMPTS_DIR / name
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


ANALYZE_PROMPT = _read_prompt("analyze.txt") or (
    "Expand the transcript into a detailed knowledge draft. Preserve every "
    "decision, file change, error, and outstanding item. Prefer completeness "
    "over brevity."
)

SYNTHESIZE_PROMPT = _read_prompt("synthesize.txt") or (
    "Compress the knowledge draft. Output <facts>{...}</facts><summary>...</summary><q>...</q>."
)

# Retained for backward compatibility with tests importing COMPACTION_PROMPT.
COMPACTION_PROMPT = SYNTHESIZE_PROMPT


# ── token accounting ────────────────────────────────────────────────────────


def _count_tokens_approx(messages: list[dict]) -> int:
    from agent._tokens import count_tokens_approx

    total = 0
    for m in messages:
        content = m.get("content") or ""
        if m.get("tool_calls"):
            total += count_tokens_approx(json.dumps(m["tool_calls"]))
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += count_tokens_approx(str(part.get("text", "")))
        else:
            total += count_tokens_approx(str(content))
    return total


# ── Stage 2 output parsing ──────────────────────────────────────────────────

_FACTS_RE = re.compile(r"<facts>(.*?)</facts>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_Q_RE = re.compile(r"<q>(.*?)</q>", re.DOTALL)


def _parse_compaction_output(text: str) -> tuple[dict, str]:
    """Legacy 2-tuple parser (facts, summary) — kept for back-compat."""
    facts_m = _FACTS_RE.search(text)
    summary_m = _SUMMARY_RE.search(text)
    facts: dict = {}
    if facts_m:
        try:
            facts = json.loads(facts_m.group(1).strip())
        except json.JSONDecodeError:
            pass
    summary = summary_m.group(1).strip() if summary_m else ""
    return facts, summary


def _parse_synthesis_output(text: str) -> tuple[dict, str, str]:
    """Full 3-tuple parser (facts, summary, q_view) for Stage 2."""
    facts, summary = _parse_compaction_output(text)
    q_m = _Q_RE.search(text)
    q_view = q_m.group(1).strip() if q_m else ""
    return facts, summary, q_view


# ── helpers ─────────────────────────────────────────────────────────────────


def _truncate_tool_results_in(
    messages: list[dict], max_chars: int = 2000
) -> list[dict]:
    """Truncate oversized tool-result messages within a message list."""
    result = []
    for m in messages:
        if m.get("role") == "tool" and len(m.get("content", "")) > max_chars:
            result.append(
                {
                    **m,
                    "content": m["content"][:max_chars] + "\n[... truncated ...]",
                }
            )
        else:
            result.append(m)
    return result


def _msg_text(m: dict) -> str:
    role = m.get("role", "?")
    content = m.get("content")
    parts = []
    if m.get("tool_calls"):
        calls = [
            f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
            for tc in m["tool_calls"]
            if isinstance(tc, dict)
        ]
        parts.append("[tool_calls: " + ", ".join(calls) + "]")
    if isinstance(content, str) and content:
        parts.append(content[:2000] if len(content) > 2000 else content)
    elif isinstance(content, list):
        parts.append(json.dumps(content)[:2000])
    body = " | ".join(parts) if parts else ""
    return f"[{role}]: {body}"


def _render_transcript(messages: list[dict]) -> str:
    return "\n".join(_msg_text(m) for m in messages)


def _fit_to_budget(text: str, budget_tokens: int) -> str:
    from agent._tokens import count_tokens_approx

    tokens = count_tokens_approx(text)
    if tokens <= budget_tokens:
        return text
    ratio = budget_tokens / max(tokens, 1)
    return text[: int(len(text) * ratio)]


# ── Stage 1: Deep Analysis ──────────────────────────────────────────────────


async def _analyze_transcript(
    transcript_text: str,
    prev_round: "FactsRound | None",
    config: "Config",
    client: "AsyncOpenAI",
) -> str:
    """Produce a detailed knowledge draft.

    Incremental: if `prev_round` is given, feed its knowledge_draft and
    summary into the prompt so the model extends rather than replaces.
    Uses a generous max_tokens so nuance is not clipped.
    """
    # Shortcut: when the transcript is small enough to serve as its own
    # knowledge draft, skip the Stage-1 LLM call entirely.
    # Saves 200-800 ms per short compaction (often 30-50% of compaction
    # rounds when the new segment since last compact is only 1-3 turns).
    from agent._tokens import count_tokens_approx
    _SHORT_TRANSCRIPT_THRESHOLD = 1500
    if count_tokens_approx(transcript_text) < _SHORT_TRANSCRIPT_THRESHOLD:
        if prev_round is None:
            return transcript_text
        # Extend previous knowledge draft with the small new transcript
        # rather than making a full LLM call to re-analyze from scratch.
        return prev_round.knowledge_draft.rstrip() + "\n\n" + transcript_text

    user_parts: list[str] = []
    if prev_round is not None:
        user_parts.append(
            f"## Previous knowledge draft (round {prev_round.round_id}, "
            f"turns {prev_round.from_turn}..{prev_round.to_turn})\n"
            f"{prev_round.knowledge_draft}"
        )
        if prev_round.summary:
            user_parts.append(f"## Previous compressed summary\n{prev_round.summary}")
        if prev_round.q_view:
            user_parts.append(f"## [PREVIOUS Q_VIEW] (carry verbatim unless new instruction supersedes)\n{prev_round.q_view}")
        user_parts.append("## New transcript segment to fold in\n" + transcript_text)
    else:
        user_parts.append("## Transcript to analyze\n" + transcript_text)

    # Reserve: prompt overhead (~500) + stage-1 output (use most of
    # ctx_window minus those) for the deep draft.
    output_budget = max(
        config.token_limits.compactor_analyze_min,
        int(config.llm.ctx_window * 0.35),
    )
    input_budget = max(
        1024,
        config.llm.ctx_window - output_budget - 500,
    )
    user_text = _fit_to_budget("\n\n".join(user_parts), input_budget)

    from agent import prompt_compiler
    messages = [
        {"role": "system", "content": prompt_compiler.load("analyze.txt", ANALYZE_PROMPT, config)},
        {"role": "user", "content": user_text},
    ]
    try:
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=messages,
            max_tokens=output_budget,
            # Summarization-style tasks don't benefit from hidden reasoning,
            # and reasoning can starve the visible output budget on reasoning
            # models. Disable chain-of-thought for both compaction stages.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("analyze_transcript: stage 1 failed: %s", e)
        # Fallback: return a degraded draft made of the transcript itself
        # so Stage 2 still has something to work with.
        prefix = ""
        if prev_round is not None:
            prefix = prev_round.knowledge_draft.rstrip() + "\n\n"
        return prefix + transcript_text


def _looks_complete(text: str) -> bool:
    """Checks if all required tags are present and non-empty."""
    return all(reg.search(text) is not None for reg in [_FACTS_RE, _SUMMARY_RE, _Q_RE])


async def _synthesize_summary(
    knowledge_draft: str,
    config: "Config",
    client: "AsyncOpenAI",
) -> tuple[dict, str, str]:
    """Compress the knowledge draft into (facts, summary, q_view)."""
    # Strict budget — this is what lands in context.
    from agent import prompt_compiler
    messages = [
        {"role": "system", "content": prompt_compiler.load("synthesize.txt", SYNTHESIZE_PROMPT, config)},
        {
            "role": "user",
            "content": "Knowledge draft to compress:\n\n"
            + _fit_to_budget(knowledge_draft, int(config.llm.ctx_window * 0.6)),
        },
    ]

    async def _call(max_tokens: int) -> tuple[str, str | None]:
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)
        return raw, finish

    try:
        raw, finish = await _call(config.token_limits.compactor_synthesize_initial)
    except Exception as e:
        logger.warning("synthesize_summary: stage 2 first call failed: %s", e)
        raise CompactionError(f"stage 2 call failed: {e}") from e

    if finish == "length" or not _looks_complete(raw):
        logger.info(
            "synthesize_summary: first attempt truncated/incomplete (finish=%s); retrying",
            finish,
        )
        try:
            raw, finish = await _call(config.token_limits.compactor_synthesize_retry)
        except Exception as e:
            logger.warning("synthesize_summary: stage 2 retry failed: %s", e)
            raise CompactionError(f"stage 2 retry failed: {e}") from e

    if not _looks_complete(raw):
        raise CompactionError("stage 2 output incomplete after retry")

    return _parse_synthesis_output(raw)


# ── Goal drift detection ────────────────────────────────────────────────────

_DRIFT_SYSTEM = (
    "You are checking whether an agent's working summary has drifted from the "
    "user's original request. Compare the two and decide if the goal has "
    "significantly changed. If drifted, output a corrected one-paragraph q_view "
    "that stays true to the original request while reflecting any legitimate "
    "new instructions.\n\n"
    "Output JSON only: {\"drifted\": true/false, \"corrected_q\": \"...\"}\n"
    "If not drifted, set corrected_q to empty string."
)

_DRIFT_JSON_RE = re.compile(r'\{[^{}]*"drifted"[^{}]*\}', re.DOTALL)


async def _check_goal_drift(
    original_request: str,
    q_view: str,
    config: "Config",
    client: "AsyncOpenAI",
) -> str | None:
    """Compare original_request vs current q_view. Return corrected q_view if drifted, else None.

    Uses the summarizer model (cheap/fast). Timeout-guarded — returns None on any failure.
    """
    if not original_request or not q_view:
        return None
    prompt = (
        f"original_request: {original_request}\n\n"
        f"current q_view: {q_view}"
    )
    try:
        from agent.config import make_registry
        entry = make_registry(config).summarizer
        from openai import AsyncOpenAI as _OAI
        sum_client = _OAI(base_url=entry.base_url, api_key=entry.api_key)
        try:
            response = await sum_client.chat.completions.create(
                model=entry.model,
                messages=[
                    {"role": "system", "content": _DRIFT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        finally:
            await sum_client.close()
        raw = (response.choices[0].message.content or "").strip()
        m = _DRIFT_JSON_RE.search(raw)
        if not m:
            return None
        data = json.loads(m.group(0))
        if data.get("drifted") and data.get("corrected_q"):
            logger.info("goal_drift: drift detected; correcting q_view")
            return str(data["corrected_q"]).strip()
    except Exception as e:
        logger.debug("_check_goal_drift: skipped (%s)", e)
    return None


# ── Project-level session indexing ─────────────────────────────────────────


def _index_round_to_project(
    project_memory_store,
    round_obj,
    *,
    session_id: str | None = None,
    embedder=None,
) -> None:
    """Index a compaction round's summary into the project MemoryStore.

    scope='session_summary'. Enables cross-session recall_sessions search.
    """
    try:
        body = "\n\n".join(filter(None, [round_obj.summary, round_obj.q_view]))
        if not body.strip():
            return
        embedding = None
        if embedder is not None:
            try:
                embedding = embedder.embed_one(body[:4000])
            except Exception:
                pass
        eid = f"session:{session_id or 'unknown'}:round:{round_obj.round_id}"
        project_memory_store.add(
            scope="session_summary",
            body=body,
            source=session_id or "",
            title=f"Session {(session_id or '')[:16]}… round {round_obj.round_id} turns {round_obj.from_turn}–{round_obj.to_turn}",
            embedding=embedding,
            entry_id=eid,
        )
    except Exception:
        pass


# ── Public entry point ──────────────────────────────────────────────────────


async def compact(
    messages: list[dict],
    config: "Config",
    client: "AsyncOpenAI",
    keep_last: int = 4,
    facts_store: "FactsStore | None" = None,
    turn_index: int | None = None,
    project_memory_store=None,
    session_id: str | None = None,
) -> list[dict]:
    """Compact `messages`, returning a shorter message list.

    Two-stage flow:
      1. Separate system + recent tail (`keep_last` turns) from older content.
      2. Run Stage 1 (deep analysis) on the older content, carrying forward
         the previous round's draft if `facts_store` has one.
      3. Run Stage 2 (synthesis) to get the compact summary/q/facts.
      4. Persist the round in `facts_store` (Tier 2).
      5. Return [system, compacted_summary_message, *recent tail].

    `turn_index` — current turn counter; used for round bookkeeping.
    """
    if len(messages) <= keep_last * 2:
        token_est = _count_tokens_approx(messages)
        budget = int(config.llm.ctx_window * config.llm.compaction_threshold)
        if token_est > budget:
            return _truncate_tool_results_in(messages, max_chars=budget * 2)
        return messages

    hard_rules_msgs: list[dict] = []
    system_msg = None
    conversation = []
    for m in messages:
        if m.get("role") == "system":
            if m.get("_hard_rules_marker"):
                hard_rules_msgs.append(m)
            else:
                system_msg = m
        else:
            conversation.append(m)

    verbatim_start = max(len(conversation) - keep_last * 2, 0)
    # Always preserve the most recent user message verbatim, even if a long
    # tool-call burst has scrolled it out of the keep_last window. Otherwise the
    # user's question survives only via q_view and is lost if Stage 2 fails.
    last_user_idx = next(
        (
            i
            for i in range(len(conversation) - 1, -1, -1)
            if conversation[i].get("role") == "user"
        ),
        None,
    )
    if last_user_idx is not None and last_user_idx < verbatim_start:
        verbatim_start = last_user_idx
    to_compact = conversation[:verbatim_start]
    verbatim = conversation[verbatim_start:]

    if not to_compact:
        return _truncate_tool_results_in(messages, max_chars=2000)

    transcript_text = _render_transcript(to_compact)

    prev_round = facts_store.latest_round() if facts_store is not None else None
    round_id = prev_round.round_id if prev_round is not None else None
    from_turn = (prev_round.to_turn + 1) if prev_round else 0
    to_turn = turn_index if turn_index is not None else (from_turn + len(to_compact))

    # Stage 1
    knowledge_draft = await _analyze_transcript(
        transcript_text, prev_round, config, client
    )

    # Stage 2
    try:
        facts, summary, q_view = await _synthesize_summary(
            knowledge_draft, config, client
        )
    except CompactionError as e:
        logger.warning("compact: stage 2 failed, falling back to error summary: %s", e)
        error_msg = {"role": "assistant", "content": f"[SESSION SUMMARY ERROR: {e}]"}
        result = list(hard_rules_msgs)
        if system_msg:
            result.append(system_msg)
        result.append(error_msg)
        result.extend(_truncate_tool_results_in(verbatim, max_chars=2000))
        return result

    # Propagate original_request from previous round if synthesizer dropped it.
    prev_original = (prev_round.facts or {}).get("original_request", "") if prev_round else ""
    if prev_original and not facts.get("original_request"):
        facts["original_request"] = prev_original

    # Drift check: if original_request exists, verify q_view hasn't drifted.
    original_request = facts.get("original_request", "")
    if original_request and q_view:
        corrected = await _check_goal_drift(original_request, q_view, config, client)
        if corrected:
            q_view = corrected
            facts["_drift_corrected"] = True

    saved_round = None
    if facts_store is not None:
        saved_round = facts_store.new_round(
            from_turn=from_turn,
            to_turn=to_turn,
            knowledge_draft=knowledge_draft,
            summary=summary,
            q_view=q_view,
            facts=facts,
            prev=prev_round,
        )

    if project_memory_store is not None and saved_round is not None:
        _index_round_to_project(
            project_memory_store,
            saved_round,
            session_id=session_id,
            embedder=getattr(facts_store, "_embedder", None),
        )

    # Build the compacted system-summary message
    header = "[SESSION SUMMARY"
    if round_id is not None:
        header += f" · round {round_id}"
    header += "]"
    hint = (
        " (Earlier detail is stored as Tier-2 facts. Call `recall_facts(query=..., "
        "round_id=...)` to retrieve specifics not present here.)"
    )
    compacted_content = (
        f"{header}{hint}\n{json.dumps(facts, separators=(',', ':'))}\n\n{summary}"
    )
    if q_view:
        compacted_content += f"\n\n[OUTSTANDING USER INTENT]\n{q_view}"

    compacted_msg = {"role": "assistant", "content": compacted_content}

    verbatim = _truncate_tool_results_in(verbatim, max_chars=2000)

    result: list[dict] = list(hard_rules_msgs)
    if system_msg:
        result.append(system_msg)
    result.append(compacted_msg)
    result.extend(verbatim)
    return result
