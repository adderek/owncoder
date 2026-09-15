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

from agent.memory.smart_compactor import EntityProtector as _EntityProtector
from agent.core.context_budget import effective_ctx_window as _ctx


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
    "Compress the knowledge draft. Output <facts>{...}</facts><summary>...</summary><q>...</q>.\n"
    "IMPORTANT: If the draft contains error markers like [COMPACTION_ERROR] or looks like "
    "broken JSON/HTTP errors, IGNORE them. Do not extract facts from errors."
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
        int(_ctx(config) * 0.35),
    )
    input_budget = max(
        1024,
        _ctx(config) - output_budget - 500,
    )
    user_text = _fit_to_budget("\n\n".join(user_parts), input_budget)

    from agent import prompt_compiler
    messages = [
        {"role": "system", "content": prompt_compiler.load("analyze.txt", ANALYZE_PROMPT, config)},
        {"role": "user", "content": user_text},
    ]
    try:
        try:
            from agent.metrics import model_calls
            model_calls.record_main(config, role="compaction")
        except Exception:
            pass
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
        # Fallback: return a safe error placeholder to prevent poisoning Stage 2.
        # We avoid returning raw transcript_text because it might contain 
        # connection errors, partial JSONs, or other "dirty" data.
        prefix = ""
        if prev_round is not None:
            prefix = prev_round.knowledge_draft.rstrip() + "\n\n"
        return prefix + "[COMPACTION_ERROR: Stage 1 failed. Raw transcript omitted to prevent context poisoning.]"


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
            + _fit_to_budget(knowledge_draft, int(_ctx(config) * 0.6)),
        },
    ]

    async def _call(max_tokens: int) -> tuple[str, str | None]:
        try:
            from agent.metrics import model_calls
            model_calls.record_main(config, role="compaction")
        except Exception:
            pass
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
        from agent.core.llm_retry import call_role_with_failover
        response, _name, _entry = await call_role_with_failover(
            config, "compaction",
            messages=[
                {"role": "system", "content": _DRIFT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            metrics_role="compaction",
        )
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


# ── File ledger ─────────────────────────────────────────────────────────────
# Compaction summarises file contents away, and the model's usual response is to
# read the same files again — which is what filled the window in the first
# place. Carrying a map of what was read (path, size, and the file's landmarks
# as they are on disk NOW) makes the re-read unnecessary rather than merely
# discouraged: the model can go straight to a range.

_LEDGER_TOOLS = ("read_file", "edit_file", "write_file", "patch_file", "replace_text")
_LEDGER_MAX_FILES = 8
_LEDGER_MAX_ENTRIES = 10
_LEDGER_MAX_CHARS = 2000


def _ledger_paths(messages: list[dict]) -> list[str]:
    """Paths touched by file tools in *messages*, most recent first, deduped."""
    seen: list[str] = []
    for m in reversed(messages):
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            if fn.get("name") not in _LEDGER_TOOLS:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            for candidate in ([args] if isinstance(args, dict) else []) + list(
                    args.get("chunks", []) if isinstance(args, dict) else []):
                path = candidate.get("path") if isinstance(candidate, dict) else None
                if path and path not in seen:
                    seen.append(path)
    return seen


def _file_ledger(messages: list[dict], config) -> str:
    """Render the ledger section, or "" when there is nothing worth carrying."""
    paths = _ledger_paths(messages)
    if not paths:
        return ""
    try:
        from agent.tools.files.outline import outline as _outline
    except Exception:
        return ""

    working_dir = Path(getattr(getattr(config, "tools", None), "working_dir", ".") or ".")
    lines: list[str] = []
    for path in paths[:_LEDGER_MAX_FILES]:
        fpath = Path(path)
        if not fpath.is_absolute():
            fpath = working_dir / fpath
        try:
            if not fpath.is_file():
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        total = len(text.splitlines())
        entries = _outline(text, max_entries=_LEDGER_MAX_ENTRIES, filename=fpath.name)
        landmarks = ", ".join(
            f"{e['line']}:{e['name']}" for e in entries if e["kind"] != "..."
        )
        line = f"{path} · {total} lines"
        if landmarks:
            line += f" · {landmarks}"
        lines.append(line)
        if sum(len(x) for x in lines) > _LEDGER_MAX_CHARS:
            break
    if not lines:
        return ""
    return (
        "\n\n[FILES ALREADY READ — their contents were summarised away, but they are "
        "unchanged on disk. Line numbers below are current: read the range you need "
        "instead of reading a whole file again.]\n" + "\n".join(lines)
    )


# ── Public entry point ──────────────────────────────────────────────────────


def _is_real_user_turn(m: dict) -> bool:
    """A user message the chat template can anchor on, and that will still be
    there when the request is built.

    `_injected_kind` messages do not qualify even though they carry
    `role: "user"` (core/turn.py:217). The context-budget notice is one, and
    `_apply_budget_notice` deletes every notice at the top of the NEXT turn
    before rebuilding the request:

        kept = [m for m in messages if m.get("_injected_kind") != _BUDGET_NOTICE_KIND]

    So counting it as the user turn is counting a message that is scheduled for
    removal. That is the whole bug, and it is why five earlier hypotheses missed
    it -- each looked at one component, and this only appears in the handoff
    between two:

      1. usage crosses 60% of the budget, so a notice is appended as `user`;
      2. compaction folds the real user question into the ASSISTANT summary and
         keeps the tail verbatim -- the notice included, so the result looks
         valid and the guard stays quiet;
      3. the next turn strips the notice, the history now has no user role at
         all, and the template raises
         `Jinja Exception: No user query found in messages.`

    Observed exactly so in final-logs/Ornith-1.5-35B-A3B-IQ4_NL/runs/
    lp-filter-parens_3/agent.log:

        msgs=43 tail_roles=[assistant,tool,assistant,tool,user]   before
        msgs=7  tail_roles=[assistant,tool,assistant,tool,user]   after compaction
        msgs=8  tail_roles=[tool,assistant,tool,assistant,tool]   notice stripped

    Ignoring injected messages here makes step 2 leave a real user turn behind,
    so step 3 has nothing left to break.
    """
    return m.get("role") == "user" and not m.get("_injected_kind")


def _ensure_user_turn(result: list[dict], messages: list[dict]) -> list[dict]:
    """Guarantee the compacted history still contains a user turn.

    Compaction is not idempotent: the summary goes in as an ASSISTANT message,
    so a pass can fold the user's question into it and leave a history with no
    user role at all. Qwen-style templates -- the ornith line carries the same
    12-line block -- walk the messages backwards looking for the last user
    query and raise when there is none. llama-server turns that into

        500  Jinja Exception: No user query found in messages.

    and the agent reports it as NoUsableModelError, which sends whoever reads
    it looking at the server: the one thing that was working.

    Measured on the 2026-09-08 four-arm run, per-arm error counts matched
    per-arm 500 counts exactly -- 2/2, 2/2, 4/4, and 0/0 for Qwen, whose
    shorter tool chains never reach the state. Every `error` row in that run
    was this and nothing else, which is why the models looked like they
    differed in reliability when they differed in chain length.

    The repair is the smallest one that restores the contract: put the most
    recent real user turn back, verbatim. It only runs when the result is
    already invalid, so a healthy history passes through untouched.
    """
    if any(_is_real_user_turn(m) for m in result):
        return result
    last_user = next((m for m in reversed(messages)
                      if _is_real_user_turn(m)), None)
    if last_user is None:
        return result
    logger.warning("compact: no user message survived — reinstating the last "
                   "one so the chat template can find a query")
    return result + [dict(last_user)]


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
    # Stage 0: release reads the task has moved past (memory/read_release.py).
    # No LLM call; if it frees enough, the summary — and what it loses — is skipped.
    tools_cfg = getattr(config, "tools", None)
    if getattr(tools_cfg, "release_reads", True):
        from agent.memory.read_release import release_reads
        released, freed = release_reads(
            messages, idle_calls=int(getattr(tools_cfg, "release_idle_calls", 12)))
        if freed:
            messages = released
            try:
                from agent.core.context_budget import compaction_trigger_budget
                budget = min(compaction_trigger_budget(config, None),
                             int(_ctx(config) * config.llm.compaction_threshold))
            except Exception:
                budget = int(_ctx(config) * config.llm.compaction_threshold)
            msg_cap = config.llm.compaction_message_threshold
            if msg_cap <= 0:
                msg_cap = max(40, config.llm.ctx_window // 1000)
            token_est = _count_tokens_approx(messages)
            logger.info("compact: released stale reads, %d chars freed, ~%d tokens left (budget %d)",
                        freed, token_est, budget)
            # 80%: returning just under the trigger would compact again next step.
            if token_est <= budget * 0.8 and len(messages) <= msg_cap:
                return messages

    if len(messages) <= keep_last * 2:
        token_est = _count_tokens_approx(messages)
        budget = int(_ctx(config) * config.llm.compaction_threshold)
        if token_est > budget:
            return _truncate_tool_results_in(messages, max_chars=budget * 2)
        return messages

    hard_rules_msgs: list[dict] = []
    # Every system message of the leading run survives (system prompt, project
    # doc, skills index). Keeping only the last one let a later system message —
    # the skills index, a bg-job note — silently replace the system prompt.
    # The project preload is the exception: a start-of-session snapshot, the
    # first thing compaction is meant to drop (core/preload.py).
    system_msgs: list[dict] = []
    conversation = []
    leading = True
    for m in messages:
        if m.get("role") == "system":
            if m.get("_hard_rules_marker"):
                hard_rules_msgs.append(m)
            elif leading and not m.get("_preload_marker"):
                system_msgs.append(m)
        else:
            leading = False
            conversation.append(m)

    verbatim_start = max(len(conversation) - keep_last * 2, 0)
    # Always preserve the most recent user message verbatim, even if a long
    # tool-call burst has scrolled it out of the keep_last window. Otherwise the
    # user's question survives only via q_view and is lost if Stage 2 fails.
    # It is carried as a single message placed before the summary. Pulling
    # verbatim_start back to it instead kept the whole burst verbatim: in an
    # autonomous run with one user message at the top, nothing was ever
    # compacted, the message-count trigger fired again every turn, and each
    # pass rewrote history and threw the prompt cache away.
    last_user_idx = next(
        (
            i
            for i in range(len(conversation) - 1, -1, -1)
            if _is_real_user_turn(conversation[i])
        ),
        None,
    )
    pinned_user: list[dict] = []
    if last_user_idx is not None and last_user_idx < verbatim_start:
        pinned_user = [conversation[last_user_idx]]
    # Don't let the verbatim tail begin on an orphan tool result: its originating
    # assistant (with tool_calls) sits earlier and is about to be compacted away,
    # while the inserted summary assistant carries no tool_calls — a leading tool
    # message would make the next API call 400. Push such results into to_compact.
    while (
        verbatim_start < len(conversation)
        and conversation[verbatim_start].get("role") == "tool"
    ):
        verbatim_start += 1
    to_compact = conversation[:verbatim_start]
    verbatim = conversation[verbatim_start:]

    if not to_compact:
        return _truncate_tool_results_in(messages, max_chars=2000)

    transcript_text = _render_transcript(to_compact)

    # Protect technical entities (file paths, hex values, error codes) so the
    # Stage-1 LLM preserves them verbatim rather than paraphrasing them away.
    _protector = _EntityProtector()
    _entities = _protector.scan_text(transcript_text)
    if _entities:
        transcript_text = _protector.protect_text(transcript_text, _entities)

    prev_round = facts_store.latest_round() if facts_store is not None else None
    from_turn = (prev_round.to_turn + 1) if prev_round else 0
    to_turn = turn_index if turn_index is not None else (from_turn + len(to_compact))

    # Stage 1
    #
    # Guarded, because compaction CALLS THE MODEL, and compaction is itself what
    # run_turn does when the model says the context is full. An exception here
    # is raised from inside that handler, so nothing catches it and the whole
    # turn dies. Seen on a 24-module fixture: the model read its way to 65537
    # tokens against a 65536 window and the run ended with an uncaught
    #   BadRequestError: request (65537 tokens) exceeds the available context
    # which reads as a model failure and is not one.
    #
    # If the analysis cannot run, we still have to return something shorter than
    # what we were given, so fall through to the truncation path below rather
    # than propagating.
    try:
        knowledge_draft = await _analyze_transcript(
            transcript_text, prev_round, config, client
        )
    except Exception as e:                                  # noqa: BLE001
        logger.warning("compact: stage 1 failed (%s: %s) — falling back to "
                       "truncation", type(e).__name__, e)
        result = list(hard_rules_msgs)
        result.extend(system_msgs)
        result.extend(pinned_user)
        result.append({"role": "assistant",
                       "content": f"[SESSION SUMMARY UNAVAILABLE: {type(e).__name__}]",
                       "_compaction_marker": True})
        result.extend(_truncate_tool_results_in(verbatim, max_chars=2000))
        return result
    # Strip entity markers before saving to Tier-2 facts and feeding Stage 2.
    if _entities:
        knowledge_draft = _protector.unprotect_text(knowledge_draft)

    # Stage 2
    try:
        facts, summary, q_view = await _synthesize_summary(
            knowledge_draft, config, client
        )
    except (CompactionError, Exception) as e:               # noqa: BLE001
        # Was CompactionError only, which let an API-level failure -- notably a
        # context-exceeded 400 raised by the summariser itself -- escape the same
        # way stage 1's did.
        logger.warning("compact: stage 2 failed (%s), falling back to error summary: %s",
                       type(e).__name__, e)
        error_msg = {"role": "assistant", "content": f"[SESSION SUMMARY ERROR: {e}]",
                     "_compaction_marker": True}
        result = list(hard_rules_msgs)
        result.extend(system_msgs)
        result.extend(pinned_user)
        result.append(error_msg)
        result.extend(_truncate_tool_results_in(verbatim, max_chars=2000))
        # Same contract as the normal path. This branch is not the rare one:
        # over 80 benchmark runs it fired 5 times and the normal path's guard
        # fired 0, because an early `return` here skipped it. Every 500 we saw
        # came through here.
        return _ensure_user_turn(result, messages)

    # Propagate original_request from previous round if synthesizer dropped it.
    prev_original = (prev_round.facts or {}).get("original_request", "") if prev_round else ""
    if prev_original and not facts.get("original_request"):
        facts["original_request"] = prev_original
    # Fall back to durable store (written on first user turn in agent.py).
    if not facts.get("original_request") and facts_store is not None:
        durable = facts_store.get_original_request()
        if durable:
            facts["original_request"] = durable

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

    # Build the compacted system-summary message. Label it with the round this
    # summary was *derived from* (the one just saved), not the previous round —
    # otherwise the number and the recall_facts(round_id=...) hint point one
    # round behind the facts shown here.
    display_round_id = saved_round.round_id if saved_round is not None else None
    header = "[SESSION SUMMARY"
    if display_round_id is not None:
        header += f" · round {display_round_id}"
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
    try:
        compacted_content += _file_ledger(to_compact, config)
    except Exception:
        logger.debug("compact: file ledger failed (ignored)", exc_info=True)

    # Marked so a replayed session can say where history was compacted rather
    # than silently showing fewer rounds than the live view had.
    compacted_msg = {"role": "assistant", "content": compacted_content,
                     "_compaction_marker": True}

    verbatim = _truncate_tool_results_in(verbatim, max_chars=2000)

    result: list[dict] = list(hard_rules_msgs)
    result.extend(system_msgs)
    result.extend(pinned_user)
    result.append(compacted_msg)
    result.extend(verbatim)

    return _ensure_user_turn(result, messages)
