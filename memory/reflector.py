"""Session-end reflective pass: extract behavioral rules from agent mistakes and user corrections.

Complements promoter.py (project facts) — this module covers agent behavior only:
- User corrections detected in transcript
- Repeated tool/approach mistakes in this session
- Failure-derived patterns from .agent/failures/

Rules stored as scope='behavioral_rule' in project MemoryStore.
hit_count tracks corroboration: rules with hit_count >= 2 get hard-injected
at session start; one-offs (hit_count == 1) stay as soft/relevance notes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.memory.facts_store import FactsStore
    from agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Vector similarity thresholds — tunable, embedding-model-dependent.
_SKIP_THRESHOLD = 0.88   # cosine similarity: above → near-duplicate, skip
_MERGE_THRESHOLD = 0.72  # cosine similarity: above → corroborate existing rule

# FTS BM25 thresholds — tunable, corpus-dependent.
# BM25 in sqlite-fts5 is negative; MORE negative = better match.
_FTS_SKIP_THRESHOLD = -1.5   # below (more negative) → near-duplicate, skip
_FTS_MERGE_THRESHOLD = -4.0  # below (more negative) → corroborate

# Read last N bytes of index.jsonl to avoid O(total-history) scan.
_FAILURES_TAIL_BYTES = 65536

_REFLECT_PROMPT = """\
You are analyzing a coding session to extract behavioral rules for an AI coding agent.

Focus ONLY on agent behavior — NOT project facts (those are captured separately).
Look for:
1. Moments where the user explicitly corrected the agent ("no", "don't", "wrong", "instead", "stop")
2. Agent mistakes that wasted turns: wrong assumption, wrong tool, looping on same error
3. Failure-derived patterns: what the agent should have done differently given the failures

Rules must be:
- Agent-action oriented ("When X, do Y" / "Never Z" / "Always check W before V")
- Durable across sessions, not specific to this task's content
- Non-obvious (skip: "read the file before editing", "run tests after changes")

Output JSON only:
{"rules": [{"rule": "...", "category": "correction|mistake|procedure", "confidence": 0.9}]}

Max 5 rules. Confidence must be a number 0.0-1.0, >= 0.75 only.
Skip rules about project-specific facts.
Output {"rules": []} if nothing qualifies.
"""


def reflect_session(
    session_id: str,
    config: "Config",
    facts_store: "FactsStore | None" = None,
    embedder=None,
    store: "MemoryStore | None" = None,
) -> int:
    """Extract behavioral rules from session; persist to project MemoryStore.

    Returns number of rules saved/corroborated (0 on skip or error).
    store: optional pre-opened project MemoryStore (avoids redundant connection).
    """
    if facts_store is None:
        return 0

    latest = facts_store.latest_round()
    source_text = ""
    if latest is not None:
        source_text = latest.knowledge_draft or latest.summary or ""

    failure_summary = _read_session_failures(config, session_id)

    if len(source_text) < 50 and not failure_summary:
        return 0

    prompt_input = source_text[:5000]
    if failure_summary:
        prompt_input += f"\n\n[FAILURES THIS SESSION]\n{failure_summary}"

    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": _REFLECT_PROMPT},
                {"role": "user", "content": prompt_input},
            ],
            max_tokens=600,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("reflect_session: LLM call failed: %s", e)
        return 0

    rules_list = _parse_rules(raw)
    if not rules_list:
        return 0

    _store_opened = False
    if store is None:
        try:
            from agent.memory.store import MemoryStore
            agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
            store = MemoryStore(agent_dir / "memory.db")
            _store_opened = True
        except Exception as e:
            logger.debug("reflect_session: store init failed: %s", e)
            return 0

    try:
        return _process_rules(rules_list, session_id, store, embedder)
    finally:
        if _store_opened:
            try:
                store.close()
            except Exception:
                pass


def _process_rules(rules_list: list, session_id: str, store, embedder) -> int:
    saved = 0
    for item in rules_list[:5]:
        rule_text = (item.get("rule") or "").strip()
        category = (item.get("category") or "behavior").strip()
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            logger.debug("reflect_session: non-numeric confidence %r, skipping rule", item.get("confidence"))
            continue
        if not rule_text or confidence < 0.75:
            continue

        embedding = None
        if embedder is not None:
            try:
                embedding = embedder.embed_one(rule_text[:1000])
            except Exception:
                pass

        action = _dedup_action(store, rule_text, embedding)
        if action == "skip":
            logger.debug("reflect_session: skipping near-duplicate rule: %.60s", rule_text)
            continue
        if action is not None:
            store.increment_hit_count(action)
            logger.debug("reflect_session: corroborated rule %s", action)
            saved += 1
            continue

        # New rule: insert then immediately increment so hit_count starts at 1
        # (represents "seen once"). hit_count==1 → soft inject; ==2 → hard inject.
        tags = [f"category:{category}", f"auto:{session_id[:8]}"]
        try:
            entry_id = store.add(
                scope="behavioral_rule",
                body=rule_text,
                title=rule_text[:80],
                tags=tags,
                embedding=embedding,
            )
            store.increment_hit_count(entry_id)
            saved += 1
        except Exception:
            pass

    if saved:
        logger.info("reflect_session: %d rule(s) saved/corroborated for session %s", saved, session_id[:16])
    return saved


def _dedup_action(store, rule_text: str, embedding) -> str | None:
    """Return entry_id to merge into, 'skip' if near-duplicate, None if new."""
    # Vector dedup (preferred): cosine similarity, higher = more similar.
    if embedding is not None:
        results = store.vector_search(embedding, scope="behavioral_rule", top_k=1)
        if results:
            score = results[0].get("score", 0.0)
            if score >= _SKIP_THRESHOLD:
                return "skip"
            if score >= _MERGE_THRESHOLD:
                return results[0]["id"]

    # FTS fallback: BM25 is negative, more negative = better match.
    fts = store.fts_search(rule_text, scope="behavioral_rule", top_k=1)
    if fts:
        bm25 = fts[0].get("score", 0.0)
        if bm25 < _FTS_SKIP_THRESHOLD:
            return "skip"
        if bm25 < _FTS_MERGE_THRESHOLD:
            return fts[0]["id"]

    return None


def _read_session_failures(config: "Config", session_id: str) -> str:
    """Read failure entries for session_id from the tail of index.jsonl."""
    try:
        agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
        index_path = agent_dir / "failures" / "index.jsonl"
        if not index_path.exists():
            return ""

        file_size = index_path.stat().st_size
        lines_raw: list[str] = []
        with index_path.open("rb") as fbin:
            if file_size > _FAILURES_TAIL_BYTES:
                fbin.seek(-_FAILURES_TAIL_BYTES, 2)
                fbin.readline()  # discard partial first line
            lines_raw = fbin.read().decode("utf-8", errors="replace").splitlines()

        lines = []
        for line in lines_raw:
            try:
                rec = json.loads(line)
                if rec.get("session_id") == session_id:
                    kind = rec.get("kind", "")
                    tool = rec.get("tool") or ""
                    reason = rec.get("reason") or rec.get("error") or ""
                    lines.append(f"- [{kind}] tool={tool}: {str(reason)[:120]}")
            except Exception:
                pass
        return "\n".join(lines[:20])
    except Exception:
        return ""


def _parse_rules(raw: str) -> list[dict]:
    """Parse LLM JSON output, tolerating markdown fences and bare arrays."""
    import re

    def _extract(data) -> list[dict]:
        if isinstance(data, dict):
            return data.get("rules") or []
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []

    try:
        return _extract(json.loads(raw))
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if m:
        try:
            return _extract(json.loads(m.group(1)))
        except Exception:
            pass

    return []
