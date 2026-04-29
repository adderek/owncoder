"""Session-end fact promotion.

After a session ends cleanly, extracts 3-5 durable facts from the final
compaction round's knowledge_draft and saves them as a note in the project
MemoryStore. This is how session knowledge persists across sessions without
manual intervention.

The LLM call is synchronous (blocking) and uses the same model as the agent.
It runs in the session teardown path, so latency is acceptable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import Config
    from agent.memory.facts_store import FactsStore

logger = logging.getLogger(__name__)

_PROMOTE_PROMPT = """\
You are extracting durable project knowledge from a coding session.

Given the session's knowledge draft, identify 3-5 facts worth remembering
across future sessions. Focus on:
- Architectural decisions and their rationale
- User preferences and constraints ("always X", "never Y")
- Resolved ambiguities about the codebase
- Non-obvious conventions or patterns

Skip: routine edits, test runs, bug fixes with no lasting lesson.

Output JSON only:
{"notes": [{"title": "...", "body": "...", "tags": ["..."]}]}

Max 3 notes. Each body ≤ 120 words. Skip if nothing durable was learned.
Output {"notes": []} if nothing qualifies.
"""


def promote_session_to_notes(
    session_id: str,
    config: "Config",
    facts_store: "FactsStore | None" = None,
    embedder=None,
) -> int:
    """Extract durable facts from final compaction round; save as notes.

    Returns number of notes saved (0 on skip or error).
    """
    if facts_store is None:
        return 0

    latest = facts_store.latest_round()
    if latest is None or not (latest.knowledge_draft or latest.summary):
        return 0

    # Use knowledge_draft if available, fall back to summary.
    source_text = latest.knowledge_draft or latest.summary
    if len(source_text) < 50:
        return 0

    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.llm.base_url, api_key=config.llm.api_key)
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": _PROMOTE_PROMPT},
                {"role": "user", "content": source_text[:6000]},
            ],
            max_tokens=800,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.debug("promote_session_to_notes: LLM call failed: %s", e)
        return 0

    try:
        data = json.loads(raw)
        notes_list = data.get("notes") or []
    except Exception:
        # Try to extract JSON from markdown code block
        import re
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                notes_list = data.get("notes") or []
            except Exception:
                return 0
        else:
            return 0

    if not notes_list:
        return 0

    try:
        from agent.memory.store import MemoryStore
        agent_dir = Path(config.tools.working_dir) / config.tools.agent_dir
        store = MemoryStore(agent_dir / "memory.db")
    except Exception as e:
        logger.debug("promote_session_to_notes: store init failed: %s", e)
        return 0

    saved = 0
    for note in notes_list[:3]:
        title = (note.get("title") or "").strip()
        body = (note.get("body") or "").strip()
        if not title or not body:
            continue
        tags = list(note.get("tags") or []) + [f"auto:{session_id[:8]}"]
        embedding = None
        if embedder is not None:
            try:
                embedding = embedder.embed_one(f"{title}\n\n{body}"[:2000])
            except Exception:
                pass
        try:
            store.add(
                scope="note",
                body=body,
                title=title,
                tags=tags,
                embedding=embedding,
            )
            saved += 1
        except Exception:
            pass

    if saved:
        logger.info("promote_session_to_notes: saved %d notes from session %s", saved, session_id[:16])
    return saved
