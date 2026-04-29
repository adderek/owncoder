"""`recall_facts` tool — retrieve detail from Tier-2 facts storage.

The two-stage compactor writes a detailed `knowledge_draft` per round to
`<session>/facts/round-NNNN.json`. Only a compressed summary of each round
lives in the active context. When the model needs specifics that have been
elided from the summary — an exact filename, an earlier decision, an error
it handled three rounds ago — it can call this tool to fetch them without
having to re-explore the codebase.
"""

from __future__ import annotations

from typing import Any

from agent.tools import register


_facts_store = None  # Set by setup() when a session is active.


def setup(facts_store) -> None:
    """Wire the per-session FactsStore in. Called by Agent.set_session_id."""
    global _facts_store
    _facts_store = facts_store


@register(
    "recall_facts",
    {
        "description": (
            "Retrieve detailed facts from earlier in this session that have been "
            "compressed out of the active context. The current context shows a "
            "[SESSION SUMMARY] block; when that summary lacks a specific detail "
            "(filename, decision, error, signature) you need to answer accurately, "
            "call this tool. Returns excerpts from the detailed Tier-2 knowledge "
            "drafts saved after each compaction round."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords describing the missing detail — e.g. a filename, "
                        "function name, error message, or decision topic."
                    ),
                },
                "round_id": {
                    "type": "integer",
                    "description": (
                        "Optional: restrict the search to a single round id "
                        "(see round numbers shown in the [SESSION SUMMARY] header)."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matching excerpts to return. Default 3.",
                },
            },
            "required": ["query"],
        },
    },
)
def recall_facts(
    query: str,
    round_id: int | None = None,
    max_results: int = 3,
) -> dict[str, Any]:
    if _facts_store is None:
        return {
            "error": "No facts store configured for this session.",
            "hint": "Tier-2 recall requires a session id; start a session first.",
        }
    if not (query or "").strip():
        return {"error": "`query` is required and must be non-empty."}

    try:
        max_results = max(1, min(int(max_results or 3), 10))
    except Exception:
        max_results = 3

    # Use semantic (vector) search when embedder is wired in; keyword fallback.
    if round_id is not None:
        hits = _facts_store.search(query, round_id=round_id, max_results=max_results)
    else:
        hits = _facts_store.semantic_search(query, max_results=max_results)

    rounds_available = _facts_store.list_round_ids()
    if not hits:
        return {
            "query": query,
            "matches": [],
            "rounds_available": rounds_available,
            "hint": (
                "No match. Try broader terms, or pass round_id to see "
                "a full round's detail."
                if rounds_available
                else "No compaction rounds have been saved yet."
            ),
        }
    return {
        "query": query,
        "matches": hits,
        "rounds_available": rounds_available,
    }
