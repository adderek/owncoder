"""Tier 2 "Source Facts" storage for two-stage compaction.

Each compaction round produces:
  * a detailed `knowledge_draft` (Stage 1, the "thinking" output) — kept on disk
    and *not* injected into the active context.
  * a compressed `summary` + structured `facts` (Stage 2) — what lands in the
    live context as [SESSION SUMMARY].

When the agent realises the live summary lacks detail it needs, it can call
the `recall_facts` tool, which reads from this store.

Layout:
    <session-dir>/<session-id>/facts/round-0001.json
    <session-dir>/<session-id>/facts/round-0002.json
    <session-dir>/<session-id>/facts/latest.txt   ← plain int, the latest round id
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.memory.session import _get_session_dir, get_session_subpath


@dataclass
class FactsRound:
    round_id: int
    timestamp: str
    from_turn: int
    to_turn: int
    prev_round_id: int | None = None
    # Summary of the *previous* round fed into Stage 1 (audit trail).
    prev_summary: str = ""
    # Stage 1 output — long-form, detailed. Tier 2 source of truth.
    knowledge_draft: str = ""
    # Stage 2 output — compressed summary inserted into live context.
    summary: str = ""
    # Stage 2 output — compressed refined user intent / outstanding request.
    q_view: str = ""
    # Stage 2 structured extract (files_modified, decisions, open_issues, ...).
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FactsRound":
        return cls(
            round_id=int(data.get("round_id", 0)),
            timestamp=str(data.get("timestamp", "")),
            from_turn=int(data.get("from_turn", 0)),
            to_turn=int(data.get("to_turn", 0)),
            prev_round_id=data.get("prev_round_id"),
            prev_summary=str(data.get("prev_summary", "")),
            knowledge_draft=str(data.get("knowledge_draft", "")),
            summary=str(data.get("summary", "")),
            q_view=str(data.get("q_view", "")),
            facts=dict(data.get("facts") or {}),
        )


_ROUND_FILE_RE = re.compile(r"^round-(\d+)\.json$")


class FactsStore:
    """Per-session on-disk store of compaction rounds."""

    def __init__(self, session_id: str, base_dir: Path | None = None, embedder=None):
        self.session_id = session_id
        base = base_dir if base_dir is not None else (_get_session_dir() / get_session_subpath(session_id))
        self.dir = base / "facts"
        self._embedder = embedder
        self._mem_store: Any | None = None
        if embedder is not None:
            from agent.memory.store import MemoryStore
            self._mem_store = MemoryStore(base / "memory.db")

    # ── internal paths ──────────────────────────────────────────────────────
    def _round_path(self, round_id: int) -> Path:
        return self.dir / f"round-{round_id:04d}.json"

    def _latest_pointer(self) -> Path:
        return self.dir / "latest.txt"

    # ── discovery ───────────────────────────────────────────────────────────
    def next_round_id(self) -> int:
        latest = self.latest_round_id()
        return (latest or 0) + 1

    def latest_round_id(self) -> int | None:
        pointer = self._latest_pointer()
        if pointer.exists():
            try:
                return int(pointer.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        # Fallback: scan directory.
        if not self.dir.exists():
            return None
        ids: list[int] = []
        for p in self.dir.glob("round-*.json"):
            m = _ROUND_FILE_RE.match(p.name)
            if m:
                ids.append(int(m.group(1)))
        return max(ids) if ids else None

    def list_round_ids(self) -> list[int]:
        if not self.dir.exists():
            return []
        ids: list[int] = []
        for p in self.dir.glob("round-*.json"):
            m = _ROUND_FILE_RE.match(p.name)
            if m:
                ids.append(int(m.group(1)))
        return sorted(ids)

    # ── read ────────────────────────────────────────────────────────────────
    def load_round(self, round_id: int) -> FactsRound | None:
        p = self._round_path(round_id)
        if not p.exists():
            return None
        try:
            return FactsRound.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return None

    def latest_round(self) -> FactsRound | None:
        rid = self.latest_round_id()
        return self.load_round(rid) if rid is not None else None

    def iter_rounds(self):
        for rid in self.list_round_ids():
            r = self.load_round(rid)
            if r is not None:
                yield r

    # ── write ───────────────────────────────────────────────────────────────
    def save_round(self, r: FactsRound) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._round_path(r.round_id)
        path.write_text(
            json.dumps(r.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._latest_pointer().write_text(str(r.round_id), encoding="utf-8")
        return path

    def new_round(
        self,
        *,
        from_turn: int,
        to_turn: int,
        knowledge_draft: str,
        summary: str,
        q_view: str = "",
        facts: dict[str, Any] | None = None,
        prev: FactsRound | None = None,
    ) -> FactsRound:
        rid = self.next_round_id()
        r = FactsRound(
            round_id=rid,
            timestamp=datetime.now(timezone.utc).isoformat(),
            from_turn=from_turn,
            to_turn=to_turn,
            prev_round_id=prev.round_id if prev else None,
            prev_summary=prev.summary if prev else "",
            knowledge_draft=knowledge_draft,
            summary=summary,
            q_view=q_view,
            facts=dict(facts or {}),
        )
        self.save_round(r)
        self._index_round(r)
        return r

    def _index_round(self, r: FactsRound) -> None:
        """Embed and store round in MemoryStore for semantic recall."""
        if self._mem_store is None or self._embedder is None:
            return
        body = "\n\n".join(filter(None, [r.knowledge_draft, r.summary, r.q_view]))
        if not body.strip():
            return
        try:
            embedding = self._embedder.embed_one(body[:8000])
            self._mem_store.add(
                scope="facts_round",
                body=body,
                source=str(r.round_id),
                title=f"Round {r.round_id} turns {r.from_turn}–{r.to_turn}",
                embedding=embedding if embedding else None,
                entry_id=f"{self.session_id}:round:{r.round_id}",
            )
        except Exception:
            pass

    # ── recall / search ─────────────────────────────────────────────────────
    def search(
        self,
        query: str,
        *,
        round_id: int | None = None,
        max_results: int = 3,
        snippet_chars: int = 800,
    ) -> list[dict[str, Any]]:
        """Naive keyword search across knowledge_draft and facts of saved rounds.

        Returns a list of hits, each with:
            round_id, timestamp, from_turn, to_turn, snippet, score
        Score is the number of query-term occurrences.
        """
        terms = [t for t in re.split(r"\W+", (query or "").lower()) if len(t) >= 2]
        if not terms:
            return []

        rounds: list[FactsRound]
        if round_id is not None:
            r = self.load_round(round_id)
            rounds = [r] if r else []
        else:
            rounds = list(self.iter_rounds())

        hits: list[dict[str, Any]] = []
        for r in rounds:
            haystack = "\n".join([
                r.knowledge_draft or "",
                r.summary or "",
                r.q_view or "",
                json.dumps(r.facts, ensure_ascii=False) if r.facts else "",
            ])
            hay_lower = haystack.lower()
            score = sum(hay_lower.count(t) for t in terms)
            if score <= 0:
                continue
            # Build a snippet around the first match.
            first_idx = min(
                (hay_lower.find(t) for t in terms if hay_lower.find(t) != -1),
                default=0,
            )
            start = max(0, first_idx - snippet_chars // 3)
            snippet = haystack[start : start + snippet_chars]
            hits.append({
                "round_id": r.round_id,
                "timestamp": r.timestamp,
                "from_turn": r.from_turn,
                "to_turn": r.to_turn,
                "score": score,
                "snippet": snippet,
            })
        hits.sort(key=lambda h: (-h["score"], -h["round_id"]))
        return hits[:max_results]

    def semantic_search(
        self,
        query: str,
        *,
        max_results: int = 3,
        snippet_chars: int = 800,
    ) -> list[dict[str, Any]]:
        """Vector search over indexed rounds via MemoryStore.

        Falls back to keyword search when embedder/MemoryStore unavailable.
        Returns same shape as search().
        """
        if self._mem_store is None or self._embedder is None:
            return self.search(query, max_results=max_results, snippet_chars=snippet_chars)
        try:
            embedding = self._embedder.embed_one(query)
        except Exception:
            return self.search(query, max_results=max_results, snippet_chars=snippet_chars)
        mem_hits = self._mem_store.hybrid_search(
            query, embedding=embedding if embedding else None,
            scope="facts_round", top_k=max_results,
        )
        results = []
        for h in mem_hits:
            try:
                round_id = int(h.get("source", 0))
            except (ValueError, TypeError):
                continue
            r = self.load_round(round_id)
            if r is None:
                continue
            body = h.get("body", "")
            snippet = body[:snippet_chars]
            results.append({
                "round_id": r.round_id,
                "timestamp": r.timestamp,
                "from_turn": r.from_turn,
                "to_turn": r.to_turn,
                "score": h.get("combined_score", h.get("score", 0.0)),
                "snippet": snippet,
            })
        return results
