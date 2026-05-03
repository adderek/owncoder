"""Unit tests for agent/memory/facts_store.py — Tier-2 round persistence & search."""
from __future__ import annotations

from pathlib import Path

from agent.memory.facts_store import FactsRound, FactsStore


def _store(tmp_path: Path) -> FactsStore:
    return FactsStore(session_id="test-session", base_dir=tmp_path)


class TestRoundRoundTrip:
    def test_round_to_dict_and_back(self):
        r = FactsRound(
            round_id=3,
            timestamp="2026-04-17T00:00:00+00:00",
            from_turn=5,
            to_turn=10,
            prev_round_id=2,
            prev_summary="prev",
            knowledge_draft="lots of text",
            summary="short",
            q_view="user wants x",
            facts={"files_modified": ["a.py"]},
        )
        r2 = FactsRound.from_dict(r.to_dict())
        assert r2 == r


class TestSaveAndLatest:
    def test_latest_none_when_empty(self, tmp_path):
        s = _store(tmp_path)
        assert s.latest_round_id() is None
        assert s.latest_round() is None
        assert s.list_round_ids() == []

    def test_new_round_monotonic(self, tmp_path):
        s = _store(tmp_path)
        r1 = s.new_round(
            from_turn=0, to_turn=3,
            knowledge_draft="d1", summary="s1", q_view="q1",
            facts={"k": 1},
        )
        r2 = s.new_round(
            from_turn=4, to_turn=7,
            knowledge_draft="d2", summary="s2", q_view="q2",
            facts={"k": 2}, prev=r1,
        )
        assert r1.round_id == 1
        assert r2.round_id == 2
        assert r2.prev_round_id == 1
        assert r2.prev_summary == "s1"
        assert s.latest_round_id() == 2
        assert s.list_round_ids() == [1, 2]

    def test_round_persists_to_disk(self, tmp_path):
        s = _store(tmp_path)
        r = s.new_round(
            from_turn=0, to_turn=1,
            knowledge_draft="draft", summary="sum", q_view="q",
            facts={"a": "b"},
        )
        # Open a fresh store and confirm it sees the round.
        s2 = FactsStore("test-session", base_dir=tmp_path)
        loaded = s2.load_round(r.round_id)
        assert loaded is not None
        assert loaded.knowledge_draft == "draft"
        assert loaded.facts == {"a": "b"}

    def test_latest_pointer_updates(self, tmp_path):
        s = _store(tmp_path)
        s.new_round(from_turn=0, to_turn=1, knowledge_draft="d", summary="")
        s.new_round(from_turn=2, to_turn=3, knowledge_draft="d2", summary="")
        # Simulate a fresh process: construct new store, no cache.
        s3 = FactsStore("test-session", base_dir=tmp_path)
        assert s3.latest_round_id() == 2


class TestSearch:
    def test_hits_ranked_by_score(self, tmp_path):
        s = _store(tmp_path)
        s.new_round(
            from_turn=0, to_turn=2,
            knowledge_draft="We modified foo.py to add validate_email().",
            summary="edit to foo",
        )
        s.new_round(
            from_turn=3, to_turn=5,
            knowledge_draft="Added tests/test_foo.py for validate_email; "
                            "validate_email now rejects empty.",
            summary="tests",
        )
        hits = s.search("validate_email")
        assert len(hits) == 2
        # Round 2 mentions it twice, should outrank round 1.
        assert hits[0]["round_id"] == 2
        assert hits[0]["score"] >= hits[1]["score"]

    def test_empty_query(self, tmp_path):
        s = _store(tmp_path)
        s.new_round(from_turn=0, to_turn=1, knowledge_draft="x", summary="")
        assert s.search("") == []
        assert s.search("   ") == []

    def test_restrict_to_round(self, tmp_path):
        s = _store(tmp_path)
        s.new_round(from_turn=0, to_turn=1,
                    knowledge_draft="alpha", summary="")
        s.new_round(from_turn=2, to_turn=3,
                    knowledge_draft="alpha beta", summary="")
        hits = s.search("alpha", round_id=1)
        assert len(hits) == 1
        assert hits[0]["round_id"] == 1

    def test_no_match_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        s.new_round(from_turn=0, to_turn=1,
                    knowledge_draft="only apples here", summary="")
        assert s.search("bananas") == []
