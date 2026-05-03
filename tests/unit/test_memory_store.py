"""Unit tests for agent/memory/store.py — MemoryStore."""
from __future__ import annotations

from pathlib import Path

import pytest
from agent.memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.db")


class TestAddAndGet:
    def test_add_returns_id(self, store):
        eid = store.add("note", "hello world", title="test")
        assert isinstance(eid, str) and len(eid) > 0

    def test_get_round_trip(self, store):
        eid = store.add("note", "body text", title="t", tags=["a", "b"], source="s")
        entry = store.get(eid)
        assert entry is not None
        assert entry["body"] == "body text"
        assert entry["title"] == "t"
        assert entry["scope"] == "note"
        assert entry["source"] == "s"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent-id") is None

    def test_explicit_id(self, store):
        store.add("note", "body", entry_id="my-id")
        assert store.get("my-id") is not None

    def test_upsert_replaces(self, store):
        store.add("note", "v1", entry_id="eid")
        store.add("note", "v2", entry_id="eid")
        assert store.get("eid")["body"] == "v2"


class TestDelete:
    def test_delete_removes_entry(self, store):
        eid = store.add("note", "gone")
        store.delete(eid)
        assert store.get(eid) is None

    def test_delete_nonexistent_ok(self, store):
        store.delete("no-such-id")  # must not raise


class TestListEntries:
    def test_list_all(self, store):
        store.add("note", "a")
        store.add("note", "b")
        store.add("facts_round", "c")
        assert len(store.list_entries()) == 3

    def test_list_by_scope(self, store):
        store.add("note", "a")
        store.add("facts_round", "b")
        notes = store.list_entries(scope="note")
        assert len(notes) == 1
        assert notes[0]["scope"] == "note"

    def test_limit(self, store):
        for i in range(5):
            store.add("note", f"entry {i}")
        assert len(store.list_entries(limit=3)) == 3


class TestFTSSearch:
    def test_basic_keyword_hit(self, store):
        store.add("note", "validate_email was added to auth.py")
        hits = store.fts_search("validate_email")
        assert len(hits) == 1

    def test_scope_filter(self, store):
        store.add("note", "alpha beta")
        store.add("facts_round", "alpha gamma")
        hits = store.fts_search("alpha", scope="note")
        assert len(hits) == 1
        assert hits[0]["scope"] == "note"

    def test_empty_query_returns_empty(self, store):
        store.add("note", "something")
        assert store.fts_search("") == []
        assert store.fts_search("   ") == []

    def test_no_match_returns_empty(self, store):
        store.add("note", "bananas only here")
        assert store.fts_search("oranges") == []


class TestTagFilter:
    def test_fts_tags_filter_matches(self, store):
        store.add("session_summary", "refactor auth module", source="s1", tags=["outcome:good"])
        store.add("session_summary", "fix login bug", source="s2", tags=["outcome:bad"])
        hits = store.fts_search("auth", scope="session_summary", tags_filter=["outcome:good"])
        assert len(hits) == 1
        assert hits[0]["source"] == "s1"

    def test_fts_tags_filter_excludes(self, store):
        store.add("session_summary", "deploy pipeline", source="s1", tags=["outcome:bad"])
        hits = store.fts_search("deploy", scope="session_summary", tags_filter=["outcome:good"])
        assert hits == []

    def test_hybrid_tags_filter(self, store):
        store.add("session_summary", "implement caching layer", source="s1", tags=["outcome:good"])
        store.add("session_summary", "implement retry logic", source="s2", tags=["outcome:bad"])
        hits = store.hybrid_search("implement", scope="session_summary", tags_filter=["outcome:good"])
        assert len(hits) == 1
        assert hits[0]["source"] == "s1"

    def test_hybrid_no_filter_returns_all(self, store):
        store.add("session_summary", "task alpha", source="s1", tags=["outcome:good"])
        store.add("session_summary", "task beta", source="s2", tags=["outcome:bad"])
        hits = store.hybrid_search("task", scope="session_summary")
        assert len(hits) == 2

    def test_no_filter_param_returns_all(self, store):
        store.add("note", "alpha", tags=["x"])
        store.add("note", "alpha", tags=["y"])
        hits = store.fts_search("alpha")
        assert len(hits) == 2


class TestUpdateSourceTags:
    def test_update_changes_tags(self, store):
        store.add("session_summary", "session one summary", source="sess-1", entry_id="e1")
        store.add("session_summary", "session one detail", source="sess-1", entry_id="e2")
        updated = store.update_source_tags("session_summary", "sess-1", ["outcome:good"])
        assert updated == 2
        e = store.get("e1")
        import json
        assert "outcome:good" in json.loads(e["tags"])

    def test_update_unmatched_source_returns_zero(self, store):
        store.add("session_summary", "stuff", source="other")
        assert store.update_source_tags("session_summary", "missing", ["outcome:good"]) == 0

    def test_updated_tags_searchable_by_fts(self, store):
        store.add("session_summary", "some work done", source="s99", entry_id="e99")
        store.update_source_tags("session_summary", "s99", ["outcome:good"])
        hits = store.fts_search("work", scope="session_summary", tags_filter=["outcome:good"])
        assert any(h["id"] == "e99" for h in hits)


class TestVectorSearch:
    def test_vector_search_no_dims_returns_empty(self, store):
        # No embeddings stored yet — should return empty list gracefully.
        result = store.vector_search([0.1, 0.2, 0.3])
        assert result == []

    def test_hybrid_falls_back_to_fts(self, store):
        store.add("note", "recall this important fact")
        # No embeddings — hybrid should still return FTS results.
        hits = store.hybrid_search("important", embedding=None)
        assert len(hits) >= 1
