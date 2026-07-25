"""The backlog store, with the parts added for the UI: rank and migration.

The `rank` column arrived after databases already existed in the field, so the
migration path is the important test here: an old .agent/ideas.db must open,
keep its contents, and come back in the order it already had.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from agent.ideas.store import IdeasStore


@pytest.fixture()
def store(tmp_path):
    return IdeasStore(tmp_path / "ideas.db")


def _titles(store):
    return [i["title"] for i in store.list()]


class TestMigration:
    def _legacy_db(self, path):
        """A database exactly as the pre-rank version created it."""
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE ideas (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, type TEXT NOT NULL DEFAULT 'idea',
            status TEXT NOT NULL DEFAULT 'raw', priority INTEGER DEFAULT 3,
            effort_score REAL, value_score REAL, tags TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL DEFAULT 'human', created_at REAL NOT NULL,
            updated_at REAL NOT NULL, body TEXT NOT NULL DEFAULT '',
            requirements_ref TEXT, plan_ref TEXT, session_ref TEXT, project TEXT)""")
        now = time.time()
        for i, title in enumerate(["oldest", "middle", "newest"]):
            con.execute("INSERT INTO ideas (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                        (f"id{i}", title, now + i, now + i))
        con.commit()
        con.close()

    def test_an_old_database_opens_and_keeps_its_rows(self, tmp_path):
        path = tmp_path / "ideas.db"
        self._legacy_db(path)
        store = IdeasStore(path)
        assert store.count() == 3

    def test_it_opens_in_the_order_it_already_had(self, tmp_path):
        """Backfilled ranks come from created_at, so nothing appears to have
        been shuffled by the upgrade."""
        path = tmp_path / "ideas.db"
        self._legacy_db(path)
        assert _titles(IdeasStore(path)) == ["newest", "middle", "oldest"]

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "ideas.db"
        self._legacy_db(path)
        IdeasStore(path)
        store = IdeasStore(path)
        assert store.count() == 3
        assert _titles(store) == ["newest", "middle", "oldest"]

    def test_a_migrated_row_can_then_be_reordered(self, tmp_path):
        path = tmp_path / "ideas.db"
        self._legacy_db(path)
        store = IdeasStore(path)
        assert store.reorder("id2", after="id0") is True     # newest → after oldest
        assert _titles(store) == ["middle", "oldest", "newest"]


class TestRank:
    def test_add_puts_the_new_item_first(self, store):
        store.add(title="one")
        store.add(title="two")
        assert _titles(store) == ["two", "one"]

    def test_reorder_to_the_top(self, store):
        store.add(title="a")
        b = store.add(title="b")
        store.add(title="c")            # c, b, a
        assert store.reorder(b) is True  # neither neighbour = to the top
        assert _titles(store) == ["b", "c", "a"]

    def test_reorder_to_the_bottom(self, store):
        a = store.add(title="a")
        store.add(title="b")
        c = store.add(title="c")        # c, b, a
        store.reorder(c, after=a)
        assert _titles(store) == ["b", "a", "c"]

    def test_status_filtered_listing_keeps_the_manual_order(self, store):
        a = store.add(title="a")
        b = store.add(title="b")
        store.reorder(a, before=b)
        assert [i["title"] for i in store.list(status="raw")] == ["a", "b"]

    def test_an_export_round_trip_preserves_the_order(self, store, tmp_path):
        a = store.add(title="a")
        store.add(title="b")
        store.reorder(a, before="")     # to the top
        exported = store.list()
        other = IdeasStore(tmp_path / "other.db")
        for record in exported:
            other.upsert(record)
        assert _titles(other) == _titles(store)

    def test_an_imported_record_without_a_rank_still_sorts_sanely(self, store):
        """Records from an external tracker will not have one."""
        store.upsert({"id": "x1", "title": "older", "created_at": 1000.0})
        store.upsert({"id": "x2", "title": "newer", "created_at": 2000.0})
        assert _titles(store) == ["newer", "older"]
