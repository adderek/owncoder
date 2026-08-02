"""Browsing the memory tiers, not just counting them.

The overview says "122 notes"; the only way to read one was a tool call or
sqlite. These are the reads the memory view is built on.
"""
import shutil
import sqlite3
from pathlib import Path

import pytest

from agent.config.models import Config
from agent.memory import browse
from agent.memory.store import MemoryStore

KB_FIXTURE = Path(__file__).resolve().parents[3] / "kb" / "fixtures" / "example_corpus"


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    return cfg


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / ".agent" / "memory.db")


def _keys(tiers):
    return [t["key"] for t in tiers]


class TestTiers:
    def test_empty_tiers_are_still_listed(self, config):
        """A rail that changes shape as data arrives is harder to navigate."""
        keys = _keys(browse.tiers(config))
        assert "note" in keys and "session_summary" in keys and "kb" in keys

    def test_counts_come_from_the_store(self, config, store):
        store.add(scope="note", title="a", body="one")
        store.add(scope="note", title="b", body="two")
        tiers = {t["key"]: t["count"] for t in browse.tiers(config)}
        assert tiers["note"] == 2 and tiers["session_summary"] == 0


class TestBrowse:
    def test_newest_first_with_previews(self, config, store):
        store.add(scope="note", title="old", body="first body")
        store.add(scope="note", title="new", body="second body")
        out = browse.browse(config, "note")
        assert [i["title"] for i in out["items"]] == ["new", "old"]
        assert out["items"][0]["preview"] == "second body"

    def test_a_query_searches_the_tier(self, config, store):
        store.add(scope="note", title="tabs", body="indentation policy")
        store.add(scope="note", title="deploys", body="never on fridays")
        out = browse.browse(config, "note", query="fridays")
        assert [i["title"] for i in out["items"]] == ["deploys"]

    def test_tiers_do_not_leak_into_each_other(self, config, store):
        store.add(scope="note", title="a note", body="x")
        store.add(scope="session_summary", title="a summary", body="y")
        assert [i["title"] for i in browse.browse(config, "note")["items"]] == ["a note"]

    def test_a_long_body_is_truncated_in_the_list(self, config, store):
        store.add(scope="note", title="long", body="x" * 500)
        preview = browse.browse(config, "note")["items"][0]["preview"]
        assert preview.endswith("…") and len(preview) < 300

    def test_an_unknown_tier_is_an_error_not_a_crash(self, config):
        out = browse.browse(config, "nope")
        assert out["items"] == [] and "unknown tier" in out["error"]


class TestItem:
    def test_the_full_body_and_meta_come_back(self, config, store):
        entry_id = store.add(scope="note", title="prefer tabs", body="y" * 400,
                             tags=["style"])
        d = browse.item(config, "note", entry_id)
        assert d["body"] == "y" * 400
        assert d["tags"] == ["style"]
        assert d["meta"]["scope"] == "note"

    def test_an_id_from_another_tier_is_not_found(self, config, store):
        entry_id = store.add(scope="session_summary", title="s", body="b")
        assert browse.item(config, "note", entry_id)["error"] == "not found"

    def test_a_missing_id_is_not_found(self, config):
        assert "error" in browse.item(config, "note", "nope")


class TestDescribedUnits:
    """The LLM-written descriptions of code (and of assembly) were reachable
    only through retrieval — never as something a person could read."""

    @pytest.fixture
    def units(self, config, tmp_path):
        from agent.rag.code_store import CodeStore
        db = tmp_path / ".agent" / "summaries.db"
        config.summarization.db_path = str(db)
        store = CodeStore(str(db))
        store.upsert_unit({
            "id": "u1", "path": "auth.py", "language": "python",
            "node_type": "function", "name": "login", "level": 0,
            "start_line": 10, "end_line": 40,
            "description": "checks the password and issues a session token",
            "status": "described", "analysis_model": "qwen"})
        store.upsert_unit({
            "id": "u2", "path": "auth.py", "language": "python",
            "node_type": "file", "name": "auth", "level": 1,
            "start_line": 1, "end_line": 200,
            "description": "authentication module rollup",
            "status": "described"})
        store.close()
        return config

    def test_no_database_is_a_note_not_a_crash(self, config, tmp_path):
        config.summarization.db_path = str(tmp_path / "nope.db")
        out = browse.browse(config, "unit")
        assert out["items"] == [] and "no unit database" in out["note"]

    def test_browsing_never_creates_the_database(self, config, tmp_path):
        db = tmp_path / "nope.db"
        config.summarization.db_path = str(db)
        browse.browse(config, "unit")
        assert not db.exists()

    def test_rollups_come_before_leaves(self, units):
        """The higher level says more per row, so it reads first."""
        items = browse.browse(units, "unit")["items"]
        assert [i["tags"][0] for i in items] == ["L1", "L0"]

    def test_a_unit_shows_where_it_lives(self, units):
        items = browse.browse(units, "unit")["items"]
        assert "auth.py:10" in [i["title"] for i in items][1]
        assert "login" in [i["title"] for i in items][1]

    def test_search_matches_the_description(self, units):
        items = browse.browse(units, "unit", query="password")["items"]
        assert len(items) == 1 and items[0]["id"] == "u1"

    def test_the_detail_carries_the_analysis_metadata(self, units):
        d = browse.item(units, "unit", "u1")
        assert "password" in d["body"]
        assert d["meta"]["lines"] == "10–40"
        assert d["meta"]["analysis_model"] == "qwen"

    def test_asm_units_share_the_reader(self, config, tmp_path):
        """Same columns, different DB — one pair of readers serves both."""
        import dataclasses
        from agent.rag.asm_store import AsmStore
        db = tmp_path / ".agent" / "index.db"
        config.rag.db_path = str(db)
        store = AsmStore(dataclasses.replace(config.rag, db_path=str(db)))
        store.upsert_unit({
            "id": "a1", "path": "boot.asm", "level": 0, "start_line": 1,
            "end_line": 20, "description": "sets up the stack",
            "checksum": "x", "status": "described"})
        items = browse.browse(config, "asm_unit")["items"]
        assert [i["id"] for i in items] == ["a1"]
        assert "stack" in browse.item(config, "asm_unit", "a1")["body"]


class TestKB:
    def test_kb_off_explains_itself_rather_than_erroring(self, config):
        out = browse.browse(config, "kb")
        assert out["items"] == [] and "kb.enabled" in out["note"]

    @pytest.fixture
    def kb_config(self, config, tmp_path):
        if not KB_FIXTURE.exists():
            pytest.skip("kb fixture corpus not present")
        root = tmp_path / "corpus"
        shutil.copytree(KB_FIXTURE, root)
        from kb.model import apply_schema
        from kb.store import rebuild
        conn = sqlite3.connect(str(root / "index.sqlite"))
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        rebuild(root, conn)
        conn.commit()
        conn.close()
        config.kb.enabled = True
        config.kb.corpus_path = str(root)
        return config

    def test_the_corpus_is_counted_and_listed(self, kb_config):
        tiers = {t["key"]: t["count"] for t in browse.tiers(kb_config)}
        assert tiers["kb"] > 0
        items = browse.browse(kb_config, "kb")["items"]
        assert items and all(i["id"] for i in items)

    def test_a_node_reads_back_with_its_description(self, kb_config):
        items = browse.browse(kb_config, "kb")["items"]
        d = browse.item(kb_config, "kb", items[0]["id"])
        assert d["title"] and "error" not in d
        assert "completeness" in d["meta"]

    def test_search_narrows_the_corpus(self, kb_config):
        all_items = browse.browse(kb_config, "kb")["items"]
        hits = browse.browse(kb_config, "kb", query="main")["items"]
        assert len(hits) <= len(all_items)
