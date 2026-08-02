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


class TestChunks:
    """The RAG index decides what the agent can find; being able to read a
    chunk is how you learn why a search missed."""

    @pytest.fixture
    def indexed(self, config, tmp_path):
        import dataclasses
        from agent.rag.store import VectorStore
        db = tmp_path / ".agent" / "index.db"
        config.rag.db_path = str(db)
        store = VectorStore(dataclasses.replace(config.rag, db_path=str(db)))
        store.upsert_many([
            {"id": "c1", "path": "auth.py", "language": "python",
             "node_type": "function_definition", "name": "login",
             "start_line": 10, "end_line": 40,
             "content": "def login(user, password):\n    return check(password)",
             "mtime": 1.0, "git_hash": "abc123"},
            {"id": "c2", "path": "util.py", "language": "python",
             "node_type": "function_definition", "name": "slugify",
             "start_line": 1, "end_line": 5,
             "content": "def slugify(text):\n    return text.lower()",
             "mtime": 1.0, "git_hash": ""},
        ])
        store.close()
        return config

    def test_no_index_is_a_note_not_a_crash(self, config, tmp_path):
        config.rag.db_path = str(tmp_path / "missing.db")
        out = browse.browse(config, "chunk")
        assert out["items"] == [] and "no code index" in out["note"]

    def test_chunks_are_counted_and_listed_by_position(self, indexed):
        tiers = {t["key"]: t["count"] for t in browse.tiers(indexed)}
        assert tiers["chunk"] == 2
        assert [i["source"] for i in browse.browse(indexed, "chunk")["items"]] \
            == ["auth.py", "util.py"]

    def test_search_uses_the_index_not_a_scan(self, indexed):
        items = browse.browse(indexed, "chunk", query="password")["items"]
        assert [i["id"] for i in items] == ["c1"]

    def test_a_query_fts_cannot_parse_still_answers(self, indexed):
        """Bare punctuation is invalid FTS syntax; a parser error is not an answer."""
        out = browse.browse(indexed, "chunk", query='def login(')
        assert "items" in out and "error" not in out

    def test_the_detail_shows_the_source_and_whether_it_is_embedded(self, indexed):
        d = browse.item(indexed, "chunk", "c1")
        assert "def login" in d["body"]
        assert d["meta"]["lines"] == "10–40"
        assert d["meta"]["embedded"] == "no"   # nothing embedded in this fixture


class TestArchive:
    """Pruned chunks are kept for a fortnight so a deletion can be undone —
    but only if someone can see what left the index, and why."""

    @pytest.fixture
    def archived(self, config, tmp_path):
        from agent.rag.archive import ArchiveStore
        db = tmp_path / ".agent" / "index-archive.db"
        config.rag.archive_db_path = str(db)
        store = ArchiveStore(str(db))
        store.ingest([
            {"id": "c1", "path": "old.py", "language": "python",
             "node_type": "function_definition", "name": "gone",
             "start_line": 1, "end_line": 9, "content": "def gone(): pass",
             "mtime": 1.0, "git_hash": ""},
        ], reason="missing")
        store.close()
        return config

    def test_an_empty_archive_explains_itself(self, config, tmp_path):
        config.rag.archive_db_path = str(tmp_path / "none.db")
        out = browse.browse(config, "archive")
        assert out["items"] == [] and "archived" in out["note"]

    def test_the_reason_for_removal_is_on_the_row(self, archived):
        item = browse.browse(archived, "archive")["items"][0]
        assert "missing" in item["tags"]

    def test_the_detail_says_when_and_why_it_left(self, archived):
        d = browse.item(archived, "archive", "c1")
        assert d["meta"]["reason"] == "missing"
        assert d["meta"]["archived_at"]
        assert "def gone" in d["body"]

    def test_search_reaches_archived_content(self, archived):
        assert [i["id"] for i in
                browse.browse(archived, "archive", query="gone")["items"]] == ["c1"]

    def test_the_archive_is_counted_separately_from_the_index(self, archived):
        tiers = {t["key"]: t["count"] for t in browse.tiers(archived)}
        assert tiers["archive"] == 1 and tiers["chunk"] == 0


class TestRules:
    """The files that enter the prompt every turn — the memory people are most
    often wrong about, because nothing showed which ones were picked up."""

    @pytest.fixture
    def ruled(self, config, tmp_path):
        (tmp_path / "AGENT.md").write_text("# project\nalways run the tests\n",
                                           encoding="utf-8")
        (tmp_path / ".agent.ignore").write_text("node_modules/\n", encoding="utf-8")
        ctx = tmp_path / ".agent" / "context" / "always"
        ctx.mkdir(parents=True)
        (ctx / "user").write_text("prefer small diffs\n", encoding="utf-8")
        return config

    def test_the_project_doc_and_rule_files_are_listed(self, ruled):
        titles = [i["title"] for i in browse.browse(ruled, "rule")["items"]]
        assert any("project doc" in t for t in titles)
        assert any(".agent.ignore" in t for t in titles)
        assert any("context/user" in t for t in titles)

    def test_each_row_says_which_layer_it_came_from(self, ruled):
        items = browse.browse(ruled, "rule")["items"]
        assert {"project", "agent dir"} & {t for i in items for t in i["tags"]}

    def test_a_file_named_for_another_tool_is_called_out(self, config, tmp_path):
        """AGENTS.md looks like project instructions and is never read."""
        (tmp_path / "AGENTS.md").write_text("# not loaded\n", encoding="utf-8")
        out = browse.browse(config, "rule")
        assert "AGENTS.md" in out["note"] and "not loaded" in out["note"]

    def test_the_content_reads_back(self, ruled):
        items = browse.browse(ruled, "rule")["items"]
        doc = [i for i in items if "project doc" in i["title"]][0]
        assert "always run the tests" in browse.item(ruled, "rule", doc["id"])["body"]

    def test_search_looks_inside_the_files(self, ruled):
        items = browse.browse(ruled, "rule", query="small diffs")["items"]
        assert [i["title"] for i in items] == ["context/user — agent dir"]

    def test_arbitrary_paths_are_not_readable(self, ruled):
        """Browsing rules must not become a way to read the host filesystem."""
        assert browse.item(ruled, "rule", "/etc/passwd") == {"error": "not found"}


class TestSkills:
    """The one tier the agent writes for itself and revises — worth reading
    to catch it learning the wrong lesson."""

    @pytest.fixture
    def skilled(self, config, tmp_path):
        from agent.skills import SkillLoader
        loader = SkillLoader(config)
        loader.save("release", "bump, tag, push", description="how to cut a release")
        loader.save("release", "bump, tag, push, announce",
                    description="how to cut a release")
        return config

    def test_project_and_bundled_skills_are_told_apart(self, skilled):
        items = browse.browse(skilled, "skill")["items"]
        by_name = {i["title"]: i["tags"] for i in items}
        assert "project" in by_name["release"]
        assert any("bundled" in tags for name, tags in by_name.items()
                   if name != "release")

    def test_the_version_rides_along(self, skilled):
        items = {i["title"]: i["tags"] for i in browse.browse(skilled, "skill")["items"]}
        assert "v2" in items["release"]

    def test_the_body_and_its_revisions_read_back(self, skilled):
        d = browse.item(skilled, "skill", "release")
        assert "announce" in d["body"]
        assert "revisions" in d["body"] and "v1" in d["body"]
        assert d["meta"]["origin"] == "project" and d["meta"]["versions"] == 2

    def test_search_looks_inside_the_skill(self, skilled):
        items = browse.browse(skilled, "skill", query="announce")["items"]
        assert [i["title"] for i in items] == ["release"]

    def test_a_missing_skill_is_not_found(self, config):
        assert browse.item(config, "skill", "nope") == {"error": "not found"}


class TestSessionFacts:
    """What compaction threw out of the context, and what it kept. The
    memory.db scope only fills when an embedder is configured; the JSON on
    disk is always written."""

    @pytest.fixture
    def compacted(self, config, tmp_path):
        from agent.memory import session as session_mod
        from agent.memory.facts_store import FactsStore
        session_mod.configure(str(tmp_path), ".agent")
        store = FactsStore("S1")
        store.new_round(from_turn=1, to_turn=8,
                        knowledge_draft="long draft about the parser rewrite",
                        summary="rewrote the parser", q_view="make it faster",
                        facts={"files_modified": ["parser.py"]})
        return config

    def test_without_a_session_it_says_why_it_is_empty(self, config):
        out = browse.browse(config, "session_facts")
        assert out["items"] == [] and "per session" in out["note"]

    def test_an_uncompacted_session_says_so(self, config, tmp_path):
        from agent.memory import session as session_mod
        session_mod.configure(str(tmp_path), ".agent")
        out = browse.browse(config, "session_facts", session_id="S404")
        assert out["items"] == [] and "not been compacted" in out["note"]

    def test_rounds_are_listed_newest_first_with_their_turn_range(self, compacted):
        items = browse.browse(compacted, "session_facts", session_id="S1")["items"]
        assert items[0]["title"].startswith("round 1 — turns 1–8")

    def test_the_detail_separates_what_survived_from_the_draft(self, compacted):
        d = browse.item(compacted, "session_facts", "1", session_id="S1")
        assert "what the model still sees" in d["body"]
        assert "long draft" in d["body"]
        assert "parser.py" in d["body"]
        assert d["meta"]["turns"] == "1–8"

    def test_the_count_follows_the_live_session(self, compacted):
        by_key = {t["key"]: t["count"]
                  for t in browse.tiers(compacted, session_id="S1")}
        assert by_key["session_facts"] == 1
        assert {t["key"]: t["count"]
                for t in browse.tiers(compacted)}["session_facts"] == 0

    def test_a_bad_round_id_is_not_found(self, compacted):
        assert "error" in browse.item(compacted, "session_facts", "nope",
                                      session_id="S1")


class TestKB:
    def test_kb_off_explains_itself_rather_than_erroring(self, config):
        out = browse.browse(config, "kb")
        assert out["items"] == [] and "kb.enabled" in out["note"]

    def test_a_path_that_is_not_a_corpus_says_how_to_make_one(self, config, tmp_path):
        """An empty directory is the normal first mistake after setting the path."""
        empty = tmp_path / "not-a-corpus"
        empty.mkdir()
        config.kb.enabled = True
        config.kb.corpus_path = str(empty)
        out = browse.browse(config, "kb")
        assert out["items"] == [] and "kb corpus init" in out["note"]

    def test_a_missing_kb_package_names_the_fix(self, config, tmp_path, monkeypatch):
        """kb/ imports as a namespace package with no submodules, so a missing
        install looks like a plain ImportError deep in the call."""
        import builtins
        root = tmp_path / "corpus"
        root.mkdir()
        (root / "corpus.yaml").write_text("name: x\n", encoding="utf-8")
        config.kb.enabled = True
        config.kb.corpus_path = str(root)
        real_import = builtins.__import__

        def _fail(name, *a, **kw):
            if name == "kb.api":
                raise ImportError("No module named 'kb.api'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fail)
        out = browse.browse(config, "kb")
        assert "not installed" in out["note"] and "pip install" in out["note"]

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


class TestUnindexedRoundsAreVisible:
    """Rounds written without an embedder are invisible to recall. The old
    behaviour was a bare `return` — for 293 sessions the tier read zero and
    nothing anywhere said why."""

    def test_the_skip_is_logged_once_with_the_fix(self, tmp_path, caplog):
        import logging
        from agent.memory import facts_store as fs
        from agent.memory import session as session_mod

        session_mod.configure(str(tmp_path), ".agent")
        fs._warned_no_index = False
        store = fs.FactsStore("S1")          # no embedder
        with caplog.at_level(logging.WARNING, logger="agent.memory.facts_store"):
            store.new_round(from_turn=1, to_turn=2, knowledge_draft="d",
                            summary="s")
            store.new_round(from_turn=3, to_turn=4, knowledge_draft="d2",
                            summary="s2")

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "one warning per process, not per round"
        assert "embed --start" in warnings[0].getMessage()

    def test_the_rounds_are_still_written(self, tmp_path):
        """The warning is about recall, not about losing the data."""
        from agent.memory import facts_store as fs
        from agent.memory import session as session_mod

        session_mod.configure(str(tmp_path), ".agent")
        fs._warned_no_index = False
        store = fs.FactsStore("S2")
        store.new_round(from_turn=1, to_turn=2, knowledge_draft="d", summary="s")
        assert store.list_round_ids() == [1]

    def test_the_overview_says_so_where_someone_will_see_it(self, tmp_path):
        from agent.config.models import Config
        from agent.memory import facts_store as fs
        from agent.memory import session as session_mod
        from agent.memory.overview import overview

        cfg = Config()
        cfg.tools.working_dir = str(tmp_path)
        cfg.tools.agent_dir = ".agent"
        session_mod.configure(str(tmp_path), ".agent")
        fs._warned_no_index = False
        fs.FactsStore("S3").new_round(from_turn=1, to_turn=2,
                                      knowledge_draft="d", summary="s")

        warnings = overview(cfg, session_id="S3")["warnings"]
        assert any("not indexed for recall" in w for w in warnings)
