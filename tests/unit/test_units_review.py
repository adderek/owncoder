"""Reviewer corrections: they win over the model and survive re-analysis."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from agent.config.models import Config
from agent.memory import units_explorer as ux
from agent.memory import units_review as ur


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.summarization.db_path = str(tmp_path / ".agent" / "summaries.db")
    cfg.rag.db_path = str(tmp_path / ".agent" / "index.db")
    return cfg


def _code_unit(**kw):
    return {"id": "a", "path": "auth.py", "language": "python", "node_type": "function",
            "name": "login", "level": 0, "start_line": 1, "end_line": 2,
            "description": "model says: logs in", "inferred_name": "login",
            "object_checksum": "cs1", "status": "described", **kw}


@pytest.fixture
def code(config, tmp_path):
    from agent.rag.code_store import CodeStore
    (tmp_path / "auth.py").write_text("def login():\n    return check()\n")
    store = CodeStore(config.summarization.db_path)
    store.upsert_unit(_code_unit())
    store.close()
    return config


def _store(config):
    from agent.rag.code_store import CodeStore
    return CodeStore(config.summarization.db_path)


def test_an_edit_replaces_the_text_everyone_reads(code):
    ur.edit(code, "unit", "a", name="authenticate", description="checks the password")
    n = ux.node(code, "unit", "a")
    assert n["unit"]["description"] == "checks the password"
    assert n["unit"]["inferred_name"] == "authenticate"
    assert n["model"]["description"] == "model says: logs in"
    assert n["review"]["stale"] is False


def test_the_edit_survives_the_indexer_dropping_and_rewriting_the_file(code):
    """The code indexer deletes a file's units and re-inserts them."""
    ur.edit(code, "unit", "a", name=None, description="checks the password")
    store = _store(code)
    store.delete_units_for_file("auth.py")
    # Same code moved down a line: new id, same content checksum.
    store.upsert_unit(_code_unit(id="a2", start_line=2, end_line=3,
                                 description="model says: something new"))
    store.close()
    n = ux.node(code, "unit", "a2")
    assert n["unit"]["description"] == "checks the password"
    assert n["model"]["description"] == "model says: something new"


def test_a_pipeline_rewriting_a_dict_it_read_back_keeps_the_model_text(code):
    ur.edit(code, "unit", "a", name=None, description="human text")
    store = _store(code)
    row = store._conn.execute("SELECT * FROM units WHERE id='a'").fetchone()
    store.upsert_unit({**dict(row), "status": "described"})   # e.g. sibling relink
    store.close()
    n = ux.node(code, "unit", "a")
    assert n["unit"]["description"] == "human text"
    assert n["model"]["description"] == "model says: logs in"


def test_changed_code_flags_the_override_stale(code):
    ur.edit(code, "unit", "a", name=None, description="human text")
    store = _store(code)
    store.upsert_unit(_code_unit(object_checksum="cs2"))
    store.close()
    assert ux.node(code, "unit", "a")["review"]["stale"] is True


def test_clearing_the_edit_returns_to_the_model(code):
    ur.edit(code, "unit", "a", name="x", description="human text")
    ur.edit(code, "unit", "a", name="", description="")
    n = ux.node(code, "unit", "a")
    assert n["unit"]["description"] == "model says: logs in" and n["review"] is None


def test_the_tree_marks_corrected_and_rejected_units(code):
    ur.edit(code, "unit", "a", name=None, description="human text")
    ur.review(code, "unit", "a", "wrong", "it also locks the account")
    row = ux.units(code, "unit", path="auth.py")["items"][0]
    assert row["human"] and row["review"] == "wrong"
    assert [i["id"] for i in ux.units(code, "unit", review="wrong")["items"]] == ["a"]


class _FakeLLM:
    def __init__(self, reply: dict):
        self.prompts: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._reply = reply

    def _create(self, **kw):
        self.prompts.append(kw["messages"][-1]["content"])
        msg = SimpleNamespace(content=json.dumps(self._reply), reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def test_redescribe_passes_the_reviewers_note_and_clears_the_verdict(code, monkeypatch):
    fake = _FakeLLM({"description": "verifies credentials and locks after 3 tries",
                     "inferred_name": "login_with_lockout", "confidence": "high"})
    monkeypatch.setattr("openai.OpenAI", lambda **_kw: fake)
    ur.review(code, "unit", "a", "wrong", "it also locks the account")
    out = ur.redescribe(code, "unit", "a")
    assert out["description"].startswith("verifies credentials")
    assert "it also locks the account" in fake.prompts[0]
    assert "model says: logs in" in fake.prompts[0]
    n = ux.node(code, "unit", "a")
    assert n["unit"]["description"].startswith("verifies") and n["review"] is None
    assert n["meta"]["analysis_model"] == code.llm.model


def test_redescribe_under_a_human_override_only_refreshes_the_model_copy(code, monkeypatch):
    fake = _FakeLLM({"description": "new model text", "inferred_name": "x",
                     "confidence": "low"})
    monkeypatch.setattr("openai.OpenAI", lambda **_kw: fake)
    ur.edit(code, "unit", "a", name=None, description="human text")
    ur.redescribe(code, "unit", "a")
    n = ux.node(code, "unit", "a")
    assert n["unit"]["description"] == "human text"
    assert n["model"]["description"] == "new model text"


def test_asm_redescribe_gives_the_model_the_callees_from_the_code(config, tmp_path, monkeypatch):
    from agent.rag.asm_store import AsmStore
    (tmp_path / "rom.asm").write_text("0000: LCALL 0010H\n0003: RET\n0010: MOV A, #1\n0012: RET\n")
    store = AsmStore(dataclasses.replace(config.rag, db_path=config.rag.db_path))
    base = {"path": "rom.asm", "checksum": "x", "status": "described", "level": 0}
    store.upsert_unit({**base, "id": "boot", "start_line": 1, "end_line": 2,
                       "inferred_name": "boot", "description": "old"})
    store.upsert_unit({**base, "id": "init", "start_line": 3, "end_line": 4,
                       "inferred_name": "set_accumulator", "description": "sets A to 1"})
    store.close()
    fake = _FakeLLM({"purpose": "calls set_accumulator", "inferred_name": "boot",
                     "confidence": "medium"})
    monkeypatch.setattr("openai.OpenAI", lambda **_kw: fake)
    ur.redescribe(config, "asm_unit", "boot")
    assert "- set_accumulator: sets A to 1" in fake.prompts[0]


def test_review_rejects_an_unknown_verdict(code):
    with pytest.raises(ur.ReviewError):
        ur.review(code, "unit", "a", "maybe")


def test_an_unchanged_file_reindexed_keeps_both_texts_apart(code, tmp_path):
    """The indexer's same-path skip re-inserts what it read back; the reviewer's
    text must not turn into the model's copy on the way."""
    import hashlib
    from agent.rag.indexer import _enqueue_for_summarization
    content = "def login():\n    return check()\n"
    cs = hashlib.sha256(content.encode()).hexdigest()[:16]
    store = _store(code)
    store.upsert_unit(_code_unit(object_checksum=cs))
    ur.edit(code, "unit", "a", name="authenticate", description="human text")
    chunk = {"id": "a", "content": content, "language": "python", "node_type": "function",
             "name": "login", "start_line": 1, "end_line": 2}
    _enqueue_for_summarization(store, [chunk], "auth.py", 0.0, None, force=False,
                               abs_path=str(tmp_path / "auth.py"))
    store.close()
    n = ux.node(code, "unit", "a")
    assert (n["unit"]["inferred_name"], n["unit"]["description"]) == ("authenticate", "human text")
    assert n["model"] == {"description": "model says: logs in", "inferred_name": "login"}
