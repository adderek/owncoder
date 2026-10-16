"""Opt-in per-file roll-ups for code units, and the worker that fills them."""
from __future__ import annotations

from agent.memory import units_explorer as ux
from agent.config.models import Config
from agent.rag.code_store import CodeStore
from agent.rag.indexer import _enqueue_for_summarization


def _chunks():
    return [{"id": f"c{i}", "content": f"def f{i}(): pass\n", "language": "python",
             "node_type": "function_definition", "name": f"f{i}",
             "start_line": i, "end_line": i} for i in (1, 2, 3)]


def test_file_rollup_is_off_by_default(tmp_path):
    store = CodeStore(str(tmp_path / "s.db"))
    _enqueue_for_summarization(store, _chunks(), "m.py", 0.0, None, force=False)
    assert {u["level"] for u in store.get_units_for_file("m.py")} == {0}


def test_file_rollup_parents_the_top_level_units(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.summarization.db_path = str(tmp_path / "s.db")
    store = CodeStore(cfg.summarization.db_path, file_rollups=True)
    _enqueue_for_summarization(store, _chunks(), "m.py", 0.0, None, force=False)
    roots = ux.units(cfg, "unit", path="m.py")["items"]
    assert [(r["kind"], r["level"], r["status"]) for r in roots] == [("file", 1, "stale")]
    assert [k["id"] for k in ux.units(cfg, "unit", parent=roots[0]["id"])["items"]] == [
        "c1", "c2", "c3"]
    assert [c["id"] for c in store.get_children(roots[0]["id"])] == ["c1", "c2", "c3"]


def test_a_waiting_rollup_is_not_reported_as_work(tmp_path):
    """Otherwise the worker loop never sleeps while parts are undescribed."""
    from agent.rag.bg_worker import BgWorker
    store = CodeStore(str(tmp_path / "s.db"), file_rollups=True)
    _enqueue_for_summarization(store, _chunks(), "m.py", 0.0, None, force=False)
    store._conn.execute("UPDATE units SET status = 'failed' WHERE level = 0")
    store._conn.commit()
    assert BgWorker(store, describer=None, judge=None)._tick() is False
