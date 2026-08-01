"""Unit tests for the [ui.changeset] config section and its wiring.

Covers the P1 plumbing for core.changeset: the config section reaches
config.ui.changeset through the real loader, limits_from_config picks the
values up, capture_a persists a changeset alongside modified_files without
breaking old sessions, and from_a_data gives every UI the same fallback for
sessions that predate the feature.
"""
from __future__ import annotations

import asyncio

import agent.core.changeset as cs
from agent.config import Config, _merge_obj, load_config
from agent.config.models import ChangesetConfig
from agent.memory import session as session_mod
from agent.memory.qa_log import QALogger, read_history_sync


class TestChangesetConfigSection:
    def test_toml_section_reaches_config_ui_changeset(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "no_home")
        (tmp_path / "agent.toml").write_text(
            """
[ui.changeset]
enabled = false
inline_max_files = 7
fold_journal = "immediately"
""",
            encoding="utf-8",
        )
        c = load_config(tmp_path / "agent.toml")
        assert c.ui.changeset.enabled is False
        assert c.ui.changeset.inline_max_files == 7
        assert c.ui.changeset.fold_journal == "immediately"

    def test_unknown_key_in_section_is_ignored_not_fatal(self):
        c = Config()
        _merge_obj(c.ui.changeset, {"bogus_field": "value"})
        assert not hasattr(c.ui.changeset, "bogus_field")

    def test_limits_from_config_picks_up_configured_values(self):
        c = Config()
        c.ui.changeset.inline_max_files = 9
        c.ui.changeset.max_diff_bytes = 1234
        limits = cs.limits_from_config(c)
        assert limits.inline_max_files == 9
        assert limits.max_diff_bytes == 1234

    def test_int_defaults_match_changeset_limits(self):
        cfg_defaults = ChangesetConfig()
        limits_defaults = cs.Limits()
        for f in limits_defaults.__dataclass_fields__:
            assert getattr(cfg_defaults, f) == getattr(limits_defaults, f), f


class TestCaptureARoundTrip:
    def test_changeset_written_and_read_back(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        logger = QALogger("s1")
        payload = {"turn_id": 1, "tier": "list", "truncated": False,
                   "files": [{"path": "a.py", "added": 1, "removed": 0}]}

        async def go():
            await logger.capture_q(1, "hello")
            await logger.capture_a(1, "world", modified_files=["a.py"], changeset=payload)

        asyncio.run(go())
        _tid, _q, a = read_history_sync("s1")[0]
        assert a["changeset"] == payload
        assert a["modified_files"] == ["a.py"]

    def test_capture_a_without_changeset_still_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        logger = QALogger("s2")
        asyncio.run(logger.capture_a(1, "resp"))
        _tid, _q, a = read_history_sync("s2")[0]
        assert a["changeset"] == {}


class TestFromAData:
    def test_uses_stored_changeset_when_present(self):
        stored = {"turn_id": 3, "tier": "inline", "truncated": False,
                  "files": [{"path": "a.py", "added": 2, "removed": 1}]}
        c = cs.from_a_data({"changeset": stored, "modified_files": ["b.py"]})
        assert c.turn_id == 3
        assert [f.path for f in c.files] == ["a.py"]

    def test_falls_back_to_string_list(self):
        c = cs.from_a_data({"changeset": {}, "modified_files": ["a.py", "b.py"]})
        assert [f.path for f in c.files] == ["a.py", "b.py"]
        assert all(f.added == 0 and f.removed == 0 for f in c.files)

    def test_falls_back_to_dict_list_with_added_removed(self):
        modified = [{"path": "a.py", "added": 4, "removed": 2}]
        c = cs.from_a_data({"modified_files": modified})
        assert c.files[0].path == "a.py"
        assert (c.files[0].added, c.files[0].removed) == (4, 2)
        assert c.files[0].diff is None
        assert c.files[0].status == "modified"

    def test_duplicate_paths_are_deduped_keeping_first(self):
        modified = [{"path": "a.py", "added": 1}, {"path": "a.py", "added": 99}, "a.py"]
        c = cs.from_a_data({"modified_files": modified})
        assert len(c.files) == 1
        assert c.files[0].added == 1

    def test_the_a_record_supplies_the_turn_id_on_the_fallback_path(self):
        """An old session has no changeset, but the record still knows its turn."""
        c = cs.from_a_data({"turn_id": 7, "modified_files": ["a.py"]})
        assert c.turn_id == 7

    def test_an_untagged_stored_changeset_takes_the_turn_id_from_the_record(self):
        stored = {"tier": "list", "files": [{"path": "a.py"}]}
        assert cs.from_a_data({"turn_id": 7, "changeset": stored}).turn_id == 7

    def test_missing_changeset_and_modified_files_is_empty(self):
        assert cs.from_a_data({}).file_count == 0

    def test_garbage_input_does_not_raise(self):
        assert cs.from_a_data(None).file_count == 0
        assert cs.from_a_data({"modified_files": [None, 42, {}]}).file_count == 0
