"""The user-config ceiling on runtime path grants.

A project config may not widen its own access, and neither may anything the
agent can reach at runtime: every grant minted after startup — UI, /paths,
an agent request, a stored grants file, a resumed session — is confined to the
`[[security.grant_ceiling]]` entries of the *user* config layer.
"""
from __future__ import annotations

import json

import pytest

from agent.config.models import Config, ToolsConfig
from agent.security import path_grants as pg
from agent.config import loader


@pytest.fixture
def grant_env(tmp_path):
    """Factory: set up path_grants with a given ceiling, reset globals after."""
    root = tmp_path / "project"
    root.mkdir()

    def _setup(ceiling):
        cfg = Config(tools=ToolsConfig(working_dir=str(root)))
        cfg.tools.agent_dir = str(root / ".agent")
        cfg.security.grant_ceiling = ceiling
        pg.setup(cfg)
        return cfg

    yield _setup
    pg._grants = []
    pg._ceiling = []
    pg._grants_file = None


class TestCeilingConfinement:
    def test_no_ceiling_leaves_grants_unrestricted(self, grant_env, tmp_path):
        """Empty (the default) means no ceiling — today's behaviour."""
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        grant_env([])
        pg.add_grant(outside, "rw")
        assert pg.grant_for(outside) is not None

    def test_ro_ceiling_allows_a_read_only_grant(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        pg.add_grant(ext / "a.txt", "ro")
        assert pg.grant_for(ext / "a.txt") is not None

    def test_rw_ceiling_allows_a_read_write_grant(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "rw"}])
        pg.add_grant(ext, "rw")
        assert pg.grant_for(ext).mode == "rw"

    def test_ro_ceiling_cannot_be_escalated_to_rw(self, grant_env, tmp_path):
        """The user's central invariant: pre-approved ro never becomes rw."""
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        with pytest.raises(pg.CeilingError, match="read-only"):
            pg.add_grant(ext, "rw")

    def test_path_outside_the_ceiling_is_refused(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "rw"}])
        with pytest.raises(pg.CeilingError, match="not under any path"):
            pg.add_grant(tmp_path / "somewhere-else", "ro")

    def test_grant_may_not_cover_more_than_the_ceiling(self, grant_env, tmp_path):
        """A parent of a ceiling entry is still outside it."""
        ext = tmp_path / "ext"
        (ext / "sub").mkdir(parents=True)
        grant_env([{"path": str(ext / "sub"), "mode": "rw"}])
        with pytest.raises(pg.CeilingError, match="not under any path"):
            pg.add_grant(ext, "rw")

    def test_prefix_sibling_is_not_inside(self, grant_env, tmp_path):
        (tmp_path / "ext").mkdir()
        (tmp_path / "extended").mkdir()
        grant_env([{"path": str(tmp_path / "ext"), "mode": "rw"}])
        with pytest.raises(pg.CeilingError, match="not under any path"):
            pg.add_grant(tmp_path / "extended", "rw")

    def test_bad_ceiling_entries_are_ignored(self, grant_env, tmp_path):
        grant_env([{"path": "", "mode": "rw"},
                   {"path": str(tmp_path), "mode": "sideways"},
                   "not-a-table"])
        assert pg.ceiling() == []

    def test_ceiling_is_reported_for_the_ui(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        assert pg.ceiling() == [(str(ext), "ro")]

    def test_the_project_root_default_grant_survives(self, grant_env, tmp_path):
        """The workdir grant is seeded, not negotiated — a ceiling listing some
        unrelated path must not lock the agent out of its own project."""
        ext = tmp_path / "ext"
        ext.mkdir()
        cfg = grant_env([{"path": str(ext), "mode": "ro"}])
        assert pg.grant_for((tmp_path / "project").resolve()) is not None


class TestCeilingOnStoredState:
    def test_agent_request_outside_the_ceiling_is_refused(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        with pytest.raises(pg.CeilingError):
            pg.request_grant(tmp_path / "elsewhere", "ro", "because")
        assert pg.get_all() and all(g.state != "pending" for g in pg.get_all())

    def test_accept_drops_a_request_that_now_exceeds_the_ceiling(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "rw"}])
        pg.request_grant(ext, "rw", "because")
        # User tightens the config while the request sits in the panel.
        pg._load_ceiling([{"path": str(ext), "mode": "ro"}])
        assert pg.accept_grant(ext) is False
        assert pg.grant_for(ext) is None
        assert not pg.has_pending()

    def test_grants_file_record_outside_the_ceiling_is_dropped(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        cfg = grant_env([{"path": str(ext), "mode": "rw"}])
        (tmp_path / "project" / ".agent" / "path_grants.json").write_text(
            json.dumps([{"path": str(outside), "mode": "rw", "origin": "user"}]))
        pg.setup(cfg)
        assert pg.grant_for(outside) is None

    def test_session_record_outside_the_ceiling_is_dropped(self, grant_env, tmp_path):
        """A session snapshot is data: it must not widen a tightened ceiling."""
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        pg.apply_session([{"path": str(tmp_path / "elsewhere"), "mode": "rw",
                           "origin": "user"}])
        assert pg.grant_for(tmp_path / "elsewhere") is None

    def test_session_record_above_the_ceiling_mode_is_dropped(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        pg.apply_session([{"path": str(ext), "mode": "rw", "origin": "user"}])
        assert pg.grant_for(ext) is None

    def test_session_record_inside_the_ceiling_is_applied(self, grant_env, tmp_path):
        ext = tmp_path / "ext"
        ext.mkdir()
        grant_env([{"path": str(ext), "mode": "ro"}])
        pg.apply_session([{"path": str(ext), "mode": "ro", "origin": "user"}])
        assert pg.grant_for(ext) is not None


class TestProjectLayerCannotSetTheCeiling:
    def test_grant_ceiling_is_stripped_from_a_project_config(self):
        cfg = Config()
        data = {"security": {"grant_ceiling": [{"path": "/", "mode": "rw"}]}}
        issues = loader._clamp_project_security(cfg, data)
        assert "grant_ceiling" not in data["security"]
        assert any("grant_ceiling" in i for i in issues)

    def test_a_project_may_still_tighten(self):
        cfg = Config()
        data = {"security": {"require_sandbox": True}}
        assert loader._clamp_project_security(cfg, data) == []
        assert data["security"]["require_sandbox"] is True
