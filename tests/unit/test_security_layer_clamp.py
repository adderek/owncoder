"""Project-layer [security] may only tighten — docs/hooks-trust-boundary.md item 6.

Same threat as the hook trust boundary: <project_root>/agent.toml ships with a
clone, so without this a hostile repo disarms the sandbox, the write-deny globs
and output redaction before the first tool call.
"""
import pytest

from agent.config.loader import _security_narrows, load_config


@pytest.fixture()
def project_cfg(tmp_path, monkeypatch):
    """Write a project agent.toml and load it as the project layer."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def _load(body: str):
        p = tmp_path / "agent.toml"
        p.write_text(body, encoding="utf-8")
        return load_config(p)

    return _load


# ── the attacks this closes ───────────────────────────────────────────────────

class TestHostileRepoCannotWeaken:
    def test_cannot_disable_the_sandbox(self, project_cfg):
        cfg = project_cfg("[security]\nrequire_sandbox = false\n")
        assert cfg.security.require_sandbox is True

    def test_cannot_empty_the_write_deny_globs(self, project_cfg):
        cfg = project_cfg("[security]\nwrite_deny_globs = []\n")
        assert cfg.security.write_deny_globs != []

    def test_cannot_disable_output_redaction(self, project_cfg):
        cfg = project_cfg("[security]\nredact_tool_output = false\n")
        assert cfg.security.redact_tool_output is True

    def test_cannot_turn_the_network_on(self, project_cfg):
        cfg = project_cfg('[security]\nnetwork = "on"\n')
        assert cfg.security.network == "off"

    def test_cannot_enable_symlink_following(self, project_cfg):
        cfg = project_cfg("[security]\nfollow_symlinks = true\n")
        assert cfg.security.follow_symlinks is False

    def test_cannot_enable_legacy_shell(self, project_cfg):
        cfg = project_cfg("[security]\nallow_legacy_shell = true\n")
        assert cfg.security.allow_legacy_shell is False

    def test_cannot_raise_resource_limits(self, project_cfg):
        cfg = project_cfg("[security]\nwall_seconds = 99999\nrss_mb = 999999\n")
        assert cfg.security.wall_seconds != 99999
        assert cfg.security.rss_mb != 999999

    def test_cannot_widen_the_env_allowlist(self, project_cfg):
        cfg = project_cfg('[security]\nenv_allow = ["PATH", "AWS_SECRET_ACCESS_KEY"]\n')
        assert "AWS_SECRET_ACCESS_KEY" not in cfg.security.env_allow

    def test_cannot_shrink_the_env_denylist(self, project_cfg):
        cfg = project_cfg('[security]\nenv_deny_patterns = ["^NOTHING$"]\n')
        assert len(cfg.security.env_deny_patterns) > 1

    def test_cannot_add_to_argv_allow(self, project_cfg):
        cfg = project_cfg('[security]\nargv_allow = ["curl"]\n')
        assert cfg.security.argv_allow == []

    def test_unclassified_setting_is_ignored(self, project_cfg):
        """No defined stricter direction means no project override at all."""
        cfg = project_cfg('[security]\nsandbox_backend = "none"\n')
        assert cfg.security.sandbox_backend != "none"

    def test_rejection_is_reported(self, project_cfg, capsys):
        project_cfg("[security]\nrequire_sandbox = false\n")
        err = capsys.readouterr().err
        assert "require_sandbox" in err and "may only tighten" in err

    def test_strict_mode_aborts_on_a_weakening_repo_config(self, project_cfg, monkeypatch):
        monkeypatch.setenv("AGENT_CONFIG_STRICT", "1")
        with pytest.raises(SystemExit):
            project_cfg("[security]\nrequire_sandbox = false\n")


# ── what a project layer keeps the power to do ────────────────────────────────

class TestProjectMayTighten:
    def test_may_enable_airgap(self, project_cfg):
        cfg = project_cfg("[security]\nairgap = true\n")
        assert cfg.security.airgap is True

    def test_may_lower_resource_limits(self, project_cfg):
        cfg = project_cfg("[security]\nwall_seconds = 5\n")
        assert cfg.security.wall_seconds == 5

    def test_may_add_write_deny_globs(self, project_cfg):
        from agent.security import fs as _fs
        extra = list(_fs._DEFAULT_WRITE_DENY_GLOBS) + ["secrets/**"]
        body = "[security]\nwrite_deny_globs = [" + ", ".join(
            f'"{g}"' for g in extra) + "]\n"
        cfg = project_cfg(body)
        assert "secrets/**" in cfg.security.write_deny_globs

    def test_may_shrink_the_env_allowlist(self, project_cfg):
        cfg = project_cfg('[security]\nenv_allow = ["PATH"]\n')
        assert cfg.security.env_allow == ["PATH"]

    def test_env_override_still_wins(self, project_cfg, monkeypatch):
        """The clamp runs before the merge, so the user's env keeps priority.

        The env var is the user's own decision, not the repo's — a clamp that
        replayed layers afterwards would silently discard it.
        """
        monkeypatch.setenv("AGENT_SECURITY_REQUIRE_SANDBOX", "0")
        cfg = project_cfg("[security]\nairgap = true\n")
        assert cfg.security.require_sandbox is False
        assert cfg.security.airgap is True

    def test_user_layer_may_still_loosen(self, tmp_path, monkeypatch):
        """Only *project* layers are clamped; the user's own config is trusted."""
        home = tmp_path / "home"
        (home / ".config" / "agent").mkdir(parents=True)
        (home / ".config" / "agent" / "agent.toml").write_text(
            "[security]\nrequire_sandbox = false\n", encoding="utf-8")
        monkeypatch.setenv("HOME", str(home))
        assert load_config().security.require_sandbox is False

    def test_no_security_section_leaves_defaults(self, project_cfg):
        cfg = project_cfg('[agent]\nmodel_mode = "any"\n')
        assert cfg.security.require_sandbox is True
        assert cfg.security.network == "off"


# ── the direction predicate ───────────────────────────────────────────────────

class TestNarrowsPredicate:
    @pytest.mark.parametrize("field,current,value,expected", [
        ("require_sandbox", True, False, False),
        ("require_sandbox", False, True, True),
        ("require_sandbox", True, True, True),
        ("follow_symlinks", False, True, False),
        ("follow_symlinks", True, False, True),
        ("wall_seconds", 30, 10, True),
        ("wall_seconds", 30, 60, False),
        ("wall_seconds", 30, "abc", False),
        ("network", "off", "on", False),
        ("network", "on", "off", True),
        ("env_allow", ["PATH", "HOME"], ["PATH"], True),
        ("env_allow", ["PATH"], ["PATH", "AWS_KEY"], False),
        ("env_deny_patterns", ["a"], ["a", "b"], True),
        ("env_deny_patterns", ["a", "b"], ["a"], False),
    ])
    def test_direction(self, field, current, value, expected):
        assert _security_narrows(field, current, value) is expected

    def test_unknown_field_never_narrows(self):
        assert _security_narrows("sandbox_backend", "auto", "none") is False

    def test_deny_globs_compare_against_resolved_defaults(self):
        """None means "built-in defaults", so [] must not read as a change."""
        assert _security_narrows("write_deny_globs", None, []) is False
