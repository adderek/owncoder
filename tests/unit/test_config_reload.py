"""/models reload — re-reads [models] only, and never on a broken file."""
import os
from pathlib import Path

import pytest

from agent.config import load_config
from agent.config.reload import reload_models


def _write(p: Path, body: str) -> Path:
    p.write_text(body, encoding="utf-8")
    return p


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config whose only layer is a project file we can rewrite."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    proj = _write(tmp_path / "agent.toml", """
[models.alpha]
base_url = "http://localhost:9001/v1"
model = "alpha-v1"

[models.beta]
base_url = "http://localhost:9002/v1"
model = "beta-v1"

[models]
default = "alpha"
""")
    c = load_config(proj)
    assert c.model_entries["alpha"].model == "alpha-v1"
    return c, proj


def test_picks_up_edited_entry(cfg):
    c, proj = cfg
    _write(proj, """
[models.alpha]
base_url = "http://localhost:9001/v1"
model = "alpha-v2"

[models]
default = "alpha"
""")
    ok, msg = reload_models(c, include_project=True)
    assert ok, msg
    assert c.model_entries["alpha"].model == "alpha-v2"
    assert "beta" not in c.model_entries          # removed entries go away
    assert c.llm.model == "alpha-v2"              # active entry re-bridged


def test_project_layers_skipped_by_default(cfg):
    """The common path must not re-read repo-shipped config: it is writable by
    anything with access to the tree, including the agent's own file tools."""
    c, proj = cfg
    _write(proj, '[models.alpha]\nmodel = "alpha-evil"\n')
    ok, msg = reload_models(c)
    assert not ok
    assert "project" in msg
    assert c.model_entries["alpha"].model == "alpha-v1"


def test_broken_file_leaves_config_untouched(cfg):
    c, proj = cfg
    _write(proj, "[models.alpha\nmodel = broken")
    ok, msg = reload_models(c, include_project=True)
    assert not ok
    assert "Config error" in msg
    assert c.model_entries["alpha"].model == "alpha-v1"
    assert "beta" in c.model_entries


def test_session_role_pin_survives(cfg):
    """`/model role=x` writes the same dict the files do — a reload must not
    silently undo the user's live choice."""
    c, proj = cfg
    c.model_roles["summarizer"] = "beta"
    ok, _ = reload_models(c, include_project=True)
    assert ok
    assert c.model_roles["summarizer"] == "beta"


def test_missing_active_entry_keeps_connection(cfg):
    c, proj = cfg
    before = c.llm.base_url
    _write(proj, '[models.gamma]\nbase_url = "http://localhost:9003/v1"\nmodel = "gamma-v1"\n')
    ok, msg = reload_models(c, include_project=True)
    assert ok, msg
    assert c.llm.base_url == before
    assert "keeping the current connection" in msg


def test_reload_does_not_reset_rate_limit_cooldowns(cfg):
    """A repeatable command must not be a way to clear a 429 backoff."""
    from agent.config import model_probe
    c, _ = cfg
    model_probe.mark_rate_limited("http://localhost:9001/v1", "alpha-v1", 300.0)
    ok, _ = reload_models(c, include_project=True)
    assert ok
    assert model_probe.is_rate_limited("http://localhost:9001/v1", "alpha-v1")


def test_only_models_section_is_reloaded(cfg):
    """Permissions/hooks/security stay at their startup values — re-merging
    them mid-session is the thing this command deliberately does not do."""
    c, proj = cfg
    before = c.tools.working_dir
    _write(proj, """
[tools]
working_dir = "/tmp/hijacked"

[models.alpha]
model = "alpha-v2"

[models]
default = "alpha"
""")
    ok, _ = reload_models(c, include_project=True)
    assert ok
    assert c.tools.working_dir == before
    assert c.model_entries["alpha"].model == "alpha-v2"


def test_no_recorded_layers_is_reported(cfg):
    c, _ = cfg
    c.loaded_config_layers = []
    ok, msg = reload_models(c, include_project=True)
    assert not ok
    assert "Restart" in msg


def test_agent_client_recreated_on_reload(cfg):
    from agent.core.agent import Agent
    from agent.ui.slash import handle_models_reload
    c, proj = cfg
    agent = Agent(c)
    assert "9001" in str(agent._client.base_url)

    _write(proj, """
[models.alpha]
base_url = "http://localhost:9099/v1"
model = "alpha-v1"
[models]
default = "alpha"
""")
    ok, msg = handle_models_reload(c, "project", agent)
    assert ok, msg
    assert "9099" in str(agent._client.base_url)


def test_default_role_change_applied(cfg):
    c, proj = cfg
    assert c.model_roles.get("default") == "alpha"
    _write(proj, """
[models.alpha]
base_url = "http://localhost:9001/v1"
model = "alpha-v1"
[models.beta]
base_url = "http://localhost:9002/v1"
model = "beta-v1"
[models]
default = "beta"
""")
    ok, msg = reload_models(c, include_project=True)
    assert ok, msg
    assert c.model_roles.get("default") == "beta"
    assert c.llm.model == "beta-v1"


def test_ensure_model_registry_keys_not_stale(cfg):
    c, proj = cfg
    _write(proj, """
[models.alpha]
base_url = "http://localhost:9999/v1"
model = "alpha-v2"
[models]
default = "alpha"
""")
    ok, msg = reload_models(c, include_project=True)
    assert ok, msg
    assert c.model_entries["default"].base_url == "http://localhost:9999/v1"


def test_embeddings_reloaded(cfg):
    c, proj = cfg
    _write(proj, """
[models.alpha]
base_url = "http://localhost:9001/v1"
model = "alpha-v1"
[models.emb]
base_url = "http://localhost:8888/v1"
model = "emb-v2"
dimensions = 768
[models]
default = "alpha"
embeddings = "emb"
""")
    ok, msg = reload_models(c, include_project=True)
    assert ok, msg
    assert c.embeddings.base_url == "http://localhost:8888/v1"
    assert c.embeddings.model == "emb-v2"
    assert c.embeddings.dimensions == 768


def test_project_models_preserved_when_project_skipped(tmp_path):
    user_file = _write(tmp_path / "user.toml", """
[models.user_m]
base_url = "http://localhost:8000/v1"
model = "user-v1"
""")
    proj_file = _write(tmp_path / "proj.toml", """
[models.proj_m]
base_url = "http://localhost:9000/v1"
model = "proj-v1"
""")
    c = load_config(proj_file)
    c.loaded_config_layers = [(str(user_file), False), (str(proj_file), True)]
    assert "proj_m" in c.project_model_entries

    ok, msg = reload_models(c, include_project=False)
    assert ok, msg
    assert "user_m" in c.model_entries
    assert "proj_m" in c.model_entries

