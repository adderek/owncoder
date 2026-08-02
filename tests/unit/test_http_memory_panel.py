"""The browser can see what the agent remembers.

Notes, session summaries, skills and the code index all existed, each behind a
different tool or CLI flag. The HTML UI showed none of it: no way to answer
"what is in my memory, and is the index current?" without leaving the page.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config.models import Config
from agent.memory.store import MemoryStore
from agent.ui.http_loop import _HttpUI, _PAGE

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Server:
    def __init__(self, agent=None):
        self._agent = agent


def _ui(config, session=None):
    ui = _HttpUI.__new__(_HttpUI)
    ui.server = _Server(agent=SimpleNamespace(config=config) if config else None)
    ui.session = session
    return ui


@pytest.fixture
def config(tmp_path):
    cfg = Config()
    cfg.tools.working_dir = str(tmp_path)
    cfg.tools.agent_dir = ".agent"
    cfg.rag.db_path = str(tmp_path / ".agent" / "index.db")
    # Session listing reads a process-global dir; point it at the tmp project
    # so counts here are this test's, not whatever ran before it.
    from agent.memory import session as session_mod
    session_mod.configure(str(tmp_path), ".agent")
    return cfg


def _tier(out, name):
    for t in out["tiers"]:
        if t["name"] == name:
            return t
    return None


class TestEndpoint:
    def test_an_empty_project_is_not_an_error(self, config):
        out = _ui(config).memory_info()
        assert out["tiers"] == [] or _tier(out, "notes") is None
        assert out["notes"] == []

    def test_notes_are_counted_and_the_recent_ones_listed(self, config, tmp_path):
        store = MemoryStore(tmp_path / ".agent" / "memory.db")
        store.add(scope="note", title="prefer tabs", body="the user said so",
                  tags=["style"])
        store.add(scope="note", title="deploy on fridays", body="never")
        store.add(scope="session_summary", title="s1", body="did things")

        out = _ui(config).memory_info()

        assert _tier(out, "notes")["count"] == 2
        assert _tier(out, "session summaries")["count"] == 1
        titles = [n["title"] for n in out["notes"]]
        assert "prefer tabs" in titles and "deploy on fridays" in titles
        assert ["style"] in [n["tags"] for n in out["notes"]]

    def test_skills_show_up_with_their_descriptions(self, config, tmp_path):
        skills = tmp_path / ".agent" / "skills"
        skills.mkdir(parents=True)
        (skills / "release.md").write_text(
            "---\ndescription: how to cut a release\n---\n\nbump, tag, push.\n",
            encoding="utf-8")

        out = _ui(config).memory_info()

        # Built-in skills count too, so the project one just has to be there.
        assert _tier(out, "skills")["count"] >= 1
        named = {s["name"]: s["description"] for s in out["skills"]}
        assert "cut a release" in named["release"]

    def test_an_off_the_record_session_says_so(self, config, monkeypatch):
        # Mode is process-wide truth (agent/security/vault.py), not a session field.
        from agent.security import vault
        monkeypatch.setattr(vault, "mode", lambda: "incognito")
        ui = _ui(config, session=SimpleNamespace(id="S1", mode="incognito"))
        out = ui.memory_info()
        assert any("incognito" in w for w in out["warnings"])

    def test_a_remote_backend_degrades_instead_of_crashing(self):
        out = _ui(None).memory_info()
        assert out["tiers"] == [] and out["warnings"]

    def test_a_broken_store_does_not_take_the_panel_down(self, config, tmp_path):
        """One unreadable store must not cost the user every other tier."""
        db = tmp_path / ".agent" / "memory.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_bytes(b"this is not a database")

        out = _ui(config).memory_info()

        assert _tier(out, "sessions") is not None


class TestSharedCommand:
    """Same overview text for every UI — the browser and the console cannot drift."""

    def test_a_fresh_project_renders_without_stored_memory(self, config):
        from agent.memory.overview import run_memory_command
        out = run_memory_command(config)
        assert out.startswith("Memory and indexes:")
        assert "Recent notes" not in out

    def test_the_summary_lists_the_tiers(self, config, tmp_path):
        from agent.memory.overview import run_memory_command
        store = MemoryStore(tmp_path / ".agent" / "memory.db")
        store.add(scope="note", title="prefer tabs", body="the user said so")

        out = run_memory_command(config)

        assert "notes" in out and "prefer tabs" in out

    def test_notes_subcommand_takes_a_limit(self, config, tmp_path):
        from agent.memory.overview import run_memory_command
        store = MemoryStore(tmp_path / ".agent" / "memory.db")
        for i in range(5):
            store.add(scope="note", title=f"note {i}", body="body")

        out = run_memory_command(config, "notes 2")

        assert out.count("\n  - ") == 2

    def test_index_subcommand_without_an_index(self, config):
        from agent.memory.overview import run_memory_command
        assert "No code index" in run_memory_command(config, "index")

    def test_unknown_subcommand_shows_usage(self, config):
        from agent.memory.overview import run_memory_command
        assert "Usage:" in run_memory_command(config, "wat")

    def test_the_index_path_is_resolved_against_the_project(self, config, tmp_path):
        """Relative rag.db_path read from another cwd must not open a new db."""
        from agent.memory.overview import rag_db_path
        config.rag.db_path = ".agent/index.db"
        assert rag_db_path(config) == tmp_path / ".agent" / "index.db"


class TestWiring:
    def test_every_dispatcher_knows_the_command(self):
        root = Path(__file__).resolve().parents[2] / "ui"
        for name in ("slash.py", "slash_mixin.py", "readline_loop.py", "http_loop.py"):
            assert "/memory" in (root / name).read_text(encoding="utf-8"), name


    def test_the_endpoint_is_routed(self):
        assert '"/api/memory"' in HTTP_LOOP

    def test_the_panel_exists_in_the_page(self):
        assert 'id="memfold"' in _PAGE and 'id="membody"' in _PAGE

    def test_the_browser_loads_it(self):
        assert "loadMemory" in APP_JS and "/api/memory" in APP_JS
        assert "['memfold', loadMemory]" in APP_JS
