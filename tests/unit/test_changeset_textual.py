"""Unit tests for the Textual UI's changeset rendering (ui/event_mixin.py,
ui/textual_widgets.py, ui/view_mixin.py).

These guard the P2 rewrite that replaced ``_compute_file_diffs`` — a
``git diff --numstat`` shell-out that reported every uncommitted change in a
touched file (not just this round's), returned 0/0 for newly created
untracked files, and mis-attributed paths by suffix matching — with rendering
straight off ``core.changeset.Changeset``, the agent's own per-round record
sourced from the edit journal. A regression here means either the tiered
disclosure (inline/list/count) breaks, a foreign_edit warning goes silently
missing, or a restored turn stops matching what was shown live.
"""
from __future__ import annotations

import asyncio

from agent.core.changeset import Changeset, FileChange, from_a_data
from agent.ui import event_mixin as em
from agent.ui.textual_widgets import build_widget_classes


class _Theme:
    def __getattr__(self, k):
        return "white"


class _FakeChatLog:
    def __init__(self):
        self.lines: list = []


class _FakeApp:
    def __init__(self):
        self._t = _Theme()
        self._chat_file_lines: dict = {}
        self._chat_changeset_lines: dict = {}


def _write(app, cs) -> _FakeChatLog:
    log = _FakeChatLog()

    def _cw(text):
        log.lines.append(text)

    em.write_changeset_rows(app, _cw, log, cs)
    return log


class TestTieredRendering:
    def test_inline_tier_shows_header_row_and_diff(self):
        fc = FileChange(path="a.py", added=1, removed=0, status="added",
                         diff="--- a/a.py\n+++ b/a.py\n@@\n+hello\n")
        cs = Changeset(turn_id=1, files=[fc], tier="inline")
        app = _FakeApp()
        joined = "\n".join(_write(app, cs).lines)
        assert "1 file changed" in joined
        assert "a.py" in joined
        assert "+hello" in joined
        assert app._chat_file_lines

    def test_list_tier_shows_rows_but_not_diffs(self):
        files = [FileChange(path=f"f{i}.py", added=1, diff=f"+line{i}\n")
                 for i in range(2)]
        cs = Changeset(turn_id=2, files=files, tier="list")
        app = _FakeApp()
        joined = "\n".join(_write(app, cs).lines)
        assert "f0.py" in joined and "f1.py" in joined
        assert "+line0" not in joined and "+line1" not in joined
        assert len({id(v) for v in app._chat_file_lines.values()}) == 2

    def test_count_tier_shows_only_the_headline(self):
        files = [FileChange(path=f"f{i}.py", added=1) for i in range(5)]
        cs = Changeset(turn_id=3, files=files, tier="count")
        app = _FakeApp()
        log = _write(app, cs)
        assert len(log.lines) == 1
        assert "5 files changed" in log.lines[0]
        assert not app._chat_file_lines
        assert next(iter(app._chat_changeset_lines.values())) is cs

    def test_an_empty_changeset_writes_nothing(self):
        app = _FakeApp()
        log = _write(app, Changeset())
        assert log.lines == []
        assert not app._chat_file_lines and not app._chat_changeset_lines


class TestForeignEditWarning:
    def test_a_foreign_edit_renders_a_visible_warning_row(self):
        fc = FileChange(path="shared.py", added=1, removed=1, foreign_edit=True)
        cs = Changeset(turn_id=4, files=[fc], tier="list")
        log = _write(_FakeApp(), cs)
        assert any("outside any agent" in l and "⚠" in l for l in log.lines)

    def test_an_actor_named_foreign_edit_names_them_in_the_row(self):
        fc = FileChange(path="shared.py", foreign_edit=True,
                         foreign_actors=["owncoder-4711"])
        cs = Changeset(turn_id=5, files=[fc], tier="list")
        log = _write(_FakeApp(), cs)
        assert any("owncoder-4711" in l for l in log.lines)

    def test_an_untouched_file_gets_no_warning_row(self):
        fc = FileChange(path="clean.py", added=1)
        cs = Changeset(turn_id=6, files=[fc], tier="list")
        log = _write(_FakeApp(), cs)
        assert not any("⚠" in l for l in log.lines)


class TestRestoredHistoryMatchesLive:
    """The path view_mixin._render_full_turn and textual_widgets._dedup_files
    actually take: changeset.from_a_data → the same write_changeset_rows a
    live turn uses."""

    def test_a_legacy_modified_files_record_renders_through_the_same_path(self):
        a_data = {"turn_id": 7, "modified_files": [
            {"path": "old.py", "added": 4, "removed": 2},
        ]}
        cs = from_a_data(a_data)
        app = _FakeApp()
        joined = "\n".join(_write(app, cs).lines)
        assert "old.py" in joined and "+4" in joined and "-2" in joined
        assert app._chat_file_lines

    def test_a_record_carrying_a_full_changeset_reproduces_its_diff(self):
        stored = {"turn_id": 8, "changeset": {
            "turn_id": 8, "tier": "inline", "truncated": False,
            "files": [{"path": "n.py", "added": 1, "removed": 0,
                       "status": "added", "diff": "+new\n"}],
        }}
        cs = from_a_data(stored)
        log = _write(_FakeApp(), cs)
        assert any("+new" in l for l in log.lines)

    def test_the_turn_detail_screen_dedups_restored_files_by_path(self):
        """dict.fromkeys() used to crash on dict entries (unhashable); the
        fix routes through changeset.from_a_data, which dedups by path and
        keeps the first entry."""
        async def go():
            from textual.app import App
            from textual.widgets import Button

            w = build_widget_classes(_Theme())
            a_data = {
                "turn_id": 9, "content": "done", "tool_calls": [],
                "modified_files": [
                    {"path": "a.py", "added": 3, "removed": 1},
                    {"path": "a.py", "added": 9, "removed": 9},  # dup path
                    "legacy.py",  # legacy string form
                ],
            }
            screen = w.TurnDetailScreen(0, {"turn_id": 9, "content": "q"}, a_data)

            class _App(App):
                async def on_mount(self) -> None:
                    await self.push_screen(screen)

            async with _App().run_test() as pilot:
                await pilot.pause()
                file_btns = [b for b in pilot.app.screen.query(Button)
                             if str(b.id or "").startswith("file-btn-")]
                assert len(file_btns) == 2
                assert "a.py" in str(file_btns[0].label)
                await pilot.app.action_quit()

        asyncio.run(go())


class TestGitShellOutIsGone:
    def test_compute_file_diffs_no_longer_exists(self):
        assert not hasattr(em, "_compute_file_diffs")
        assert "_compute_file_diffs" not in dir(em)


class _Server:
    """Enough of a UIServer for the rollup helper: it reaches through
    ``_server._agent.config`` for the ui.changeset section."""

    def __init__(self, config=None):
        self._agent = type("A", (), {"config": config})()


class _Session:
    def __init__(self, sid):
        self.id = sid


class _Config:
    def __init__(self, session_rollup=True):
        section = type("S", (), {"enabled": True, "session_rollup": session_rollup})()
        self.ui = type("U", (), {"changeset": section})()


class TestSessionRollupFooter:
    """The session total pinned under a round's changeset block.

    The readline UI has had ``/changes session`` since the feature shipped;
    this is the same merge (core.changeset.merge_changesets) rendered as a
    footer, so all three UIs say the same thing.
    """

    def _app(self, config=None, sid="s1", last=None):
        app = _FakeApp()
        app._server = _Server(config if config is not None else _Config())
        app._session = _Session(sid)
        app._last_changeset = last
        return app

    def test_the_rollup_reads_the_session_log_not_just_this_round(self, monkeypatch):
        """A resumed session must roll up rounds it never watched live."""
        history = [(1, {}, {"turn_id": 1, "changeset": {
            "turn_id": 1, "files": [{"path": "old.py", "added": 10}]}})]
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: history)
        app = self._app(last=Changeset(turn_id=2, files=[FileChange(path="new.py", added=1)]))
        assert em.session_rollup_line(app) == "session: 2 files changed, +11 -0"

    def test_the_footer_is_written_under_the_block(self, monkeypatch):
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: [])
        cs = Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)], tier="list")
        app = self._app(last=cs)
        log = _FakeChatLog()
        em.write_changeset_rows(app, log.lines.append, log, cs, em.session_rollup_line(app))
        assert "session: 1 file changed, +1 -0" in log.lines[-1]

    def test_a_count_tier_block_still_gets_the_footer(self, monkeypatch):
        """count returns early after the headline — the footer must survive it."""
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: [])
        cs = Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)], tier="count")
        app = self._app(last=cs)
        log = _FakeChatLog()
        em.write_changeset_rows(app, log.lines.append, log, cs, "session: 1 file changed")
        assert "session: 1 file changed" in log.lines[-1]

    def test_history_re_render_gets_no_footer(self):
        """A rollup under an old round would state a total that was not true
        when that round ended, so replay passes none."""
        cs = Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)], tier="list")
        log = _write(_FakeApp(), cs)
        assert not any("session:" in line for line in log.lines)

    def test_the_config_toggle_silences_it(self, monkeypatch):
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: [])
        app = self._app(config=_Config(session_rollup=False),
                        last=Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)]))
        assert em.session_rollup_line(app) == ""

    def test_a_round_offered_twice_is_counted_once(self, monkeypatch):
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", lambda sid: [])
        app = self._app(last=Changeset(turn_id=1, files=[FileChange(path="a.py", added=3)]))
        assert em.session_rollup_line(app) == em.session_rollup_line(app)
        assert em.session_rollup_line(app) == "session: 1 file changed, +3 -0"

    def test_an_unreadable_log_costs_the_footer_not_the_round(self, monkeypatch):
        def _boom(sid):
            raise OSError("gone")
        monkeypatch.setattr("agent.memory.qa_log.read_history_sync", _boom)
        app = self._app(last=Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)]))
        assert em.session_rollup_line(app) == "session: 1 file changed, +1 -0"
