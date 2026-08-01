"""A turn says which files it changed — with per-round diffs, not the tree's.

Before this, the browser re-derived the changed-file list from mutating
tool-call arguments (a second, drifting copy of core.changeset.MUTATING_TOOLS)
and clicking a name ran `git diff` over the *current* working tree, so the
diff shown for an early round changed as later rounds touched the same file.
Now the server emits the round's own core.changeset.Changeset as a
`changeset` SSE event and the browser only renders it; /api/changeset serves
the diff that was actually captured for that (turn, file), reading it back
from the persisted A-record (falling back to a spilled diff file) so a
reload shows exactly what a live round showed.
"""
import asyncio
import json
from pathlib import Path

import pytest

from agent.memory import session as session_mod
from agent.ui import http_loop
from agent.memory.qa_log import QALogger

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
APP_CSS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.css"
           ).read_text(encoding="utf-8")


class TestClientSideDerivationIsGone:
    def test_the_tool_argument_parse_no_longer_exists(self):
        """core.changeset.MUTATING_TOOLS / paths_from_tool_call is now the
        only parse — the JS copy (toolPaths/noteFiles/MUTATING_TOOLS) and the
        button-strip renderer (filesStrip) it fed are dead code."""
        assert "function toolPaths(" not in APP_JS
        assert "function noteFiles(" not in APP_JS
        assert "function filesStrip(" not in APP_JS
        assert "MUTATING_TOOLS" not in APP_JS

    def test_tool_call_folds_no_longer_note_files(self):
        for fn, end in (("function toolCall(", "function toolResult("),
                        ("function replayToolCall(", "function replayToolResult(")):
            body = APP_JS[APP_JS.index(fn):APP_JS.index(end)]
            assert "noteFiles(" not in body, fn


class TestChangesetEvent:
    def test_it_is_stashed_on_the_open_turn_not_rendered_immediately(self):
        """`changeset` fires before `response`, while the turn's work fold is
        still open — it must land on that turn, not whatever comes next."""
        i = APP_JS.index("ev.type === 'changeset'")
        j = APP_JS.index("ev.type === 'response'")
        assert i < j
        assert "turn.changeset = ev" in APP_JS[i:j]

    def test_endturn_mounts_it(self):
        i = APP_JS.index("function endTurn()")
        assert "if (t.changeset) mount(renderChangeset(t.changeset));" in APP_JS[i:i + 1600]

    def test_the_turn_carries_a_changeset_slot(self):
        i = APP_JS.index("function beginTurn()")
        assert "changeset: null" in APP_JS[i:i + 900]
        assert "files: []" not in APP_JS[i:i + 900]


class TestThreeTiers:
    RENDER = APP_JS[APP_JS.index("function renderChangeset("):
                     APP_JS.index("function toolCall(")]

    def test_inline_shows_the_diff_expanded(self):
        assert "cs.tier === 'inline'" in self.RENDER

    def test_list_and_inline_share_a_flat_label(self):
        assert "fc-label" in self.RENDER

    def test_count_unfolds_from_a_details_summary(self):
        assert "cs.tier === 'count'" in self.RENDER
        assert "cs-summary" in self.RENDER
        assert "createElement('details')" in self.RENDER

    def test_each_file_is_its_own_nested_fold(self):
        row = APP_JS[APP_JS.index("function csFileRow("):APP_JS.index("function renderChangeset(")]
        assert "document.createElement('details')" in row

    def test_inline_rows_are_expanded_eagerly(self):
        row = APP_JS[APP_JS.index("function csFileRow("):APP_JS.index("function renderChangeset(")]
        assert "if (eager)" in row and "d.open = true" in row


class TestEscaping:
    """Paths and diff text are model-supplied strings landing in innerHTML."""

    def test_the_file_path_is_escaped(self):
        row = APP_JS[APP_JS.index("function csFileRow("):APP_JS.index("function renderChangeset(")]
        assert "esc(f.path)" in row

    def test_diff_text_is_escaped(self):
        i = APP_JS.index("function renderDiff(")
        assert "esc(text)" in APP_JS[i:i + 200]

    def test_a_fetched_diff_error_is_escaped(self):
        i = APP_JS.index("function csLoadDiff(")
        assert "esc(d.error)" in APP_JS[i:i + 700]

    def test_foreign_actor_names_are_escaped(self):
        i = APP_JS.index("function csForeignNote(")
        assert "foreign_actors.map(esc)" in APP_JS[i:i + 700]


class TestForeignEditWarning:
    NOTE = APP_JS[APP_JS.index("function csForeignNote("):APP_JS.index("function csLoadDiff(")]

    def test_named_actors_are_listed(self):
        assert "also edited by" in self.NOTE

    def test_unattributed_edits_get_the_generic_warning(self):
        assert "changed outside any agent" in self.NOTE

    def test_the_warning_is_not_silently_dropped(self):
        row = APP_JS[APP_JS.index("function csFileRow("):APP_JS.index("function renderChangeset(")]
        assert "cs-warn" in row
        assert "csForeignNote(f)" in row


class TestStyled:
    def test_it_is_styled_consistently_with_the_old_strip(self):
        assert ".files-changed {" in APP_CSS
        assert ".files-changed .cs-file {" in APP_CSS
        assert ".files-changed .cs-file[open] > summary {" in APP_CSS
        assert ".cs-warn {" in APP_CSS
        assert ".cs-foreign" in APP_CSS


# ---------------------------------------------------------------------------
# Server side: /api/changeset and the from_a_data history path.
# ---------------------------------------------------------------------------


def _make_ui(session=None):
    from agent.ui.http_loop import _HttpUI
    loop = asyncio.new_event_loop()
    try:
        return _HttpUI(object(), session, loop)
    finally:
        loop.close()


def _file(path, **kw):
    base = {"path": path, "added": 0, "removed": 0, "status": "modified",
            "diff": None, "diff_ref": None, "binary": False, "truncated": False,
            "foreign_edit": False, "foreign_actors": []}
    base.update(kw)
    return base


def _changeset(turn_id=1, tier="list", truncated=False, files=None):
    return {"turn_id": turn_id, "tier": tier, "truncated": truncated,
            "files": files or []}


def _capture(session_id, turn_id, changeset=None, modified_files=None):
    """Write both the QA-log turn and the session.json qa_info needs to find
    it — a bare QA log with no session file reads as "session not found"."""
    from agent.memory.session import Session, save_session
    save_session(Session(id=session_id), [{"role": "user", "content": "hi"}])

    logger = QALogger(session_id)

    async def go():
        await logger.capture_q(turn_id, "do it")
        await logger.capture_a(turn_id, "done", changeset=changeset,
                                modified_files=modified_files)
    asyncio.run(go())


class TestChangesetEndpoint:
    def test_returns_the_stored_diff_for_a_known_turn_and_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        cs = _changeset(files=[_file("a.py", diff="+x\n", added=1)])
        _capture("s1", 1, changeset=cs)

        result = _make_ui().changeset_info("1", "a.py", "s1")
        assert result["diff"] == "+x\n"
        assert "error" not in result

    def test_reads_a_spilled_diff_via_diff_ref(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        cs = _changeset(files=[_file("big.py", diff_ref="ab12.diff", truncated=True)])
        _capture("s2", 1, changeset=cs)

        from agent.memory.session import get_session_full_dir
        spill_dir = get_session_full_dir("s2") / "changesets" / "1"
        spill_dir.mkdir(parents=True)
        (spill_dir / "ab12.diff").write_text("+spilled\n", encoding="utf-8")

        result = _make_ui().changeset_info("1", "big.py", "s2")
        assert result["diff"] == "+spilled\n"

    def test_unknown_turn_returns_a_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        QALogger("s3")  # no captures — empty history

        result = _make_ui().changeset_info("9", "a.py", "s3")
        assert "error" in result and "diff" not in result

    def test_unknown_file_in_a_known_turn_returns_a_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        cs = _changeset(files=[_file("a.py", diff="+x\n")])
        _capture("s4", 1, changeset=cs)

        result = _make_ui().changeset_info("1", "missing.py", "s4")
        assert "error" in result

    def test_an_unparseable_turn_is_rejected_without_a_traceback(self):
        result = _make_ui().changeset_info("not-a-number", "a.py", "s5")
        assert result == {"error": "invalid turn"}

    def test_no_session_and_no_active_session_is_a_clean_error(self):
        result = _make_ui().changeset_info("1", "a.py", "")
        assert result == {"error": "no active session"}


class TestHistoryUsesFromAData:
    """The qa_info "files" payload (backs the condensed view's reload of a
    saved session) must read through core.changeset.from_a_data, not
    `a["modified_files"]` directly, so it agrees with a live round and
    doesn't hand the JS side unrenderable dicts from old session shapes."""

    def test_reads_the_new_changeset_field(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        cs = _changeset(files=[_file("a.py", added=2), _file("b.py", added=1)])
        _capture("h1", 1, changeset=cs)

        result = _make_ui().qa_info("h1")
        assert result["turns"][0]["files"] == ["a.py", "b.py"]

    def test_falls_back_to_modified_files_for_old_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("h2", 1, modified_files=["x.py"])

        result = _make_ui().qa_info("h2")
        assert result["turns"][0]["files"] == ["x.py"]

    def test_handles_dict_shaped_modified_files_from_the_wild(self, tmp_path, monkeypatch):
        """Some sessions store {path, added, removed} dicts in modified_files,
        not bare strings — the old `a.get("modified_files") or []` handed
        those straight to the JS side, which only knew how to esc() a string."""
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("h3", 1)

        from agent.memory.session import get_session_full_dir
        a_dir = get_session_full_dir("h3") / "A"
        a_file = next(a_dir.glob("A-*.json"))
        data = json.loads(a_file.read_text(encoding="utf-8"))
        data["modified_files"] = [{"path": "y.py", "added": 3, "removed": 1}]
        data["changeset"] = {}
        a_file.write_text(json.dumps(data), encoding="utf-8")

        result = _make_ui().qa_info("h3")
        assert result["turns"][0]["files"] == ["y.py"]


class TestReplayShowsWhatEachRoundChanged:
    """Previewing a saved session must say what each round changed.

    It used to, by re-deriving paths from replayed tool-call arguments. That
    derivation is gone, so the server now hangs each round's changeset on the
    assistant message that carried its response. The mapping deliberately does
    NOT assume "Nth user message is turn N": compaction rewrites the message
    array while turn ids keep counting, so that guess breaks on exactly the
    long sessions where replay is worth having.
    """

    def _messages(self, *contents):
        return [{"role": "assistant", "content": c} for c in contents]

    def test_a_round_is_matched_to_the_message_carrying_its_answer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("r1", 1, changeset=_changeset(files=[_file("a.py", added=2)]))

        messages = self._messages("unrelated", "done")
        http_loop._attach_changesets("r1", messages)

        assert "changeset" not in messages[0]
        assert [f["path"] for f in messages[1]["changeset"]["files"]] == ["a.py"]

    def test_a_turn_whose_message_was_compacted_away_is_skipped(self, tmp_path, monkeypatch):
        """Better a missing file list than one bound to the wrong round."""
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("r2", 1, changeset=_changeset(files=[_file("a.py", added=2)]))

        messages = self._messages("a summary that replaced the original turns")
        http_loop._attach_changesets("r2", messages)

        assert "changeset" not in messages[0]

    def test_rounds_bind_in_order_so_a_repeated_answer_cannot_steal_an_earlier_one(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("r3", 1, changeset=_changeset(files=[_file("first.py", added=1)]))
        _capture("r3", 2, changeset=_changeset(files=[_file("second.py", added=1)]))

        messages = self._messages("done", "done")
        http_loop._attach_changesets("r3", messages)

        assert [f["path"] for f in messages[0]["changeset"]["files"]] == ["first.py"]
        assert [f["path"] for f in messages[1]["changeset"]["files"]] == ["second.py"]

    def test_a_round_that_changed_nothing_adds_no_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("r4", 1)

        messages = self._messages("done")
        http_loop._attach_changesets("r4", messages)

        assert "changeset" not in messages[0]

    def test_a_session_with_no_qa_log_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        messages = self._messages("done")
        http_loop._attach_changesets("nope", messages)

        assert messages == [{"role": "assistant", "content": "done"}]


class TestReplayMountsIt:
    def test_the_replayed_turn_carries_the_changeset_into_end_turn(self):
        i = APP_JS.index("function replayTranscriptInner(")
        body = APP_JS[i:i + 1200]
        assert "if (turn && m.changeset) turn.changeset = m.changeset;" in body
        assert body.index("turn.changeset = m.changeset") < body.index("endTurn();\n        work = null;")


class TestJournalFoldsWhenTheNextRoundStarts:
    """A finished round's journal should stay readable until it is stale.

    The work fold used to collapse the moment the round ended, hiding the
    freshest thing on screen. It now stays open until the next round begins,
    and a round the user opened by hand is never folded for them.
    """

    BEGIN = APP_JS[APP_JS.index("function beginTurn("):APP_JS.index("function metaMount(")]
    END = APP_JS[APP_JS.index("function endTurn("):APP_JS.index("function endTurn(") + 1400]

    def test_the_round_that_just_ended_is_left_open_by_default(self):
        assert "foldJournal === 'immediately'" in self.END
        assert "if (!t.userToggled) t.details.open = false;" not in self.END

    def test_the_previous_round_folds_when_a_new_one_starts(self):
        assert "foldJournal === 'on_next_round'" in self.BEGIN
        assert "lastTurn.details.open = false" in self.BEGIN

    def test_a_round_the_user_opened_is_never_folded_for_them(self):
        assert "!lastTurn.userToggled" in self.BEGIN

    def test_never_mode_folds_nothing(self):
        """Neither branch fires, so no auto-fold path can run."""
        for block in (self.BEGIN, self.END):
            assert "'never'" not in block

    def test_the_click_handler_binds_to_its_own_turn(self):
        """It compared against the live `turn`, so a click on an already
        finished round never registered as a manual toggle."""
        assert "if (turn && turn.details === d) turn.userToggled = true;" not in APP_JS
        assert "rec.userToggled = true;" in self.BEGIN

    def test_rebuilding_the_log_drops_the_stale_turn_reference(self):
        assert APP_JS.count("turn = null; lastTurn = null;") == 4

    def test_the_mode_comes_from_the_server(self):
        assert "if (s.fold_journal) foldJournal = s.fold_journal;" in APP_JS


def _ui_with_config(mode=None):
    """_make_ui's server is a bare object(); _fold_journal reads the agent's
    config off it, so give it just enough to walk."""
    from agent.config import Config
    cfg = Config()
    if mode is not None:
        cfg.ui.changeset.fold_journal = mode
    ui = _make_ui()
    ui.server = type("S", (), {"_agent": type("A", (), {"config": cfg})()})()
    return ui


class TestFoldJournalSetting:
    def test_it_defaults_to_holding_the_round_open(self):
        assert _ui_with_config()._fold_journal() == "on_next_round"

    @pytest.mark.parametrize("mode", ["on_next_round", "immediately", "never"])
    def test_each_valid_mode_is_served(self, mode):
        assert _ui_with_config(mode)._fold_journal() == mode

    def test_an_unknown_mode_falls_back_instead_of_reaching_the_browser(self):
        assert _ui_with_config("sideways")._fold_journal() == "on_next_round"

    def test_a_server_with_no_agent_still_answers(self):
        assert _make_ui()._fold_journal() == "on_next_round"


class TestReconnectKeepsTheFileLists:
    def test_the_state_payload_attaches_per_round_changesets(self):
        """/api/state replays the transcript after a reconnect, so it needs the
        same attachment the preview pane gets — this was missed when replay was
        first restored, leaving reconnects without file lists."""
        import inspect
        from agent.ui.http_loop import _HttpUI

        src = inspect.getsource(_HttpUI.state)
        assert "_attach_changesets(self.session.id, messages)" in src


class TestSessionRollup:
    """The session total under a round's block, the browser's half of the
    rollup the readline UI serves as `/changes session`. Server-computed so all
    three UIs agree on the wording and on which rounds count."""

    def _ui(self, session_id="s-roll", session_rollup=True):
        from agent.config import Config
        from agent.memory.session import Session
        cfg = Config()
        cfg.ui.changeset.session_rollup = session_rollup
        ui = _make_ui(Session(id=session_id))
        ui.server = type("S", (), {"_agent": type("A", (), {"config": cfg})()})()
        return ui

    def test_it_counts_rounds_from_the_log_the_page_never_saw(self, tmp_path, monkeypatch):
        """A page opened against a resumed session shows the whole session."""
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("s-roll", 1, changeset=_changeset(files=[_file("old.py", added=10)]))
        rollup = self._ui().session_rollup()
        assert rollup["line"] == "session: 1 file changed, +10 -0"

    def test_the_live_round_is_folded_in(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("s-roll", 1, changeset=_changeset(files=[_file("old.py", added=10)]))
        from agent.core.changeset import Changeset, FileChange
        live = Changeset(turn_id=2, files=[FileChange(path="new.py", added=1)])
        assert self._ui().session_rollup(live)["files"] == 2

    def test_a_round_already_in_the_log_is_not_double_counted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("s-roll", 1, changeset=_changeset(files=[_file("a.py", added=10)]))
        from agent.core.changeset import Changeset, FileChange
        same = Changeset(turn_id=1, files=[FileChange(path="a.py", added=10)])
        assert self._ui().session_rollup(same)["added"] == 10

    def test_the_changeset_event_carries_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        _capture("s-roll", 1, changeset=_changeset(files=[_file("a.py", added=2)]))
        from agent.core.changeset import Changeset, FileChange
        cs = Changeset(turn_id=2, files=[FileChange(path="b.py", added=1)])
        event = self._ui().changeset_event(cs)
        assert event["type"] == "changeset"
        assert [f["path"] for f in event["files"]] == ["b.py"]     # the round
        assert event["rollup"]["files"] == 2                        # the session

    def test_the_config_toggle_leaves_it_off_the_event(self, tmp_path, monkeypatch):
        monkeypatch.setattr(session_mod, "_session_dir", tmp_path)
        from agent.core.changeset import Changeset, FileChange
        cs = Changeset(turn_id=1, files=[FileChange(path="a.py", added=1)])
        assert "rollup" not in self._ui(session_rollup=False).changeset_event(cs)

    def test_replay_hangs_it_on_the_last_round_only(self):
        messages = [
            {"role": "assistant", "content": "one", "changeset": {"files": [{"path": "a.py"}]}},
            {"role": "assistant", "content": "two", "changeset": {"files": [{"path": "b.py"}]}},
        ]
        http_loop._attach_session_rollup(messages, {"line": "session: 2 files changed"})
        assert "rollup" not in messages[0]["changeset"]
        assert messages[1]["changeset"]["rollup"]["line"] == "session: 2 files changed"

    def test_no_rollup_changes_nothing(self):
        messages = [{"role": "assistant", "changeset": {"files": []}}]
        http_loop._attach_session_rollup(messages, None)
        assert "rollup" not in messages[0]["changeset"]


class TestRollupRendering:
    def test_the_browser_renders_the_footer_under_both_layouts(self):
        body = APP_JS[APP_JS.index("function renderChangeset("):]
        body = body[:body.index("function toolCall(")]
        assert body.count("wrap.appendChild(rollup)") == 2   # count tier + list/inline

    def test_it_is_skipped_when_the_server_sent_none(self):
        assert "if (!cs.rollup || !cs.rollup.line) return null;" in APP_JS

    def test_it_is_styled_dimmer_than_the_round_headline(self):
        assert ".files-changed .cs-rollup" in APP_CSS
