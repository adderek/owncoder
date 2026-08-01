"""On-demand heal: the user can ask the agent to diagnose itself mid-session.

The failure mode this covers is the one from the field report: tool calls kept
failing, the agent escalated to a stronger model, and that failed too — because
the root cause was the tool *schema*, which no amount of model power fixes. The
heal has to put that schema in front of the agent.
"""
import json
from pathlib import Path

from agent.core import self_heal
from agent.ui.http_loop import _HttpUI

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")
HTTP_LOOP = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
             ).read_text(encoding="utf-8")


class _Tools:
    def __init__(self, working_dir, agent_dir=".agent"):
        self.working_dir = str(working_dir)
        self.agent_dir = agent_dir


class _Config:
    def __init__(self, working_dir):
        self.tools = _Tools(working_dir)


def _write_failures(tmp_path, records):
    d = Path(tmp_path) / ".agent" / "failures"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "index.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestFailureClusters:
    def test_repeats_are_grouped_and_counted(self, tmp_path):
        """Ten identical reports are one diagnosis — the count is the signal."""
        _write_failures(tmp_path, [
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "edit_file",
             "reason": "unknown argument 'path'", "session_id": "s1"},
        ] * 3 + [
            {"ts": "2026-08-01T10:01:00", "kind": "exception", "tool": "shell",
             "reason": "boom", "session_id": "s1"},
        ])
        out = self_heal._failure_clusters(_Config(tmp_path), "s1")
        assert out[0]["tool"] == "edit_file" and out[0]["count"] == 3
        assert [c["count"] for c in out] == [3, 1]

    def test_other_sessions_are_excluded(self, tmp_path):
        _write_failures(tmp_path, [
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "a",
             "reason": "x", "session_id": "s1"},
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "b",
             "reason": "x", "session_id": "other"},
        ])
        out = self_heal._failure_clusters(_Config(tmp_path), "s1")
        assert [c["tool"] for c in out] == ["a"]

    def test_mislabelled_reports_fall_back_to_the_session_window(self, tmp_path):
        """A batch stamped with the wrong session id must not hide everything."""
        _write_failures(tmp_path, [
            {"ts": "2026-07-01T09:00:00", "kind": "old", "tool": "a",
             "reason": "before", "session_id": "zzz"},
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "b",
             "reason": "after", "session_id": "zzz"},
        ])
        out = self_heal._failure_clusters(_Config(tmp_path), "2026-08-01T09:00:00_abc")
        assert [c["tool"] for c in out] == ["b"]

    def test_a_missing_index_is_not_an_error(self, tmp_path):
        assert self_heal._failure_clusters(_Config(tmp_path), "s1") == []


class TestTranscriptErrors:
    def test_it_pairs_the_error_with_the_arguments_that_caused_it(self):
        """Schema vs. what the model actually sent is the whole diagnosis."""
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "edit_file",
                                          "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": json.dumps({"error": "unknown argument 'path'"})},
            {"role": "assistant", "tool_calls": [
                {"id": "c2", "function": {"name": "edit_file",
                                          "arguments": '{"file": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": json.dumps({"error": "unknown argument 'file'"})},
        ]
        out = self_heal._tool_errors(msgs)
        assert out == [{"tool": "edit_file", "count": 2,
                        "error": "unknown argument 'path'",
                        "args": '{"path": "a.py"}'}]

    def test_successful_results_are_ignored(self):
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "shell", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"ok": 1})},
            {"role": "tool", "tool_call_id": "c1", "content": "plain text output"},
        ]
        assert self_heal._tool_errors(msgs) == []


class TestSignals:
    def test_suspect_tools_carry_their_current_schema(self, tmp_path, monkeypatch):
        """The motivating root cause is the contract, not the model."""
        monkeypatch.setattr(
            "agent.tools.get_schemas",
            lambda: [{"type": "function", "function": {
                "name": "edit_file", "description": "edit a file",
                "parameters": {"properties": {"file_path": {"type": "string"}}}}}])
        msgs = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "function": {"name": "edit_file",
                                          "arguments": '{"path": "a.py"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": json.dumps({"error": "unknown argument 'path'"})},
        ]
        signals = self_heal.collect_signals(_Config(tmp_path), "s1", msgs)
        assert signals["suspects"] == ["edit_file"]
        assert "file_path" in signals["schemas"][0]["schema"]

        prompt = self_heal.build_prompt(signals, focus="tool calls keep failing")
        assert "file_path" in prompt                    # the schema
        assert '{"path": "a.py"}' in prompt             # what the model sent
        assert "tool calls keep failing" in prompt      # the user's observation
        assert "ROOT CAUSE" in prompt

    def test_the_prompt_names_the_escalation_trap(self, tmp_path):
        """Escalating the model does not fix a broken tool contract."""
        prompt = self_heal.build_prompt(
            self_heal.collect_signals(_Config(tmp_path), "s1", []))
        assert "Escalating the model does not fix a broken" in prompt

    def test_no_signals_still_produces_a_usable_prompt(self, tmp_path):
        signals = self_heal.collect_signals(_Config(tmp_path), "s1", [])
        assert self_heal.summary_line(signals) == \
            "no failure signals recorded this session"
        assert "NO RECORDED SIGNALS" in self_heal.build_prompt(signals)

    def test_collection_survives_a_broken_config(self, tmp_path, monkeypatch):
        """A heal asked for while something is already broken must not raise."""
        monkeypatch.chdir(tmp_path)
        signals = self_heal.collect_signals(object(), "s1", None)
        assert signals["counts"]["failures"] == 0
        assert "SELF-DIAGNOSIS" in self_heal.build_prompt(signals)

    def test_the_summary_counts_what_it_found(self, tmp_path):
        _write_failures(tmp_path, [
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "edit_file",
             "reason": "unknown argument", "session_id": "s1"}] * 4)
        line = self_heal.summary_line(
            self_heal.collect_signals(_Config(tmp_path), "s1", []))
        assert "4 failure report(s)" in line and "edit_file" in line


class _Server:
    def __init__(self, msgs, config):
        self.msgs = list(msgs)
        self._agent = type("A", (), {"config": config})()

    def get_messages(self):
        return self.msgs


class _Session:
    id = "s1"


def _ui(tmp_path, msgs=()):
    ui = _HttpUI.__new__(_HttpUI)
    ui.session = _Session()
    ui.server = _Server(msgs, _Config(tmp_path))
    ui.busy = False
    ui.submitted = []
    ui.bus = type("B", (), {"published": [],
                            "publish": lambda self, ev: self.published.append(ev)})()
    ui.submit = lambda text: ui.submitted.append(text) or False
    return ui


class TestHttpEndpoint:
    def test_the_preview_reports_the_evidence_without_spending_a_turn(self, tmp_path):
        _write_failures(tmp_path, [
            {"ts": "2026-08-01T10:00:00", "kind": "bad_args", "tool": "edit_file",
             "reason": "unknown argument", "session_id": "s1"}])
        ui = _ui(tmp_path)
        info = ui.heal_info()
        assert info["ok"] and "edit_file" in info["summary"]
        assert "edit_file" in info["evidence"]
        assert ui.submitted == []

    def test_the_heal_runs_in_the_current_session(self, tmp_path):
        """Not a side-channel call: the user stays on the session view and the
        agent keeps the tools it needs to actually fix what it finds."""
        ui = _ui(tmp_path)
        out = ui.heal_action({"focus": "deep model failed too"})
        assert out["ok"]
        assert len(ui.submitted) == 1
        assert "deep model failed too" in ui.submitted[0]
        assert "SELF-DIAGNOSIS REQUEST" in ui.submitted[0]

    def test_it_is_reachable_over_http(self):
        assert '"/api/heal"' in HTTP_LOOP
        assert "ui.heal_info()" in HTTP_LOOP and "ui.heal_action(payload)" in HTTP_LOOP

    def test_the_session_view_has_the_button(self):
        assert 'id="heal"' in HTTP_LOOP
        assert "getElementById('heal')" in APP_JS
        assert "'/api/heal'" in APP_JS
