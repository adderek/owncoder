"""A reload brings back the tool work, not just the answers.

/api/state and /api/history filtered the conversation down to user and
assistant text, so refreshing the page turned a reasoned turn into a bare
assertion — the evidence outlived by the conclusion.
"""
import json
from pathlib import Path

from agent.ui.http_loop import _transcript

APP_JS = (Path(__file__).resolve().parents[2] / "ui" / "static" / "app.js"
          ).read_text(encoding="utf-8")

CALL = {"role": "assistant", "content": "",
        "tool_calls": [{"id": "c1",
                        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]}


class TestShape:
    def test_a_plain_exchange_survives(self):
        out = _transcript([{"role": "user", "content": "hi"},
                           {"role": "assistant", "content": "hello"}])
        assert [m["role"] for m in out] == ["user", "assistant"]
        assert out[1]["content"] == "hello"

    def test_tool_calls_come_through_with_their_arguments(self):
        out = _transcript([CALL])
        call = out[0]["tool_calls"][0]
        assert call["id"] == "c1" and call["name"] == "bash"
        assert "cmd" in call["args"] and "cmd" in call["args_full"]

    def test_results_are_paired_by_id(self):
        out = _transcript([CALL, {"role": "tool", "tool_call_id": "c1",
                                  "content": json.dumps({"output": "a"})}])
        assert out[1]["role"] == "tool" and out[1]["id"] == "c1"
        assert out[1]["content"] == "a"

    def test_a_failed_call_is_marked(self):
        out = _transcript([{"role": "tool", "tool_call_id": "c1",
                            "content": json.dumps({"error": "boom"})}])
        assert out[0]["ok"] is False

    def test_a_plain_result_counts_as_success(self):
        out = _transcript([{"role": "tool", "tool_call_id": "c", "content": "fine"}])
        assert out[0]["ok"] is True

    def test_system_messages_stay_out(self):
        out = _transcript([{"role": "system", "content": "prompt"}])
        assert out == []

    def test_an_empty_assistant_turn_is_dropped(self):
        """No content and no calls is nothing to show."""
        assert _transcript([{"role": "assistant", "content": ""}]) == []

    def test_results_are_shortened_harder_than_live_events(self):
        """This is a whole session in one response, not one event."""
        out = _transcript([{"role": "tool", "tool_call_id": "c", "content": "x" * 9000}])
        assert len(out[0]["content"]) == 2001

    def test_sdk_objects_work_as_well_as_dicts(self):
        class Fn:
            name, arguments = "grep", '{"q": "x"}'

        class TC:
            id, function = "c9", Fn()

        out = _transcript([{"role": "assistant", "content": "", "tool_calls": [TC()]}])
        assert out[0]["tool_calls"][0] == {
            "id": "c9", "name": "grep", "args": "q='x'",
            "args_full": '{\n  "q": "x"\n}'}


class TestBothEndpointsUseIt:
    def test_live_state_and_session_history_agree(self):
        src = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
               ).read_text(encoding="utf-8")
        assert src.count("_transcript(") == 3      # definition + two call sites


class TestReplay:
    def test_the_browser_rebuilds_the_folds(self):
        i = APP_JS.index("function replayTranscript(")
        body = APP_JS[i:APP_JS.index("function applyState(")]
        assert "replayToolCall(" in body and "replayToolResult(" in body
        assert "beginTurn()" in body

    def test_replayed_folds_land_inside_the_work_fold(self):
        """metaMount routes by busyFlag, which is false while replaying."""
        i = APP_JS.index("function replayToolCall(")
        body = APP_JS[i:APP_JS.index("function replayToolResult(")]
        assert "turn.body.appendChild(d)" in body
        assert "metaMount(d);" not in body

    def test_a_replayed_turn_invents_no_duration(self):
        i = APP_JS.index("function endTurn()")
        body = APP_JS[i:i + 1200]
        assert "if (!t.replay) bits.push(secs" in body
        assert "'replayed from the session transcript'" in body

    def test_both_replay_paths_share_it(self):
        """Live state and the session preview render the same way."""
        assert APP_JS.count("replayTranscript(") == 3


class TestCollapsedRounds:
    """A resumed session shows the tool work a live one showed.

    History is stored folded — a round's calls become `<agent_exec>` tags in
    the assistant text — so replaying it verbatim printed that markup as prose
    and lost every fold. The transcript unfolds it back into call/result pairs.
    """

    FOLDED = {
        "role": "assistant",
        "content": ("Looking at the file.\n\n"
                    "<agent_exec tool=\"read_file\" args=\"path='a.py'\">"
                    "['content']</agent_exec>\n\n"
                    "Done."),
    }

    def test_the_markup_never_reaches_the_browser(self):
        for m in _transcript([self.FOLDED]):
            assert "<agent_exec" not in (m.get("content") or "")

    def test_the_call_and_its_result_come_back(self):
        out = _transcript([self.FOLDED])
        calls = [m for m in out if m.get("tool_calls")]
        assert calls and calls[0]["tool_calls"][0]["name"] == "read_file"
        results = [m for m in out if m["role"] == "tool"]
        assert results and results[0]["content"] == "['content']"

    def test_prose_keeps_its_place_around_the_calls(self):
        """The model wrote before the call and after it; both stay, in order."""
        texts = [m["content"] for m in _transcript([self.FOLDED])
                 if m["role"] == "assistant" and m["content"]]
        assert texts == ["Looking at the file.", "Done."]

    def test_an_error_result_is_marked(self):
        out = _transcript([{"role": "assistant", "content":
                            '<agent_exec tool="bash" args="cmd=\'x\'">'
                            'ERROR: boom</agent_exec>'}])
        assert [m for m in out if m["role"] == "tool"][0]["ok"] is False

    def test_the_side_log_supplies_the_full_arguments(self, tmp_path, monkeypatch):
        """The folded tag only kept a preview; the side-log kept everything."""
        import agent.ui.http_loop as H
        sdir = tmp_path / "s1"
        sdir.mkdir()
        (sdir / "tool_calls.jsonl").write_text(json.dumps({
            "seq": 4, "tool_call_id": "t4", "tool": "read_file",
            "arguments": {"path": "very/long/path/a.py"},
            "result": json.dumps({"output": "hello"}),
        }) + "\n", encoding="utf-8")
        monkeypatch.setattr("agent.memory.session.get_session_full_dir",
                            lambda sid: sdir)
        folded = {**self.FOLDED, "_tool_refs": [4]}
        out = H._transcript([folded], sid="s1")
        call = [m for m in out if m.get("tool_calls")][0]["tool_calls"][0]
        assert call["id"] == "t4"
        assert "very/long/path/a.py" in call["args_full"]
        assert [m for m in out if m["role"] == "tool"][0]["content"] == "hello"


class TestInjectedContext:
    """Context the agent injects for itself was never shown live, so replaying
    it as a user message put words in the user's mouth."""

    def test_recalled_sessions_stay_out_of_the_conversation(self):
        out = _transcript([{"role": "user", "content": "# Similar past sessions",
                            "_similar_sessions_marker": True},
                           {"role": "user", "content": "hi"}])
        assert [m["content"] for m in out] == ["hi"]

    def test_a_verify_failure_is_still_shown(self):
        """It is the one injected message the user has to see."""
        out = _transcript([{"role": "user", "content": "[verify] `pytest` failed"}])
        assert out and out[0]["content"].startswith("[verify]")

    def test_the_live_view_hears_about_them_too(self):
        """Same message, both views: the server publishes what it injects."""
        src = (Path(__file__).resolve().parents[2] / "ui" / "http_loop.py"
               ).read_text(encoding="utf-8")
        assert "on_injected_message=lambda text: pub(" in src
        assert '{"type": "user", "text": text}' in src


class TestReasoningSurvivesAReload:
    """Live streams the thinking into a fold and history keeps it; replay used
    to drop it, so a reloaded turn lost what explained it."""

    def test_it_rides_along_with_the_round(self):
        out = _transcript([{"role": "assistant", "content": "answer",
                            "_reasoning_content": "because of X"}])
        assert out[0]["reasoning"] == "because of X"

    def test_a_folded_round_carries_it_on_the_first_entry(self):
        out = _transcript([{"role": "assistant", "_reasoning_content": "why",
                            "content": '<agent_exec tool="ls" args="">ok</agent_exec>'}])
        assert out[0]["reasoning"] == "why"

    def test_a_long_trace_is_shortened(self):
        out = _transcript([{"role": "assistant", "content": "a",
                            "_reasoning_content": "x" * 9000}])
        assert len(out[0]["reasoning"]) == 4001

    def test_the_browser_mounts_it_inside_the_work_fold(self):
        i = APP_JS.index("function replayReasoning(")
        body = APP_JS[i:i + 500]
        assert "turn.body.appendChild(d)" in body
        assert "replayReasoning(m.reasoning)" in APP_JS
