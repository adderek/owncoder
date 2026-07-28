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
