"""Prompt loop parsing/state + scheduler result delivery cursor."""
import json

from agent.core.prompt_loop import PromptLoop, parse_loop_args


def test_parse_status_and_stop():
    assert parse_loop_args("") == ("status", {})
    assert parse_loop_args("stop")[0] == "stop"
    assert parse_loop_args("off")[0] == "stop"


def test_parse_interval_count_prompt():
    action, p = parse_loop_args("5m x20 check the build")
    assert action == "start"
    assert p["interval"] == 300.0
    assert p["limit"] == 20
    assert p["prompt"] == "check the build"


def test_parse_bare_number_is_prompt_not_interval():
    action, p = parse_loop_args("5 ideas for the readme")
    assert action == "start"
    assert p["interval"] == 0.0
    assert p["prompt"] == "5 ideas for the readme"


def test_parse_interval_floor():
    _, p = parse_loop_args("1s poll")
    assert p["interval"] == 5.0


def test_parse_no_prompt_is_error():
    assert parse_loop_args("5m")[0] == "error"
    assert parse_loop_args("x3")[0] == "error"


def test_loop_state_limit():
    pl = PromptLoop()
    pl.start("p", 0.0, 2)
    assert pl.record_iteration()          # 1/2 -> continue
    assert not pl.record_iteration()      # 2/2 -> done
    assert not pl.active
    assert not pl.record_iteration()      # inactive -> no count
    assert pl.done == 2


def test_loop_state_unlimited_until_stop():
    pl = PromptLoop()
    pl.start("p", 30.0, 0)
    for _ in range(5):
        assert pl.record_iteration()
    pl.stop()
    assert not pl.record_iteration()
    assert pl.done == 5


def test_unseen_results_cursor(tmp_path, monkeypatch):
    from agent.core import scheduler

    monkeypatch.setattr(scheduler, "_schedule_dir", lambda cfg: tmp_path)
    runs = tmp_path / "runs.jsonl"

    def rec(name, status="ok", result=""):
        with open(runs, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": 0, "job": name, "name": name,
                                "status": status, "session": "", "result": result}) + "\n")

    cfg = object()
    rec("old-1")
    # First call initializes the cursor: history is NOT delivered.
    assert scheduler.unseen_results(cfg) == []
    rec("new-1", result="did the thing")
    rec("new-2", status="error: boom")
    got = scheduler.unseen_results(cfg)
    assert [r["name"] for r in got] == ["new-1", "new-2"]
    # Delivered once only.
    assert scheduler.unseen_results(cfg) == []
    # Truncated/rotated file resets cleanly.
    runs.write_text("")
    assert scheduler.unseen_results(cfg) == []
    rec("after-rotate")
    assert [r["name"] for r in scheduler.unseen_results(cfg)] == ["after-rotate"]
