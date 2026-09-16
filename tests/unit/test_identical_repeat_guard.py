"""Second identical call (same args + same result) -> error, temperature bump."""
import json
from types import SimpleNamespace

from agent.config import Config
from agent.core.turn import run_turn
from agent.core.turn_guards import flag_identical_repeats, temperature_override


def _tc(name: str, args: dict, cid: str = "c1"):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class TestFlagIdenticalRepeats:
    def test_first_call_passes_through(self):
        out, rep, prev = flag_identical_repeats([_tc("grep_code", {"q": "x"})], ["R"], {})
        assert out == ["R"] and rep == []
        assert list(prev.values()) == ["R"]

    def test_same_args_same_result_is_error(self):
        _, _, prev = flag_identical_repeats([_tc("grep_code", {"q": "x"})], ["R"], {})
        out, rep, _ = flag_identical_repeats([_tc("grep_code", {"q": "x"})], ["R"], prev)
        assert rep == ["grep_code"]
        err = json.loads(out[0])
        assert err["error_type"] == "IdenticalRepeat"
        assert "identical call aborted" in err["error"]

    def test_arg_order_does_not_matter(self):
        _, _, prev = flag_identical_repeats([_tc("t", {"a": 1, "b": 2})], ["R"], {})
        tc = SimpleNamespace(id="c", function=SimpleNamespace(name="t", arguments='{"b": 2, "a": 1}'))
        _, rep, _ = flag_identical_repeats([tc], ["R"], prev)
        assert rep == ["t"]

    def test_changed_result_is_not_flagged(self):
        _, _, prev = flag_identical_repeats([_tc("bash", {"cmd": "make"})], ["building"], {})
        out, rep, _ = flag_identical_repeats([_tc("bash", {"cmd": "make"})], ["done"], prev)
        assert rep == [] and out == ["done"]

    def test_changed_args_is_not_flagged(self):
        _, _, prev = flag_identical_repeats([_tc("grep_code", {"q": "x"})], ["R"], {})
        _, rep, _ = flag_identical_repeats([_tc("grep_code", {"q": "y"})], ["R"], prev)
        assert rep == []

    def test_only_previous_round_counts(self):
        _, _, p1 = flag_identical_repeats([_tc("a", {})], ["R"], {})
        _, _, p2 = flag_identical_repeats([_tc("b", {})], ["S"], p1)
        _, rep, _ = flag_identical_repeats([_tc("a", {})], ["R"], p2)
        assert rep == []


class TestTemperatureOverride:
    def test_raises_and_restores(self):
        cfg = Config(); cfg.llm.temperature = 0.2
        with temperature_override(cfg, 0.6):
            assert cfg.llm.temperature == 0.6
        assert cfg.llm.temperature == 0.2

    def test_never_lowers(self):
        cfg = Config(); cfg.llm.temperature = 0.9
        with temperature_override(cfg, 0.6):
            assert cfg.llm.temperature == 0.9
        assert cfg.llm.temperature == 0.9

    def test_none_is_noop(self):
        cfg = Config(); cfg.llm.temperature = 0.2
        with temperature_override(cfg, None):
            assert cfg.llm.temperature == 0.2

    def test_newer_value_set_inside_wins(self):
        cfg = Config(); cfg.llm.temperature = 0.2
        with temperature_override(cfg, 0.6):
            cfg.llm.temperature = 0.4   # e.g. model routing swapped the entry
        assert cfg.llm.temperature == 0.4


class _Client:
    """Emits the same grep_code call every request; records temperatures."""

    def __init__(self):
        self.temps: list[float] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.temps.append(kw.get("temperature"))
        msg = SimpleNamespace(content=None, tool_calls=[_tc("grep_code", {"q": "x"}, f"c{len(self.temps)}")])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None)


async def _run(monkeypatch, cfg):
    import agent.core.turn as turn_mod

    async def _exec(tc, config=None):
        return json.dumps({"matches": []})

    monkeypatch.setattr(turn_mod, "execute_tool", _exec)
    monkeypatch.setattr(turn_mod, "get_schemas", lambda: [])
    client = _Client()
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "go"}]
    _, out = await run_turn(msgs, cfg, client)
    tool_results = [m["content"] for m in out if m.get("role") == "tool"]
    return client, tool_results


async def test_run_turn_second_identical_call_gets_error_and_hotter_sampling(monkeypatch):
    cfg = Config()
    cfg.llm.temperature = 0.2
    cfg.loop_guard.repeat_threshold = 3
    cfg.llm.max_iterations = 10
    client, results = await _run(monkeypatch, cfg)
    assert json.loads(results[0]) == {"matches": []}
    assert json.loads(results[1])["error_type"] == "IdenticalRepeat"
    assert client.temps[0] == 0.2
    assert client.temps[1] == 0.2      # request that produced the repeat: not yet bumped
    assert client.temps[2] == 0.6      # request after the aborted repeat
    assert cfg.llm.temperature == 0.2  # config restored


async def test_run_turn_guard_disabled(monkeypatch):
    cfg = Config()
    cfg.llm.temperature = 0.2
    cfg.loop_guard.repeat_threshold = 3
    cfg.loop_guard.identical_repeat_error = False
    cfg.llm.max_iterations = 10
    client, results = await _run(monkeypatch, cfg)
    assert all("IdenticalRepeat" not in r for r in results)
    assert set(client.temps) == {0.2}


class TestEditsAndRepeats:
    def test_test_rerun_after_edit_round_is_not_flagged(self):
        _, _, prev = flag_identical_repeats([_tc("bash", {"cmd": "pytest"})], ["FAIL"], {})
        _, _, prev = flag_identical_repeats([_tc("edit_file", {"path": "a"})], ["ok"], prev)
        _, rep, _ = flag_identical_repeats([_tc("bash", {"cmd": "pytest"})], ["FAIL"], prev)
        assert rep == []

    def test_identical_edit_twice_is_flagged(self):
        call = {"path": "a", "anchor": "x", "new": "y"}
        _, _, prev = flag_identical_repeats([_tc("edit_file", call)], ["anchor not found"], {})
        _, rep, _ = flag_identical_repeats([_tc("edit_file", call)], ["anchor not found"], prev)
        assert rep == ["edit_file"]


def test_reworded_purpose_is_still_a_repeat():
    _, _, prev = flag_identical_repeats([_tc("grep_code", {"q": "x", "purpose": "find it"})], ["R"], {})
    _, rep, _ = flag_identical_repeats([_tc("grep_code", {"q": "x", "purpose": "look again"})], ["R"], prev)
    assert rep == ["grep_code"]
