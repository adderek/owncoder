"""Attempt ledger + shell near-duplicate loop detection (weak-model harness).

Families below are reformatted reruns taken from a weak-model session that
looped on them 49x / 21x without the exact-args loop guard ever tripping.
"""
import json
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core import attempt_ledger, tool_hints
from agent.core.loop_detector import LoopDetector
from agent.core.turn_guards import flag_identical_repeats, observe_tool_calls
from agent.memory import compactor

UNSHARE_FAMILY = [
    ["python3", "-c", "import ctypes; print(ctypes.CDLL(None).syscall(312, 0, 0))"],
    ["python3", "-c", 'import ctypes; print(ctypes.CDLL(None).syscall(312, 1, 0))'],
    ["python3", "-c", "import ctypes;  print(ctypes.CDLL(None).syscall(312,0,0))"],
    ["python3", "-c", "import ctypes; print(ctypes.CDLL(None).syscall(0x138, 0, 0))"],
    ["python3", "-c", "import ctypes; print(ctypes.CDLL(None).syscall(312, 2, 0))"],
]
SED_FAMILY = [
    ["sed", "-n", "1,80p", "agent/security/runner.py"],
    ["sh", "-c", "sed -n '1,80p' agent/security/runner.py"],
    ["sh", "-c", 'sed -n "1,120p"   agent/security/runner.py'],
]


def _tc(name, args, i=0):
    return SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture(autouse=True)
def _fresh():
    tool_hints.reset_tool_hints()
    yield
    tool_hints.reset_tool_hints()


@pytest.mark.parametrize("family", [UNSHARE_FAMILY, SED_FAMILY])
def test_reformatted_reruns_share_one_action_key(family):
    keys = {attempt_ledger.action_key("run_argv", {"argv": a}) for a in family}
    assert len(keys) == 1


def test_different_commands_differ():
    a = attempt_ledger.action_key("run_argv", {"argv": ["cat", "a.py"]})
    b = attempt_ledger.action_key("run_argv", {"argv": ["cat", "b.py"]})
    assert a != b


def test_non_shell_tools_keep_exact_args_but_ignore_purpose():
    a = attempt_ledger.action_key("read_file", {"path": "x.py", "start_line": 1})
    b = attempt_ledger.action_key("read_file", {"path": "x.py", "start_line": 2})
    c = attempt_ledger.action_key("read_file", {"path": "x.py", "start_line": 1, "purpose": "why"})
    assert a != b and a == c
    assert LoopDetector.near_signature("read_file", "{}") is None


def _detector(**kw):
    return LoopDetector(window=10, threshold=3, near_window=40, near_threshold=5, **kw)


EPERM = {"stdout": "", "stderr": "Operation not permitted", "returncode": 1}


def _run(d, name, args, i, result=EPERM):
    """observe (pre-execution) then record (post-execution), like the turn loop."""
    hits = observe_tool_calls(d, [_tc(name, args, i)])
    attempt_ledger.record(name, args, result)
    return hits


def test_detector_trips_on_reformatted_family_within_threshold():
    d = _detector()
    tripped_at = None
    for i in range(49):
        argv = UNSHARE_FAMILY[i % len(UNSHARE_FAMILY)]
        hits = _run(d, "run_argv", {"argv": argv}, i)
        if hits:
            tripped_at = i + 1
            assert "~" in hits[0][1]
            break
    assert tripped_at == 5


def test_near_dups_interleaved_with_other_calls_still_trip():
    # ~1 call per round with other work between: outside the exact window of 10.
    d = _detector()
    hits = []
    for i in range(30):
        if i % 6 == 0:
            hits += _run(d, "run_argv", {"argv": SED_FAMILY[(i // 6) % 3]}, i)
        else:
            hits += _run(d, "read_file", {"path": f"f{i}.py"}, i, {"content": "x"})
    assert any("~" in sig for _, sig, _, _ in hits)


def test_paging_with_changing_output_does_not_trip_or_hint():
    d = _detector()
    for i in range(12):
        argv = ["sed", "-n", f"{i * 80 + 1},{i * 80 + 80}p", "big.py"]
        out = {"stdout": f"line {i * 80 + 1}: page {i} body\n", "returncode": 0}
        assert _run(d, "run_argv", {"argv": argv}, i, out) == []
    entry = attempt_ledger.get("run_argv", {"argv": ["sed", "-n", "1,80p", "big.py"]})
    assert entry.count == 12 and tool_hints.attempt_hint(entry) == ""


def test_same_first_line_different_output_is_not_a_repeat():
    # Seeded test runs: identical header line, different body → distinct checksums.
    d = _detector()
    for i in range(10):
        out = {"stdout": f"=== test session starts ===\nseed={i} passed", "returncode": 0,
               "duration_ms": i}
        assert _run(d, "run_argv", {"argv": ["pytest", "-p", f"seed={i}"]}, i, out) == []
    entry = attempt_ledger.get("run_argv", {"argv": ["pytest", "-p", "seed=0"]})
    assert len(entry.outcomes) == 10 and not attempt_ledger.outcome_repeated(entry)


def test_checksum_ignores_timing_and_hints():
    a = {"stdout": "x", "returncode": 0, "duration_ms": 3}
    b = {"stdout": "x", "returncode": 0, "duration_ms": 9, "_hints": ["h"]}
    assert attempt_ledger.outcome_checksum(a) == attempt_ledger.outcome_checksum(json.dumps(b))
    assert attempt_ledger.outcome_checksum(a) != attempt_ledger.outcome_checksum({"stdout": "y", "returncode": 0})


def test_near_detection_disabled_by_zero_threshold():
    d = LoopDetector(window=10, threshold=99, near_threshold=0)
    for i, argv in enumerate(UNSHARE_FAMILY * 3):
        assert observe_tool_calls(d, [_tc("run_argv", {"argv": argv}, i)]) == []


def test_exact_repeat_not_double_reported():
    d = _detector()
    hits = []
    for i in range(5):
        hits = observe_tool_calls(d, [_tc("run_argv", {"argv": ["ls"]}, i)])
        if hits:
            break
    assert len(hits) == 1 and "~" not in hits[0][1]


def test_identical_repeat_matches_respelled_shell_call_ignoring_timing():
    r1 = json.dumps({"stdout": "", "stderr": "EPERM", "returncode": 1, "duration_ms": 12})
    r2 = json.dumps({"stdout": "", "stderr": "EPERM", "returncode": 1, "duration_ms": 15,
                     "_hints": ["[tool-hint] Already tried ×3"]})
    _, rep, prev = flag_identical_repeats([_tc("run_argv", {"argv": UNSHARE_FAMILY[0]})], [r1], {})
    assert rep == []
    out, rep, _ = flag_identical_repeats([_tc("run_argv", {"argv": UNSHARE_FAMILY[2]})], [r2], prev)
    assert rep == ["run_argv"] and "IdenticalRepeat" in out[0]


def test_identical_repeat_lets_changed_output_through():
    r1 = json.dumps({"stdout": "building", "returncode": 0, "duration_ms": 1})
    r2 = json.dumps({"stdout": "done", "returncode": 0, "duration_ms": 1})
    tc = _tc("run_argv", {"argv": ["make"]})
    _, _, prev = flag_identical_repeats([tc], [r1], {})
    _, rep, _ = flag_identical_repeats([tc], [r2], prev)
    assert rep == []


def test_ledger_counts_outcomes_and_hints_from_third_call():
    res = {"stdout": "", "stderr": "unshare: Operation not permitted\n", "returncode": 1}
    hints = []
    for argv in UNSHARE_FAMILY[:3]:
        entry = attempt_ledger.record("run_argv", {"argv": argv}, res, purpose="get new userns")
        hints.append(tool_hints.attempt_hint(entry))
    assert hints[:2] == ["", ""]
    assert "×3" in hints[2] and "Operation not permitted" in hints[2]
    assert "goal: get new userns" in hints[2]
    rendered = attempt_ledger.render()
    assert rendered.startswith("- run_argv") and "rc=1" in rendered


def test_render_skips_single_attempts():
    attempt_ledger.record("run_argv", {"argv": ["ls"]}, {"returncode": 0, "stdout": "a"})
    assert attempt_ledger.render() == ""


def test_shell_nav_hint_fires_first_and_every_tenth():
    ok = {"stdout": "x", "returncode": 0}
    fired = [bool(tool_hints.tool_hints("run_argv", {"argv": ["cat", f"f{i}"]}, ok)) for i in range(12)]
    assert fired[0] and fired[10] and sum(fired) == 2
    assert tool_hints.tool_hints("run_argv", {"argv": ["pytest", "-q"]}, ok) == []
    assert tool_hints._shell_nav_command({"argv": ["sh", "-c", "V=1 grep -rn x ."]}) == "grep"


def _conv(n):
    out = []
    for i in range(n):
        out.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"t{i}", "type": "function", "function": {"name": "run_argv", "arguments": "{}"}}]})
        out.append({"role": "tool", "tool_call_id": f"t{i}", "content": "ok"})
    return out


def test_tail_grows_with_fraction_and_respects_message_cap():
    cfg = Config()
    cfg.llm.ctx_window = 65536
    conv = _conv(100)
    fixed = len(conv) - 8
    cfg.llm.compaction_tail_fraction = 0.0
    assert compactor._tail_start(conv, fixed, cfg) == fixed
    cfg.llm.compaction_tail_fraction = 0.25
    start = compactor._tail_start(conv, fixed, cfg)
    from agent.core.context_budget import compaction_message_cap
    assert start < fixed
    assert len(conv) - start <= max(8, compaction_message_cap(cfg) // 3)
    assert start >= 1


def test_compaction_section_lists_repeats():
    for _ in range(2):
        attempt_ledger.record("run_argv", {"argv": ["cat", "/proc/1/mountinfo"]},
                              {"returncode": 1, "stderr": "Permission denied"})
    section = compactor._attempt_ledger(Config())
    assert "[ALREADY TRIED" in section and "Permission denied" in section


def test_purpose_scaffold_auto_needs_reasoning_free_rounds():
    from agent.core.turn_setup import select_tools
    cfg = Config()
    cfg.tool_compaction.enabled = False
    cfg.loop_guard.purpose_scaffold = "auto"
    cfg.llm.model = "weak"
    schema = [{"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

    def has_purpose():
        tools, _, _ = select_tools(schema, cfg)
        return "purpose" in tools[0]["function"]["parameters"]["properties"]

    assert not has_purpose()
    for _ in range(attempt_ledger.SILENT_ROUNDS):
        attempt_ledger.note_round("weak", 0)
    assert has_purpose()
    assert "purpose" not in schema[0]["function"]["parameters"]["properties"]  # not mutated
    attempt_ledger.note_round("weak", 120)          # reasoning seen → off again
    assert not has_purpose()
    cfg.loop_guard.purpose_scaffold = "off"
    attempt_ledger.reset()
    for _ in range(5):
        attempt_ledger.note_round("weak", 0)
    assert not has_purpose()
