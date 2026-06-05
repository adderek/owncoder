"""Unit tests for ConfidenceMonitor."""
import pytest
from agent.core.confidence import ConfidenceMonitor, _result_hash


def _monitor(**kwargs) -> ConfidenceMonitor:
    defaults = dict(window=6, error_rate_threshold=0.6, null_rate_threshold=0.6,
                    dup_rate_threshold=0.5, score_threshold=0.35, inject_cooldown=0)
    defaults.update(kwargs)
    return ConfidenceMonitor(**defaults)


def test_no_results_no_trigger():
    m = _monitor()
    sig = m.should_intervene()
    assert not sig.triggered
    assert sig.score == 1.0


def test_all_success_no_trigger():
    m = _monitor()
    for i in range(6):
        m.observe_result(f'{{"ok": true, "content": "result-{i}-with-enough-chars"}}', is_error=False)
        m.tick_iter()
    sig = m.should_intervene()
    assert not sig.triggered
    assert sig.score > 0.5


def test_high_error_rate_triggers():
    m = _monitor(inject_cooldown=0)
    for _ in range(6):
        m.observe_result('{"error": "not found"}', is_error=True)
        m.tick_iter()
    sig = m.should_intervene()
    assert sig.triggered
    assert sig.error_rate > 0.9


def test_duplicate_results_trigger():
    m = _monitor(inject_cooldown=0, dup_rate_threshold=0.5)
    unique_result_a = '{"content": "abcdefgh", "path": "foo.py", "lines": 10}'
    unique_result_b = '{"content": "xyzuvwxy", "path": "bar.py", "lines": 20}'
    m.observe_result(unique_result_a, is_error=False)
    m.tick_iter()
    m.observe_result(unique_result_b, is_error=False)
    m.tick_iter()
    # Repeat same result — signals model is cycling on same content
    for _ in range(4):
        m.observe_result(unique_result_a, is_error=False)
        m.tick_iter()
    sig = m.should_intervene()
    assert sig.triggered
    assert sig.dup_rate > 0.3


def test_null_results_trigger():
    m = _monitor(inject_cooldown=0)
    for _ in range(6):
        m.observe_result("", is_error=False)
        m.tick_iter()
    sig = m.should_intervene()
    assert sig.triggered
    assert sig.null_rate > 0.9


def test_cooldown_prevents_repeated_firing():
    m = _monitor(inject_cooldown=3)
    for _ in range(6):
        m.observe_result('{"error": "x"}', is_error=True)
        m.tick_iter()
    sig1 = m.should_intervene()
    assert sig1.triggered
    m.acknowledge()
    # immediately after acknowledge, cooldown blocks
    sig2 = m.should_intervene()
    assert not sig2.triggered


def test_cooldown_expires_and_refires():
    m = _monitor(inject_cooldown=2)
    for _ in range(6):
        m.observe_result('{"error": "x"}', is_error=True)
        m.tick_iter()
    m.acknowledge()
    # advance past cooldown
    m.tick_iter()
    m.tick_iter()
    # add more bad results so window stays bad
    m.observe_result('{"error": "x"}', is_error=True)
    m.tick_iter()
    sig = m.should_intervene()
    assert sig.triggered


def test_window_eviction_allows_recovery():
    m = _monitor(window=4, inject_cooldown=0)
    # fill window with errors
    for _ in range(4):
        m.observe_result('{"error": "x"}', is_error=True)
        m.tick_iter()
    assert m.should_intervene().triggered
    # replace with successes — window should clear
    for i in range(4):
        m.observe_result(f'{{"ok": true, "data": "unique-{i}-payload-content"}}', is_error=False)
        m.tick_iter()
    m.acknowledge()
    sig = m.should_intervene()
    assert not sig.triggered


def test_needs_half_window_before_triggering():
    m = _monitor(window=6, inject_cooldown=0)
    # Only 2 results (< window//2=3) — should not trigger even if all errors
    m.observe_result('{"error": "x"}', is_error=True)
    m.tick_iter()
    m.observe_result('{"error": "x"}', is_error=True)
    m.tick_iter()
    assert not m.should_intervene().triggered


def test_intervention_message_includes_rates():
    m = _monitor(inject_cooldown=0)
    for _ in range(6):
        m.observe_result('{"error": "not found"}', is_error=True)
        m.tick_iter()
    sig = m.should_intervene()
    msg = ConfidenceMonitor.intervention_message(sig)
    assert "confidence-guard" in msg
    assert "error rate" in msg


def test_result_hash_deduplicates_identical_content():
    h1 = _result_hash("hello")
    h2 = _result_hash("hello")
    h3 = _result_hash("world")
    assert h1 == h2
    assert h1 != h3
