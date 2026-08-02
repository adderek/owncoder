"""Startup asks each endpoint once, and asks while the user is answering.

`agent chat` took ~59s on a machine with one LAN box switched off. Almost none
of that was work waiting on the user: the same endpoints were probed from three
places, the two pool walks probed per *entry* rather than per URL (eight entries
on one dead host meant eight three-second waits for a fact already established),
and the ctx-window enrichment ran inline between the last prompt and the UI.
"""
from __future__ import annotations

import threading
import time

import pytest

from agent.config import probe_cache


@pytest.fixture(autouse=True)
def _clean():
    probe_cache.invalidate()
    yield
    probe_cache.invalidate()


def _counting_fetch(result=None, delay=0.0):
    calls: list[str] = []

    def fetch(url, api_key, timeout):
        calls.append(url)
        if delay:
            time.sleep(delay)
        return result

    return fetch, calls


class TestTheAnswerIsKept:
    def test_a_second_ask_does_not_reach_the_server(self):
        fetch, calls = _counting_fetch({"data": []})
        for _ in range(3):
            probe_cache.get("http://x/v1", "k", 3, fetch)
        assert calls == ["http://x/v1"]

    def test_an_unreachable_endpoint_is_remembered_too(self):
        """This is the expensive answer: it cost a full timeout to learn."""
        fetch, calls = _counting_fetch(None)
        assert probe_cache.get("http://dead/v1", "", 3, fetch) is None
        assert probe_cache.get("http://dead/v1", "", 3, fetch) is None
        assert len(calls) == 1

    def test_a_miss_is_kept_for_less_time_than_a_hit(self):
        """A server that did not answer may be one being started right now."""
        assert probe_cache._MISS_TTL < probe_cache._HIT_TTL

    def test_the_url_is_the_key_not_the_entry(self):
        """Sixteen pool entries share five URLs; the host answers the same
        whichever entry name asked."""
        fetch, calls = _counting_fetch({"data": []})
        probe_cache.get("http://x/v1", "k", 3, fetch)
        probe_cache.get("http://x/v1/", "other-key", 2, fetch)
        assert len(calls) == 1

    def test_an_explicit_retry_is_honoured(self):
        fetch, calls = _counting_fetch(None)
        probe_cache.get("http://x/v1", "", 3, fetch)
        probe_cache.get("http://x/v1", "", 3, fetch, force=True)
        assert len(calls) == 2

    def test_a_stale_answer_is_dropped(self):
        fetch, calls = _counting_fetch({"data": []})
        probe_cache.get("http://x/v1", "", 3, fetch)
        # Expire it rather than wait a minute for the TTL.
        with probe_cache._lock:
            expiry, result = probe_cache._entries["http://x/v1"]
            probe_cache._entries["http://x/v1"] = (time.monotonic() - 1, result)
        probe_cache.get("http://x/v1", "", 3, fetch)
        assert len(calls) == 2


class TestPrefetch:
    def test_distinct_urls_are_probed_at_once(self):
        """Serially this is one timeout per dead host; the point is to spend
        them together, while the profile report is being read."""
        fetch, calls = _counting_fetch(None, delay=0.2)
        pairs = [(f"http://h{i}/v1", "") for i in range(6)]
        t0 = time.monotonic()
        probe_cache.prefetch(pairs, 3, fetch)
        elapsed = time.monotonic() - t0
        assert len(calls) == 6
        assert elapsed < 0.2 * 6 / 2

    def test_repeated_entries_collapse_to_one_probe(self):
        fetch, calls = _counting_fetch(None)
        probe_cache.prefetch([("http://same/v1", "")] * 8, 3, fetch)
        assert len(calls) == 1

    def test_it_skips_what_is_already_known(self):
        fetch, calls = _counting_fetch({"data": []})
        probe_cache.get("http://x/v1", "", 3, fetch)
        probe_cache.prefetch([("http://x/v1", "")], 3, fetch)
        assert len(calls) == 1

    def test_one_failing_probe_does_not_sink_the_rest(self):
        seen = []

        def fetch(url, api_key, timeout):
            seen.append(url)
            if "boom" in url:
                raise RuntimeError("probe exploded")
            return None

        probe_cache.prefetch([("http://boom/v1", ""), ("http://ok/v1", "")], 3, fetch)
        assert sorted(seen) == ["http://boom/v1", "http://ok/v1"]


class TestTheProbesRunUnderThePrompts:
    def test_the_warmup_starts_before_the_profile_check(self):
        """The answers do not depend on the profile — only the choice made from
        them does — so the probing belongs before the question, not after it."""
        import importlib
        import inspect
        cli_main = importlib.import_module("agent.cli.main")
        src = inspect.getsource(cli_main.main)
        assert src.index("warmup.start(config)") < src.index("run_startup_profile_check(config")
        assert src.index("run_startup_profile_check(config") < src.index("check_reachability(config)")

    def test_the_warmup_is_collected_before_the_agent_is_built(self):
        import inspect
        from agent.cli.chat import cmd_chat
        src = inspect.getsource(cmd_chat)
        assert "warmup.join()" in src

    def test_it_says_nothing_on_the_way(self):
        """A background thread printing over a prompt makes both unreadable."""
        import inspect
        from agent.cli import warmup
        src = inspect.getsource(warmup)
        assert "print(" not in src

    def test_nor_does_the_enrichment_it_runs_beside(self):
        """model_probe used to print its config-vs-server mismatches. Once it
        moved to a background thread those landed mid-prompt:

            [r]ecover / [i]gnore / [d]elete / [s]kip? [model-probe] openrouter…
        """
        import inspect
        from agent.config import model_probe
        assert "print(" not in inspect.getsource(model_probe)

    def test_a_broken_warmup_is_not_a_broken_startup(self, monkeypatch):
        from agent.cli import warmup
        monkeypatch.setattr(warmup, "_warm", lambda cfg: (_ for _ in ()).throw(RuntimeError("nope")))
        monkeypatch.setattr(warmup, "_thread", None)
        warmup.start(object())
        warmup.join(5)          # must return, not raise


class TestEnrichmentIsOffTheCriticalPath:
    def test_startup_does_not_wait_for_it(self):
        import inspect
        from agent.config.loader import check_reachability
        src = inspect.getsource(check_reachability)
        assert "start_enrichment(config)" in src
        assert "enrich_model_entries(config)" not in src

    def test_the_first_turn_does(self):
        """It refines the ctx window this turn is budgeted against."""
        import inspect
        from agent.core.agent import Agent
        src = inspect.getsource(Agent.chat)
        assert "join_enrichment()" in src
        assert "turn_id == 1" in src

    def test_a_hung_probe_does_not_hold_the_turn(self):
        from agent.config import model_probe
        slow = threading.Event()
        t = threading.Thread(target=slow.wait, daemon=True)
        t.start()
        model_probe._startup_enrichment = t
        try:
            t0 = time.monotonic()
            model_probe.join_enrichment(timeout=0.2)
            assert time.monotonic() - t0 < 2
        finally:
            slow.set()
            model_probe._startup_enrichment = None


class TestNoProbeBypassesTheCache:
    def test_every_endpoint_walk_goes_through_the_prober(self):
        """Three copies of the same GET existed; two of them had no cache and
        one of those cost thirteen serial probes on every start."""
        import inspect
        from agent.config import loader, model_probe
        for src in (inspect.getsource(loader._resolve_role_pools),
                    inspect.getsource(model_probe._probe_endpoint)):
            assert "urlopen" not in src
            assert "_probe_models" in src
