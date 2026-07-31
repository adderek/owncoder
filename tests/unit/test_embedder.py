"""Embedder batching, timeout scaling and circuit breaker — rag/embedder.py.

The regression these pin: one request timeout covered a whole batch, but
timeout_s is sized for a single text, so a large batch on a slow endpoint timed
out and lost every vector in it.
"""
import pytest

from agent.config.models import EmbeddingsConfig
from agent.rag import embedder as emb_mod
from agent.rag.embedder import Embedder


class _FakeEmbeddings:
    def __init__(self, owner):
        self._owner = owner

    def create(self, model, input):
        owner = self._owner
        owner.calls.append({"n": len(input), "timeout": owner.pending_timeout,
                            "input": list(input)})
        if owner.fail_when(input):
            raise RuntimeError("boom")
        data = [type("D", (), {"embedding": [float(len(t))]})() for t in input]
        return type("R", (), {"data": data})()


class _FakeClient:
    """Stands in for the OpenAI client: records batch sizes and timeouts."""

    def __init__(self, fail_when=lambda texts: False):
        self.calls: list[dict] = []
        self.fail_when = fail_when
        self.pending_timeout = None
        self.embeddings = _FakeEmbeddings(self)

    def with_options(self, timeout=None):
        self.pending_timeout = timeout
        return self


def _embedder(fail_when=lambda texts: False, **cfg_kw):
    cfg = EmbeddingsConfig(**cfg_kw)
    e = Embedder.__new__(Embedder)          # skip __init__: it builds a real client
    e._cfg = cfg
    e._client = _FakeClient(fail_when)
    e._max_chars = cfg.max_tokens * 4 if cfg.max_tokens > 0 else 0
    e.call_count = 0
    e._total_elapsed = 0.0
    e._fail_streak = 0
    e._open_until = 0.0
    return e


# ── timeout scaling ───────────────────────────────────────────────────────────

def test_small_batch_uses_base_timeout():
    e = _embedder(timeout_s=15.0)
    assert e._timeout_for(1) == 15.0
    assert e._timeout_for(emb_mod._TEXTS_PER_BASE_TIMEOUT) == 15.0


def test_timeout_scales_with_batch_size():
    e = _embedder(timeout_s=15.0)
    assert e._timeout_for(32) == 15.0 * (32 / emb_mod._TEXTS_PER_BASE_TIMEOUT)


def test_timeout_is_capped():
    e = _embedder(timeout_s=60.0)
    assert e._timeout_for(10_000) == emb_mod._MAX_TIMEOUT_S


def test_scaled_timeout_reaches_the_request():
    e = _embedder(timeout_s=15.0)
    e.embed(["a"] * 16)
    assert e._client.calls[0]["timeout"] == pytest.approx(30.0)


# ── batching ──────────────────────────────────────────────────────────────────

def test_empty_input_makes_no_request():
    e = _embedder()
    assert e.embed([]) == []
    assert e._client.calls == []


def test_successful_batch_is_one_request():
    e = _embedder()
    out = e.embed(["a", "bb", "ccc"])
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]
    assert len(e._client.calls) == 1


def test_long_text_is_truncated_before_sending():
    e = _embedder(max_tokens=2)          # 2 tokens ≈ 8 chars
    e.embed(["x" * 100])
    assert len(e._client.calls[0]["input"][0]) == 8


def test_failing_batch_splits_and_keeps_good_vectors():
    """A batch too big for the endpoint must not lose every vector in it."""
    e = _embedder(fail_when=lambda texts: len(texts) > 2)
    out = e.embed(["a", "bb", "ccc", "dddd"])
    assert len(out) == 4
    assert all(v for v in out), "halving should have recovered every vector"
    sizes = [c["n"] for c in e._client.calls]
    assert sizes == [4, 2, 2], "expected one failed batch then two halves"


def test_split_preserves_order():
    e = _embedder(fail_when=lambda texts: len(texts) > 1)
    out = e.embed(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_single_text_failure_yields_one_empty_vector():
    e = _embedder(fail_when=lambda texts: True)
    assert e.embed(["a"]) == [[]]


def test_call_count_tracks_texts_embedded():
    e = _embedder()
    e.embed(["a", "b", "c"])
    assert e.call_count == 3


# ── circuit breaker ───────────────────────────────────────────────────────────

def test_breaker_opens_after_consecutive_failures():
    e = _embedder(fail_when=lambda texts: True)
    for _ in range(emb_mod._FAIL_STREAK_OPEN):
        e.embed(["a"])
    before = len(e._client.calls)
    assert e.embed(["a"]) == [[]]
    assert len(e._client.calls) == before, "open breaker must not hit the endpoint"


def test_breaker_returns_one_empty_per_text_while_open():
    e = _embedder(fail_when=lambda texts: True)
    for _ in range(emb_mod._FAIL_STREAK_OPEN):
        e.embed(["a"])
    out = e.embed(["a", "b", "c"])
    assert out == [[], [], []]


def test_breaker_reopens_endpoint_after_cooldown(monkeypatch):
    e = _embedder(fail_when=lambda texts: True)
    for _ in range(emb_mod._FAIL_STREAK_OPEN):
        e.embed(["a"])
    open_until = e._open_until
    e._client.fail_when = lambda texts: False
    monkeypatch.setattr(emb_mod._time, "monotonic",
                        lambda: open_until + 1.0)
    assert e.embed(["a"])[0] == [1.0]


def test_success_resets_the_failure_streak():
    state = {"fail": True}
    e = _embedder(fail_when=lambda texts: state["fail"])
    e.embed(["a"])
    e.embed(["a"])
    assert e._fail_streak == 2
    state["fail"] = False
    e.embed(["a"])
    assert e._fail_streak == 0


def test_split_retries_count_toward_the_breaker():
    """Documents a sharp edge: the streak counts *requests*, not batches.

    A single 4-text batch that fails at every level trips the 3-failure breaker
    on its own — so one slow (not dead) batch can pause embeddings for the whole
    cooldown. Change this test deliberately if the counting moves to batch level.
    """
    e = _embedder(fail_when=lambda texts: True)
    out = e.embed(["a", "b", "c", "d"])
    assert out == [[], [], [], []]
    assert e._open_until > 0.0, "breaker opened from a single batch's split-retries"
