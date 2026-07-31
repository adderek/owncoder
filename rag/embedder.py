from __future__ import annotations

import re as _re
import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import EmbeddingsConfig


_TEXTS_PER_BASE_TIMEOUT = 8   # texts covered by one `timeout_s` before scaling
_MAX_TIMEOUT_S = 120.0        # ceiling for a scaled batch request
_FAIL_STREAK_OPEN = 3  # consecutive failed requests before the breaker opens
_COOLDOWN_S = 60.0       # how long embeddings stay disabled once it opens


def _port_label(base_url: str, model: str) -> str:
    m = _re.search(r":(\d+)", base_url)
    port = f":{m.group(1)}" if m else base_url
    short = model.split("/")[-1][:16]
    return f"{short}@{port}"


class Embedder:
    def __init__(self, cfg: "EmbeddingsConfig") -> None:
        from openai import OpenAI
        self._cfg = cfg
        # Explicit timeout + single retry: the SDK defaults (600s, 2 retries)
        # let one hung endpoint freeze callers for minutes.
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key="local",
            timeout=getattr(cfg, "timeout_s", 15.0) or 15.0,
            max_retries=1,
        )
        # Approximate char limit derived from token limit (4 chars ≈ 1 token).
        self._max_chars = cfg.max_tokens * 4 if cfg.max_tokens > 0 else 0
        self.call_count: int = 0
        self._total_elapsed: float = 0.0
        # Circuit breaker: a dead endpoint must not turn every batch into a
        # cascade of split-retries, each paying the full timeout.
        self._fail_streak: int = 0
        self._open_until: float = 0.0

    @property
    def endpoint(self) -> str:
        return _port_label(str(self._client.base_url), self._cfg.model)

    @property
    def rate(self) -> float:
        return self.call_count / self._total_elapsed if self._total_elapsed > 0 else 0.0

    def _timeout_for(self, n: int) -> float:
        """Scale the per-request timeout with batch size.

        ``timeout_s`` is sized for a single text; a 32-text batch is ~32x the
        work on the same connection and was tripping the flat timeout.
        """
        base = float(getattr(self._cfg, "timeout_s", 15.0) or 15.0)
        return min(_MAX_TIMEOUT_S, base * max(1.0, n / _TEXTS_PER_BASE_TIMEOUT))

    def _truncate(self, text: str) -> str:
        if self._max_chars and len(text) > self._max_chars:
            return text[:self._max_chars]
        return text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        truncated = [self._truncate(t) for t in texts]
        return self._embed_batch(truncated)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One API call, splitting the batch in half on failure.

        A whole batch shares one request timeout, so a big batch on a slow
        endpoint times out and loses every vector in it. Halving on error
        retries the same work in cheaper requests; only a single text that
        still fails is given up on (empty vector).
        """
        import logging
        from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
        if _time.monotonic() < self._open_until:
            return [[] for _ in texts]
        _ms_inc("emb")
        t0 = _time.monotonic()
        try:
            response = self._client.with_options(
                timeout=self._timeout_for(len(texts)),
            ).embeddings.create(
                model=self._cfg.model,
                input=texts,
            )
            self.call_count += len(texts)
            self._total_elapsed += _time.monotonic() - t0
            self._fail_streak = 0
            return [item.embedding for item in response.data]
        except Exception as exc:
            log = logging.getLogger(__name__)
            self._fail_streak += 1
            if self._fail_streak >= _FAIL_STREAK_OPEN:
                self._open_until = _time.monotonic() + _COOLDOWN_S
                log.warning(
                    "embedder: %d consecutive failures (%s) — pausing embeddings for %ds",
                    self._fail_streak, exc, int(_COOLDOWN_S),
                )
                return [[] for _ in texts]
            if len(texts) > 1:
                mid = len(texts) // 2
                log.debug("embedder: batch of %d failed (%s) — splitting", len(texts), exc)
                return self._embed_batch(texts[:mid]) + self._embed_batch(texts[mid:])
            log.warning("embedder: API error: %s", exc)
            return [[]]
        finally:
            _ms_dec("emb")

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
