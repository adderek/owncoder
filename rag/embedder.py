from __future__ import annotations

import re as _re
import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import EmbeddingsConfig


def _port_label(base_url: str, model: str) -> str:
    m = _re.search(r":(\d+)", base_url)
    port = f":{m.group(1)}" if m else base_url
    short = model.split("/")[-1][:16]
    return f"{short}@{port}"


class Embedder:
    def __init__(self, cfg: "EmbeddingsConfig") -> None:
        from openai import OpenAI
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key="local")
        # Approximate char limit derived from token limit (4 chars ≈ 1 token).
        self._max_chars = cfg.max_tokens * 4 if cfg.max_tokens > 0 else 0
        self.call_count: int = 0
        self._total_elapsed: float = 0.0

    @property
    def endpoint(self) -> str:
        return _port_label(str(self._client.base_url), self._cfg.model)

    @property
    def rate(self) -> float:
        return self.call_count / self._total_elapsed if self._total_elapsed > 0 else 0.0

    def _truncate(self, text: str) -> str:
        if self._max_chars and len(text) > self._max_chars:
            return text[:self._max_chars]
        return text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import logging
        from agent.core.model_status import _inc as _ms_inc, _dec as _ms_dec
        truncated = [self._truncate(t) for t in texts]
        _ms_inc("emb")
        t0 = _time.monotonic()
        try:
            response = self._client.embeddings.create(
                model=self._cfg.model,
                input=truncated,
            )
            self.call_count += len(texts)
            self._total_elapsed += _time.monotonic() - t0
            return [item.embedding for item in response.data]
        except Exception as exc:
            logging.getLogger(__name__).warning("embedder: API error: %s", exc)
            return [[] for _ in texts]
        finally:
            _ms_dec("emb")

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
