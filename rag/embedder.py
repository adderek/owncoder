from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import EmbeddingsConfig


class Embedder:
    def __init__(self, cfg: "EmbeddingsConfig") -> None:
        from openai import OpenAI
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key="local")
        # Approximate char limit derived from token limit (4 chars ≈ 1 token).
        self._max_chars = cfg.max_tokens * 4 if cfg.max_tokens > 0 else 0

    def _truncate(self, text: str) -> str:
        if self._max_chars and len(text) > self._max_chars:
            return text[:self._max_chars]
        return text

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        truncated = [self._truncate(t) for t in texts]
        response = self._client.embeddings.create(
            model=self._cfg.model,
            input=truncated,
        )
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
