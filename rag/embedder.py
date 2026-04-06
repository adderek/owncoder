from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.config import EmbeddingsConfig


class Embedder:
    def __init__(self, cfg: "EmbeddingsConfig") -> None:
        from openai import OpenAI
        self._cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key="local")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._cfg.model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
