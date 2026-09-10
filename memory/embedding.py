"""
Deterministic, offline embedding function for vector memory stores.
Requires zero external network requests and prevents SSL/HuggingFace hangs.
"""

from __future__ import annotations

from typing import Any, Union


class DeterministicEmbeddingFunction:
    """Offline embedding function generating deterministic dense vectors from text."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def name(self) -> str:
        return "deterministic_embedding_v1"

    def is_legacy(self) -> bool:
        return True

    def embed_query(self, input: Any) -> list[list[float]]:
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def __call__(self, input: Union[str, list[str]]) -> list[list[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            vec = [0.0] * self.dim
            words = str(text).lower().split()
            for i, w in enumerate(words):
                h = abs(hash(w)) % self.dim
                vec[h] += 1.0 / (1.0 + i * 0.05)
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings
