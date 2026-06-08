"""Sentence-transformers embedding adapter (local CPU/GPU inference)."""
from __future__ import annotations

from typing import Any

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class SentenceTransformerEmbeddingProvider(AbstractEmbeddingProvider):
    """Wraps ``sentence-transformers`` for local embedding generation.

    Requires the ``bge`` optional extra::

        pip install provider-contracts[bge]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        # Lazy import so the package stays importable without the extra.
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

        self._model: Any = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dimensions: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    async def get_dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        vector: list[float] = self._model.encode(text).tolist()  # type: ignore[union-attr]
        return EmbeddingResult(
            vector=vector,
            model=self._model_name,
            dimensions=self._dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        vectors: list[list[float]] = self._model.encode(texts).tolist()  # type: ignore[union-attr]
        return [
            EmbeddingResult(
                vector=v,
                model=self._model_name,
                dimensions=self._dimensions,
            )
            for v in vectors
        ]
