"""In-memory mock embedding provider — deterministic, for dev/tests."""
import hashlib
import random

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_DEFAULT_DIMS = 384


class MockEmbeddingProvider(AbstractEmbeddingProvider):
    """Deterministic mock — returns pseudo-random vectors seeded by text hash."""

    def __init__(self, dimensions: int = _DEFAULT_DIMS) -> None:
        self._dimensions = dimensions

    async def get_dimensions(self) -> int:
        return self._dimensions

    def _embed_text(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self._dimensions)]

    async def embed(self, text: str) -> EmbeddingResult:
        vector = self._embed_text(text)
        return EmbeddingResult(
            vector=vector,
            model="mock",
            dimensions=self._dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [await self.embed(t) for t in texts]
