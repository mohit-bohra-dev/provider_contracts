"""In-memory mock re-ranker — identity order with descending fake scores."""

from __future__ import annotations

from ._base import AbstractRerankerProvider, RerankResult


class MockRerankerProvider(AbstractRerankerProvider):
    """Deterministic re-ranker for unit tests (preserves input order)."""

    async def rerank(self, query: str, documents: list[str]) -> list[RerankResult]:
        _ = query
        n = len(documents)
        return [RerankResult(index=i, score=float(n - i)) for i in range(n)]
