"""Abstract base class for re-ranker providers."""
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel


class RerankResult(BaseModel):
    """A single result from re-ranking."""
    index: int
    score: float


class AbstractRerankerProvider(ABC):
    """Contract that every re-ranker adapter must satisfy."""

    @abstractmethod
    async def rerank(self, query: str, documents: List[str]) -> List[RerankResult]:
        """
        Re-rank documents based on relevance to the query.
        
        Args:
            query: The query string
            documents: List of document texts to re-rank
            
        Returns:
            List of RerankResult ordered by relevance (highest first)
        """
        ...