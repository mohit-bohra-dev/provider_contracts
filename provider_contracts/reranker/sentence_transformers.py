"""Sentence Transformers re-ranker implementation."""
import asyncio
from typing import List

from sentence_transformers import CrossEncoder  # type: ignore

from ._base import AbstractRerankerProvider, RerankResult


class SentenceTransformerRerankerProvider(AbstractRerankerProvider):
    """Cross-encoder re-ranker using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model = CrossEncoder(model_name)
        self._model_name = model_name

    async def rerank(self, query: str, documents: List[str]) -> List[RerankResult]:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Query text
            documents: Documents to re-rank
            
        Returns:
            List of re-ranked results sorted by score (highest first)
        """
        # Cross-encoder expects pairs of (query, document)
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder (run in executor as it is synchronous and CPU bound)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, self._model.predict, pairs)
        
        # Create results with indices and scores
        results = [
            RerankResult(index=i, score=float(score)) 
            for i, score in enumerate(scores)
        ]
        
        # Sort by score descending (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results