"""Abstract base class for embedding providers."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    """Structured result from an embedding provider."""

    vector: list[float]
    model: str
    dimensions: int


class AbstractEmbeddingProvider(ABC):
    """Contract that every embedding adapter must satisfy."""

    @abstractmethod
    async def get_dimensions(self) -> int:
        """Return the embedding dimension for the current model."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """
        Return an embedding vector for the given text.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingResult containing the vector, model name, and dimensions.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Return embedding vectors for a batch of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of EmbeddingResult in the same order as *texts*.
        """
        ...
