"""Abstract base class for vector store providers."""
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class VectorDocument(BaseModel):
    """A document with its embedding and metadata."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = {}
    text: str | None = None


class SearchResult(BaseModel):
    """A single result from a similarity search."""

    id: str
    score: float
    """Cosine similarity score in range [0, 1] (higher = more similar)."""

    metadata: dict[str, Any] = {}
    text: str | None = None


class AbstractVectorStoreProvider(ABC):
    """Contract that every vector store adapter must satisfy.

    Filter semantics for ``search(..., filter=...)``:
    - scalar value → equality match on that metadata key
    - ``list`` value → any-of (document matches if its value is in the list)

    Providers that implement list any-of must set
    ``supports_any_of_filter = True`` so callers can advertise the capability.
    """

    supports_any_of_filter: ClassVar[bool] = False

    @abstractmethod
    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        """Create a new collection/namespace with the specified configuration."""
        ...

    @abstractmethod
    async def ensure_namespace(self, namespace: str, vector_size: int) -> None:
        """Idempotently ensure the physical collection for ``namespace`` exists.

        Resolves ``namespace`` to a physical collection with the **same**
        mapping used by ``upsert`` / ``search`` / ``delete``. No-op when the
        collection already exists with ``vector_size``. Raises
        ``ProviderConfigError`` when it exists with a different size.
        """
        ...

    @abstractmethod
    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        """
        Insert or update documents in the vector store.

        Args:
            documents: List of documents with embeddings.
            namespace: Logical partition (collection / schema) to write into.
        """
        ...

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        query_text: str | None = None,
        *,
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most similar documents.

        Args:
            embedding: Query embedding vector.
            query_text: Optional query text for hybrid search.
            namespace: Partition to search within.
            top_k: Maximum number of results.
            min_score: Minimum cosine similarity threshold.
            filter: Optional metadata filter. Scalar values match with
                equality; list values match with any-of when the provider
                sets ``supports_any_of_filter``.
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        """Remove documents by ID from the given namespace."""
        ...
