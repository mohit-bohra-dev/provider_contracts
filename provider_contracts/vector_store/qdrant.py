"""Qdrant vector store adapter."""
from __future__ import annotations

import uuid
from typing import Any

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument

_DEFAULT_URL = "http://localhost:6333"
_DEFAULT_COLLECTION = "default"


def _to_qdrant_id(doc_id: str) -> int | str:
    """Convert a string ID to a Qdrant-compatible point ID.

    Qdrant PointStruct accepts:
    - unsigned integers
    - UUID strings (canonical form with dashes)

    This function tries to parse the string as an integer first, then
    as a UUID, falling back to a deterministic UUID5.
    """
    # Already a pure digit string → int
    if doc_id.isdigit():
        return int(doc_id)
    # Try parsing as a UUID directly
    try:
        return str(uuid.UUID(doc_id))
    except ValueError:
        pass
    # Fallback: deterministic UUID from any string
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


class QdrantVectorStoreProvider(AbstractVectorStoreProvider):
    """Wraps ``qdrant_client.AsyncQdrantClient``.

    Requires the ``qdrant`` optional extra::

        pip install provider-contracts[qdrant]
    """

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        collection: str = _DEFAULT_COLLECTION,
    ) -> None:
        from qdrant_client import AsyncQdrantClient  # type: ignore[import-untyped]

        self._client: Any = AsyncQdrantClient(url=url)
        self._collection = collection

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        from qdrant_client.models import VectorParams  # type: ignore[import-untyped]
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        collection = self._resolve_collection(namespace)
        points = [
            PointStruct(
                id=_to_qdrant_id(doc.id),
                vector=doc.embedding,
                payload={**(doc.metadata), "text": doc.text or ""},
            )
            for doc in documents
        ]
        await self._client.upsert(collection_name=collection, points=points)

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
        collection = self._resolve_collection(namespace)
        query_filter = None
        if filter:
            from qdrant_client.models import (  # type: ignore[import-untyped]
                FieldCondition,
                Filter,
                MatchValue,
            )

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            query_filter = Filter(must=conditions)

        # .search() was removed in qdrant-client >= 1.14; use .query_points() instead.
        # query_points() returns a QueryResponse whose .points is a list[ScoredPoint].
        response: Any = await self._client.query_points(
            collection_name=collection,
            query=embedding,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
            query_filter=query_filter,
        )
        return [
            SearchResult(
                id=str(hit.id),
                score=float(hit.score),
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                text=hit.payload.get("text"),
            )
            for hit in response.points
        ]

    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        from qdrant_client.models import PointIdsList  # type: ignore[import-untyped]

        collection = self._resolve_collection(namespace)
        await self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[_to_qdrant_id(i) for i in ids]),
        )

    def _resolve_collection(self, namespace: str) -> str:
        if namespace == "default":
            return self._collection
        return f"{self._collection}_{namespace}"
