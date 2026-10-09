"""Qdrant vector store adapter."""
from __future__ import annotations

import uuid
from typing import Any, ClassVar

from provider_contracts.errors import ProviderConfigError

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument
from ._filter import validate_filter_key

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


def _vector_size_from_collection_info(info: Any) -> int | None:
    """Extract embedding dimension from a Qdrant collection info object."""
    vectors = info.config.params.vectors
    if vectors is None:
        return None
    # Named vectors: dict[str, VectorParams]; unnamed: VectorParams
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        size = getattr(first, "size", None)
        if isinstance(size, int):
            return size
    return None


class QdrantVectorStoreProvider(AbstractVectorStoreProvider):
    """Wraps ``qdrant_client.AsyncQdrantClient``.

    Requires the ``qdrant`` optional extra::

        pip install provider-contracts[qdrant]
    """

    supports_any_of_filter: ClassVar[bool] = True

    def __init__(
        self,
        url: str = _DEFAULT_URL,
        collection: str = _DEFAULT_COLLECTION,
    ) -> None:
        from qdrant_client import AsyncQdrantClient

        self._client: Any = AsyncQdrantClient(url=url)
        self._collection = collection

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        dist = distance_map.get(distance, Distance.COSINE)
        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=dist),
        )

    async def ensure_namespace(self, namespace: str, vector_size: int) -> None:
        collection = self._resolve_collection(namespace)
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if collection in names:
            info = await self._client.get_collection(collection_name=collection)
            current = _vector_size_from_collection_info(info)
            if current is not None and current != vector_size:
                raise ProviderConfigError(
                    f"Collection {collection!r} exists with vector_size "
                    f"{current}, requested {vector_size}"
                )
            return
        await self.create_collection(collection, vector_size)

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        from qdrant_client.models import PointStruct

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
        _ = query_text
        collection = self._resolve_collection(namespace)
        query_filter = None
        if filter:
            from qdrant_client.models import (
                FieldCondition,
                Filter,
                MatchAny,
                MatchValue,
            )

            conditions: list[Any] = []
            for key, value in filter.items():
                validate_filter_key(key)
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=list(value)))
                    )
                else:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
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
        from qdrant_client.models import PointIdsList

        collection = self._resolve_collection(namespace)
        await self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[_to_qdrant_id(i) for i in ids]),
        )

    def _resolve_collection(self, namespace: str) -> str:
        if namespace == "default":
            return self._collection
        return f"{self._collection}_{namespace}"
