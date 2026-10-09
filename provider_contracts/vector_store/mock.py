"""In-memory mock vector store — brute-force cosine similarity, for dev/tests."""
from __future__ import annotations

import math
from typing import Any, ClassVar

from provider_contracts.errors import ProviderConfigError

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument
from ._filter import metadata_matches_filter


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MockVectorStoreProvider(AbstractVectorStoreProvider):
    """Stores embeddings in a plain dict — zero external dependencies."""

    supports_any_of_filter: ClassVar[bool] = True

    def __init__(self) -> None:
        # namespace → {id: VectorDocument}
        self._store: dict[str, dict[str, VectorDocument]] = {}
        self._sizes: dict[str, int] = {}

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        _ = distance
        await self.ensure_namespace(name, vector_size)

    async def ensure_namespace(self, namespace: str, vector_size: int) -> None:
        if namespace in self._sizes:
            if self._sizes[namespace] != vector_size:
                raise ProviderConfigError(
                    f"Namespace {namespace!r} exists with vector_size "
                    f"{self._sizes[namespace]}, requested {vector_size}"
                )
            self._store.setdefault(namespace, {})
            return

        existing_docs = self._store.get(namespace) or {}
        if existing_docs:
            inferred = len(next(iter(existing_docs.values())).embedding)
            if inferred != vector_size:
                raise ProviderConfigError(
                    f"Namespace {namespace!r} exists with vector_size "
                    f"{inferred}, requested {vector_size}"
                )
            self._sizes[namespace] = inferred
            return

        self._store.setdefault(namespace, {})
        self._sizes[namespace] = vector_size

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        ns = self._store.setdefault(namespace, {})
        for doc in documents:
            if namespace in self._sizes and len(doc.embedding) != self._sizes[namespace]:
                raise ProviderConfigError(
                    f"Document {doc.id!r} embedding size {len(doc.embedding)} "
                    f"does not match namespace {namespace!r} size "
                    f"{self._sizes[namespace]}"
                )
            ns[doc.id] = doc

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
        ns = self._store.get(namespace, {})
        scored = [
            (doc_id, _cosine(embedding, doc.embedding), doc)
            for doc_id, doc in ns.items()
            if metadata_matches_filter(doc.metadata, filter)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[SearchResult] = []
        for doc_id, score, doc in scored:
            if score < min_score:
                break
            results.append(
                SearchResult(id=doc_id, score=score, metadata=doc.metadata, text=doc.text)
            )
            if len(results) >= top_k:
                break
        return results

    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        ns = self._store.get(namespace, {})
        for doc_id in ids:
            ns.pop(doc_id, None)
