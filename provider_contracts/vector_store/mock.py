"""In-memory mock vector store — brute-force cosine similarity, for dev/tests."""
import math
from typing import Any

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class MockVectorStoreProvider(AbstractVectorStoreProvider):
    """Stores embeddings in a plain dict — zero external dependencies."""

    def __init__(self) -> None:
        # namespace → {id: VectorDocument}
        self._store: dict[str, dict[str, VectorDocument]] = {}

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        _ = (vector_size, distance)
        self._store.setdefault(name, {})

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        ns = self._store.setdefault(namespace, {})
        for doc in documents:
            ns[doc.id] = doc

    async def search(
        self,
        embedding: list[float],
        *,
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        ns = self._store.get(namespace, {})
        scored = [
            (doc_id, _cosine(embedding, doc.embedding), doc)
            for doc_id, doc in ns.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score, doc in scored[:top_k]:
            if score < min_score:
                break
            results.append(
                SearchResult(id=doc_id, score=score, metadata=doc.metadata, text=doc.text)
            )
        return results

    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        ns = self._store.get(namespace, {})
        for doc_id in ids:
            ns.pop(doc_id, None)
