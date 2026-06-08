"""pgvector adapter for the vector store provider using asyncpg + SQLAlchemy async."""
import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument


class PgVectorProvider(AbstractVectorStoreProvider):
    """
    Vector store backed by PostgreSQL with the pgvector extension.

    Expects a table per namespace with the schema::

        CREATE TABLE <namespace> (
            id TEXT PRIMARY KEY,
            embedding vector(<dim>),
            metadata JSONB DEFAULT '{}',
            text TEXT
        );

    Tables are created automatically on first use.

    Args:
        engine: SQLAlchemy async engine connected to the PostgreSQL database.
        dimensions: Embedding vector size (must match the model used).
    """

    def __init__(self, engine: AsyncEngine, *, dimensions: int = 1536) -> None:
        self._engine = engine
        self._dimensions = dimensions
        self._initialised: set[str] = set()

    async def _ensure_table(self, namespace: str) -> None:
        if namespace in self._initialised:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS vs_{namespace} (
                        id TEXT PRIMARY KEY,
                        embedding vector(:dim),
                        metadata JSONB DEFAULT '{{}}',
                        text TEXT
                    )
                    """
                ),
                {"dim": self._dimensions},
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS vs_{namespace}_embedding_idx
                    ON vs_{namespace} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            )
        self._initialised.add(namespace)

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        await self._ensure_table(namespace)
        async with self._engine.begin() as conn:
            for doc in documents:
                embedding_str = "[" + ",".join(str(v) for v in doc.embedding) + "]"
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO vs_{namespace} (id, embedding, metadata, text)
                        VALUES (:id, :embedding::vector, :metadata::jsonb, :text)
                        ON CONFLICT (id) DO UPDATE
                            SET embedding = EXCLUDED.embedding,
                                metadata  = EXCLUDED.metadata,
                                text      = EXCLUDED.text
                        """
                    ),
                    {
                        "id": doc.id,
                        "embedding": embedding_str,
                        "metadata": json.dumps(doc.metadata),
                        "text": doc.text,
                    },
                )

    async def search(
        self,
        embedding: list[float],
        *,
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        await self._ensure_table(namespace)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text(
                    f"""
                    SELECT id, metadata, text,
                           1 - (embedding <=> :embedding::vector) AS score
                    FROM vs_{namespace}
                    WHERE 1 - (embedding <=> :embedding::vector) >= :min_score
                    ORDER BY embedding <=> :embedding::vector
                    LIMIT :top_k
                    """
                ),
                {"embedding": embedding_str, "min_score": min_score, "top_k": top_k},
            )
            return [
                SearchResult(
                    id=row.id,
                    score=float(row.score),
                    metadata=row.metadata or {},
                    text=row.text,
                )
                for row in rows
            ]

    async def delete(self, ids: list[str], *, namespace: str = "default") -> None:
        await self._ensure_table(namespace)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM vs_{namespace} WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
