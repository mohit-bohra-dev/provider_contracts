"""pgvector adapter for the vector store provider using asyncpg + SQLAlchemy async."""
from __future__ import annotations

import json
from typing import Any, ClassVar

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from provider_contracts.errors import ProviderConfigError

from ._base import AbstractVectorStoreProvider, SearchResult, VectorDocument
from ._filter import validate_filter_key


class PgVectorProvider(AbstractVectorStoreProvider):
    """
    Vector store backed by PostgreSQL with the pgvector extension.

    Expects a table per namespace with the schema::

        CREATE TABLE vs_<namespace> (
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

    supports_any_of_filter: ClassVar[bool] = True

    def __init__(self, engine: AsyncEngine, *, dimensions: int = 1536) -> None:
        self._engine = engine
        self._dimensions = dimensions
        self._initialised: set[str] = set()

    def _table_name(self, namespace: str) -> str:
        return f"vs_{namespace}"

    async def create_collection(
        self, name: str, vector_size: int, distance: str = "Cosine"
    ) -> None:
        _ = distance
        await self.ensure_namespace(name, vector_size)

    async def ensure_namespace(self, namespace: str, vector_size: int) -> None:
        table = self._table_name(namespace)
        async with self._engine.connect() as conn:
            exists_row = await conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = :table
                    )
                    """
                ),
                {"table": table},
            )
            exists = bool(exists_row.scalar())
            if exists:
                # pgvector stores dimensions in atttypmod directly (not
                # atttypmod-4 like varchar/bpchar). format_type is the
                # portable check: vector(384).
                dims_row = await conn.execute(
                    text(
                        """
                        SELECT a.atttypmod AS dims,
                               format_type(a.atttypid, a.atttypmod) AS ft
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = 'public'
                          AND c.relname = :table
                          AND a.attname = 'embedding'
                          AND NOT a.attisdropped
                        """
                    ),
                    {"table": table},
                )
                row = dims_row.first()
                current: int | None = None
                if row is not None:
                    ft = str(row.ft or "")
                    if ft.startswith("vector(") and ft.endswith(")"):
                        try:
                            current = int(ft[len("vector(") : -1])
                        except ValueError:
                            current = None
                    if current is None and row.dims is not None and int(row.dims) > 0:
                        current = int(row.dims)
                if current is not None and current != vector_size:
                    raise ProviderConfigError(
                        f"Table {table!r} exists with vector_size "
                        f"{current}, requested {vector_size}"
                    )
                self._initialised.add(namespace)
                return

        self._dimensions = vector_size
        await self._ensure_table(namespace)

    async def _ensure_table(self, namespace: str) -> None:
        if namespace in self._initialised:
            return
        table = self._table_name(namespace)
        async with self._engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        embedding vector({self._dimensions}),
                        metadata JSONB DEFAULT '{{}}',
                        text TEXT
                    )
                    """
                )
            )
            await conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                    ON {table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            )
        self._initialised.add(namespace)

    def _filter_sql(
        self, filter: dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        if not filter:
            return "", {}
        clauses: list[str] = []
        params: dict[str, Any] = {}
        for i, (key, value) in enumerate(filter.items()):
            validate_filter_key(key)
            param = f"f{i}"
            if isinstance(value, list):
                clauses.append(f"(metadata->'{key}') <@ CAST(:{param} AS jsonb)")
                params[param] = json.dumps(value)
            else:
                clauses.append(f"(metadata->'{key}') = CAST(:{param} AS jsonb)")
                params[param] = json.dumps(value)
        return " AND " + " AND ".join(clauses), params

    async def upsert(
        self, documents: list[VectorDocument], *, namespace: str = "default"
    ) -> None:
        await self._ensure_table(namespace)
        table = self._table_name(namespace)
        async with self._engine.begin() as conn:
            for doc in documents:
                embedding_str = "[" + ",".join(str(v) for v in doc.embedding) + "]"
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {table} (id, embedding, metadata, text)
                        VALUES (
                            :id,
                            CAST(:embedding AS vector),
                            CAST(:metadata AS jsonb),
                            :text
                        )
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
        query_text: str | None = None,
        *,
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        _ = query_text  # hybrid FTS can be added; vector path is primary
        await self._ensure_table(namespace)
        table = self._table_name(namespace)
        filter_sql, filter_params = self._filter_sql(filter)
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        params: dict[str, Any] = {
            "embedding": embedding_str,
            "min_score": min_score,
            "top_k": top_k,
            **filter_params,
        }
        async with self._engine.connect() as conn:
            rows = await conn.execute(
                text(
                    f"""
                    SELECT id, metadata, text,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM {table}
                    WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :min_score
                    {filter_sql}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :top_k
                    """
                ),
                params,
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
        table = self._table_name(namespace)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {table} WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
