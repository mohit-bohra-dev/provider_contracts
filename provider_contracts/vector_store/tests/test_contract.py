"""Contract tests for vector-store filter any-of + ensure_namespace (G6)."""

from __future__ import annotations

import pytest

from provider_contracts.errors import ProviderConfigError
from provider_contracts.vector_store import (
    AbstractVectorStoreProvider,
    SearchResult,
    VectorDocument,
)
from provider_contracts.vector_store._filter import build_odata_filter
from provider_contracts.vector_store.mock import MockVectorStoreProvider


@pytest.fixture
def mock_store() -> MockVectorStoreProvider:
    return MockVectorStoreProvider()


@pytest.mark.asyncio
async def test_supports_any_of_flag(mock_store: MockVectorStoreProvider) -> None:
    assert MockVectorStoreProvider.supports_any_of_filter is True
    assert mock_store.supports_any_of_filter is True
    assert AbstractVectorStoreProvider.supports_any_of_filter is False


@pytest.mark.asyncio
async def test_mock_scalar_filter_equality(mock_store: MockVectorStoreProvider) -> None:
    await mock_store.ensure_namespace("ns", vector_size=3)
    await mock_store.upsert(
        [
            VectorDocument(
                id="1",
                embedding=[1.0, 0.0, 0.0],
                metadata={"application_id": "fees", "read_only": True},
                text="fees",
            ),
            VectorDocument(
                id="2",
                embedding=[0.9, 0.1, 0.0],
                metadata={"application_id": "escrow", "read_only": True},
                text="escrow",
            ),
        ],
        namespace="ns",
    )
    results = await mock_store.search(
        [1.0, 0.0, 0.0],
        namespace="ns",
        top_k=5,
        filter={"application_id": "fees", "read_only": True},
    )
    assert [r.id for r in results] == ["1"]
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_mock_list_filter_any_of(mock_store: MockVectorStoreProvider) -> None:
    await mock_store.ensure_namespace("ns", vector_size=2)
    await mock_store.upsert(
        [
            VectorDocument(
                id="a",
                embedding=[1.0, 0.0],
                metadata={"application_id": "fees", "read_only": True},
            ),
            VectorDocument(
                id="b",
                embedding=[0.8, 0.2],
                metadata={"application_id": "escrow", "read_only": True},
            ),
            VectorDocument(
                id="c",
                embedding=[0.7, 0.3],
                metadata={"application_id": "loans", "read_only": True},
            ),
        ],
        namespace="ns",
    )
    results = await mock_store.search(
        [1.0, 0.0],
        namespace="ns",
        top_k=10,
        filter={"application_id": ["fees", "loans"], "read_only": True},
    )
    assert {r.id for r in results} == {"a", "c"}


@pytest.mark.asyncio
async def test_mock_ensure_namespace_idempotent(
    mock_store: MockVectorStoreProvider,
) -> None:
    await mock_store.ensure_namespace("ops", vector_size=4)
    await mock_store.ensure_namespace("ops", vector_size=4)
    await mock_store.upsert(
        [VectorDocument(id="1", embedding=[1.0, 0.0, 0.0, 0.0], text="x")],
        namespace="ops",
    )
    hits = await mock_store.search([1.0, 0.0, 0.0, 0.0], namespace="ops", top_k=1)
    assert hits[0].id == "1"


@pytest.mark.asyncio
async def test_mock_ensure_namespace_size_mismatch(
    mock_store: MockVectorStoreProvider,
) -> None:
    await mock_store.ensure_namespace("ops", vector_size=4)
    with pytest.raises(ProviderConfigError, match="vector_size"):
        await mock_store.ensure_namespace("ops", vector_size=8)


@pytest.mark.asyncio
async def test_mock_ensure_namespace_resolves_like_upsert(
    mock_store: MockVectorStoreProvider,
) -> None:
    """Mock resolves namespace as the collection name itself (same as upsert)."""
    await mock_store.ensure_namespace("api_ops_dev_deadbeef", vector_size=3)
    await mock_store.upsert(
        [
            VectorDocument(
                id="p1",
                embedding=[0.0, 1.0, 0.0],
                metadata={"application_id": "fees"},
            )
        ],
        namespace="api_ops_dev_deadbeef",
    )
    hits = await mock_store.search(
        [0.0, 1.0, 0.0],
        namespace="api_ops_dev_deadbeef",
        top_k=1,
        filter={"application_id": ["fees", "escrow"]},
    )
    assert len(hits) == 1
    assert hits[0].id == "p1"


def test_concrete_providers_advertise_any_of() -> None:
    pytest.importorskip("sqlalchemy")
    from provider_contracts.vector_store.pgvector import PgVectorProvider

    assert PgVectorProvider.supports_any_of_filter is True
    azure = pytest.importorskip("azure.search.documents")
    from provider_contracts.vector_store.azure_ai_search import (
        AzureAISearchVectorStoreProvider,
    )

    assert AzureAISearchVectorStoreProvider.supports_any_of_filter is True
    _ = azure


def test_pgvector_filter_sql_scalar_and_any_of() -> None:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    from unittest.mock import MagicMock

    from provider_contracts.vector_store.pgvector import PgVectorProvider

    provider = PgVectorProvider(MagicMock(), dimensions=3)
    empty_sql, empty_params = provider._filter_sql(None)
    assert empty_sql == ""
    assert empty_params == {}

    sql, params = provider._filter_sql(
        {"read_only": True, "application_id": ["fees", "escrow"]}
    )
    assert "(metadata->'read_only') = (:f0)::jsonb" in sql
    assert "(metadata->'application_id') <@ (:f1)::jsonb" in sql
    assert params["f0"] == "true"
    assert params["f1"] == '["fees", "escrow"]'
    _ = sqlalchemy


def test_azure_odata_filter_scalar_and_any_of() -> None:
    assert build_odata_filter({"read_only": True}) == "read_only eq true"
    assert (
        build_odata_filter({"application_id": "fees"})
        == "application_id eq 'fees'"
    )
    assert (
        build_odata_filter({"application_id": ["fees", "escrow"]})
        == "(application_id eq 'fees' or application_id eq 'escrow')"
    )
    combined = build_odata_filter(
        {"read_only": True, "application_id": ["a", "b"]}
    )
    assert "read_only eq true" in combined
    assert "(application_id eq 'a' or application_id eq 'b')" in combined


