"""Contract tests for graph store mock."""

from __future__ import annotations

import pytest

from provider_contracts.graph_store import FakeGraphStore, named_graph_iri


@pytest.mark.asyncio
async def test_fake_graph_store_replace_and_query() -> None:
    store = FakeGraphStore()
    iri = named_graph_iri("repo-a")
    turtle = (
        "@prefix ex: <http://example.org/> .\n"
        "ex:a ex:p ex:b .\n"
    )
    await store.replace_named_graph(iri, turtle)
    result = await store.query(
        f"SELECT ?s ?p ?o WHERE {{ GRAPH <{iri}> {{ ?s ?p ?o }} }}"
    )
    bindings = result["results"]["bindings"]
    assert len(bindings) == 1
    assert bindings[0]["s"]["value"].endswith("/a")
