"""Abstract base class for SPARQL graph store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

DEFAULT_GRAPH_NAMESPACE = "https://loanops.local/ontology/"


class GraphStoreError(Exception):
    """Base error for graph store failures."""


class GraphStoreUnavailable(GraphStoreError):
    """Endpoint unreachable or returned a transport error."""


class GraphStoreQueryError(GraphStoreError):
    """SPARQL query/update rejected by the endpoint."""


def named_graph_iri(
    repository_id: str,
    *,
    namespace: str = DEFAULT_GRAPH_NAMESPACE,
) -> str:
    """Canonical named-graph IRI for one repository shard."""
    base = namespace if namespace.endswith("/") or namespace.endswith("#") else f"{namespace}/"
    return f"{base}namedGraph/{repository_id}"


class AbstractGraphStore(ABC):
    """SPARQL query + named-graph replace surface."""

    @abstractmethod
    async def query(self, sparql: str) -> dict[str, Any]:
        """Run SPARQL SELECT/ASK; return SPARQL Results JSON."""

    @abstractmethod
    async def update(self, sparql: str) -> None:
        """Run SPARQL UPDATE (CLEAR / INSERT / DELETE)."""

    @abstractmethod
    async def replace_named_graph(self, graph_iri: str, turtle: str) -> None:
        """Replace all triples in ``graph_iri`` with ``turtle`` content."""

    async def aclose(self) -> None:
        """Release HTTP resources (optional)."""
