"""Graph store provider interfaces (SPARQL 1.1)."""

from ._base import (
    DEFAULT_GRAPH_NAMESPACE,
    AbstractGraphStore,
    GraphStoreError,
    GraphStoreQueryError,
    GraphStoreUnavailable,
    named_graph_iri,
)
from .mock import FakeGraphStore, MockGraphStoreProvider

__all__ = [
    "DEFAULT_GRAPH_NAMESPACE",
    "AbstractGraphStore",
    "FakeGraphStore",
    "GraphStoreError",
    "GraphStoreQueryError",
    "GraphStoreUnavailable",
    "MockGraphStoreProvider",
    "named_graph_iri",
]
