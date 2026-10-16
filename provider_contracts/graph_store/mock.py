"""In-process SPARQL double for unit/contract tests."""

from __future__ import annotations

from typing import Any

from rdflib import Dataset, Graph, Literal, URIRef
from rdflib.term import BNode, Identifier, Variable

from ._base import AbstractGraphStore, GraphStoreQueryError


class FakeGraphStore(AbstractGraphStore):
    """In-process SPARQL double for unit/contract tests (not a product backend)."""

    def __init__(self) -> None:
        self._ds = Dataset()

    async def query(self, sparql: str) -> dict[str, Any]:
        try:
            result = self._ds.query(sparql)
        except Exception as exc:  # noqa: BLE001 — surface as query error
            raise GraphStoreQueryError(str(exc)) from exc
        return _rdflib_result_to_json(result)

    async def update(self, sparql: str) -> None:
        try:
            self._ds.update(sparql)
        except Exception as exc:  # noqa: BLE001
            raise GraphStoreQueryError(str(exc)) from exc

    async def replace_named_graph(self, graph_iri: str, turtle: str) -> None:
        ctx = URIRef(graph_iri)
        try:
            self._ds.remove_graph(self._ds.graph(ctx))
        except Exception:  # noqa: BLE001
            pass
        g = Graph(store=self._ds.store, identifier=ctx)
        g.parse(data=turtle, format="turtle")


# Alias preferred in provider_contracts naming
MockGraphStoreProvider = FakeGraphStore


def _term_to_binding(val: Identifier) -> dict[str, str]:
    if isinstance(val, URIRef):
        return {"type": "uri", "value": str(val)}
    if isinstance(val, BNode):
        return {"type": "bnode", "value": str(val)}
    if isinstance(val, Literal):
        entry: dict[str, str] = {"type": "literal", "value": str(val)}
        if val.datatype is not None:
            entry["datatype"] = str(val.datatype)
        if val.language:
            entry["xml:lang"] = str(val.language)
        return entry
    return {"type": "literal", "value": str(val)}


def _rdflib_result_to_json(result: Any) -> dict[str, Any]:
    """Convert RDFLib query result to SPARQL Results JSON shape."""
    if getattr(result, "type", None) == "ASK":
        return {"head": {}, "boolean": bool(result.askAnswer)}
    raw_vars = list(result.vars) if result.vars else []
    var_names = [str(v) for v in raw_vars]
    bindings: list[dict[str, Any]] = []
    for row in result:
        binding: dict[str, Any] = {}
        for var in raw_vars:
            name = str(var)
            try:
                val = row[var] if isinstance(var, Variable) else row[name]
            except Exception:  # noqa: BLE001
                try:
                    val = row[name]
                except Exception:  # noqa: BLE001
                    continue
            if val is None:
                continue
            binding[name] = _term_to_binding(val)
        bindings.append(binding)
    return {
        "head": {"vars": var_names},
        "results": {"bindings": bindings},
    }
