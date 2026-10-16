"""SPARQL 1.1 over HTTP — GraphDB Free, Neptune RDF, etc."""

from __future__ import annotations

import json
from typing import Any

import httpx
from rdflib import Graph

from ._base import (
    AbstractGraphStore,
    GraphStoreError,
    GraphStoreQueryError,
    GraphStoreUnavailable,
)


class HttpSparqlGraphStore(AbstractGraphStore):
    """SPARQL 1.1 over HTTP."""

    def __init__(
        self,
        *,
        query_url: str,
        update_url: str,
        auth_token: str = "",
        timeout_seconds: int = 30,
    ) -> None:
        if not query_url.strip():
            raise GraphStoreError("query_url is required")
        if not update_url.strip():
            raise GraphStoreError("update_url is required")
        self._query_url = query_url.strip()
        self._update_url = update_url.strip()
        self._auth_token = auth_token.strip()
        self._timeout = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    def _headers(self, *, content_type: str, accept: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": content_type}
        if accept:
            headers["Accept"] = accept
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _auth_headers_only(self) -> dict[str, str]:
        if not self._auth_token:
            return {}
        return {"Authorization": f"Bearer {self._auth_token}"}

    async def query(self, sparql: str) -> dict[str, Any]:
        try:
            resp = await self._client.post(
                self._query_url,
                content=sparql.encode("utf-8"),
                headers=self._headers(
                    content_type="application/sparql-query",
                    accept="application/sparql-results+json",
                ),
            )
        except httpx.HTTPError as exc:
            raise GraphStoreUnavailable(f"SPARQL query failed: {exc}") from exc
        if resp.status_code >= 400:
            raise GraphStoreQueryError(
                f"SPARQL query HTTP {resp.status_code}: {resp.text[:500]}"
            )
        try:
            data: dict[str, Any] = resp.json()
        except json.JSONDecodeError as exc:
            raise GraphStoreQueryError("SPARQL query returned non-JSON body") from exc
        return data

    async def update(self, sparql: str) -> None:
        try:
            resp = await self._client.post(
                self._update_url,
                content=sparql.encode("utf-8"),
                headers=self._headers(content_type="application/sparql-update"),
            )
        except httpx.HTTPError as exc:
            raise GraphStoreUnavailable(f"SPARQL update failed: {exc}") from exc
        if resp.status_code < 400:
            return
        try:
            resp2 = await self._client.post(
                self._update_url,
                data={"update": sparql},
                headers=self._auth_headers_only(),
            )
        except httpx.HTTPError as exc:
            raise GraphStoreUnavailable(f"SPARQL update failed: {exc}") from exc
        if resp2.status_code >= 400:
            raise GraphStoreQueryError(
                f"SPARQL update HTTP {resp.status_code}/{resp2.status_code}: "
                f"{resp.text[:300]} | {resp2.text[:300]}"
            )

    async def replace_named_graph(self, graph_iri: str, turtle: str) -> None:
        g = Graph()
        g.parse(data=turtle, format="turtle")
        ntriples = g.serialize(format="nt")
        if isinstance(ntriples, bytes):
            ntriples = ntriples.decode("utf-8")
        update = (
            f"CLEAR SILENT GRAPH <{graph_iri}> ;\n"
            f"INSERT DATA {{ GRAPH <{graph_iri}> {{\n{ntriples}\n}} }}\n"
        )
        await self.update(update)

    async def aclose(self) -> None:
        await self._client.aclose()
