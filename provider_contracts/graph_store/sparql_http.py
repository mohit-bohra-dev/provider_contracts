"""SPARQL 1.1 over HTTP — GraphDB Free, Neptune RDF, etc."""

from __future__ import annotations

import json
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

import httpx
from rdflib import Graph

from ._base import (
    AbstractGraphStore,
    GraphStoreError,
    GraphStoreQueryError,
    GraphStoreUnavailable,
)

AuthMode = Literal["none", "bearer", "sigv4"]


class HttpSparqlGraphStore(AbstractGraphStore):
    """SPARQL 1.1 over HTTP.

    ``auth_mode=sigv4`` for Amazon Neptune (SigV4 service ``neptune-db``).
    ``auth_mode=bearer`` for GraphDB / token proxies.
    """

    def __init__(
        self,
        *,
        query_url: str,
        update_url: str,
        auth_token: str = "",
        auth_mode: AuthMode = "none",
        region: str = "",
        timeout_seconds: int = 30,
    ) -> None:
        if not query_url.strip():
            raise GraphStoreError("query_url is required")
        if not update_url.strip():
            raise GraphStoreError("update_url is required")
        mode: AuthMode = auth_mode
        if mode == "none" and auth_token.strip():
            mode = "bearer"
        if mode == "sigv4" and not region.strip():
            raise GraphStoreError("region is required when auth_mode=sigv4")
        self._query_url = query_url.strip()
        self._update_url = update_url.strip()
        self._auth_token = auth_token.strip()
        self._auth_mode = mode
        self._region = region.strip()
        self._timeout = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    def _base_headers(
        self, *, content_type: str, accept: str | None = None
    ) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": content_type}
        if accept:
            headers["Accept"] = accept
        if self._auth_mode == "bearer" and self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    def _sign(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> dict[str, str]:
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.session import Session
        except ImportError as exc:  # pragma: no cover
            raise GraphStoreError(
                "auth_mode=sigv4 requires botocore (install AWS extra)"
            ) from exc
        session = Session()
        creds = session.get_credentials()
        if creds is None:
            raise GraphStoreUnavailable("no AWS credentials for Neptune SigV4")
        frozen = creds.get_frozen_credentials()
        request = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
        # Host must match Neptune endpoint for signing.
        host = urlparse(url).netloc
        if host:
            request.headers["Host"] = host
        SigV4Auth(frozen, "neptune-db", self._region).add_auth(request)
        return {k: str(v) for k, v in request.headers.items()}

    async def _post(
        self,
        url: str,
        *,
        body: bytes,
        content_type: str,
        accept: str | None = None,
        form: dict[str, str] | None = None,
    ) -> httpx.Response:
        headers = self._base_headers(content_type=content_type, accept=accept)
        if form is not None:
            raw = urlencode(form).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            if self._auth_mode == "sigv4":
                headers = self._sign("POST", url, headers, raw)
            return await self._client.post(url, content=raw, headers=headers)
        if self._auth_mode == "sigv4":
            headers = self._sign("POST", url, headers, body)
        return await self._client.post(url, content=body, headers=headers)

    async def query(self, sparql: str) -> dict[str, Any]:
        try:
            resp = await self._post(
                self._query_url,
                body=sparql.encode("utf-8"),
                content_type="application/sparql-query",
                accept="application/sparql-results+json",
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
            resp = await self._post(
                self._update_url,
                body=sparql.encode("utf-8"),
                content_type="application/sparql-update",
            )
        except httpx.HTTPError as exc:
            raise GraphStoreUnavailable(f"SPARQL update failed: {exc}") from exc
        if resp.status_code < 400:
            return
        try:
            resp2 = await self._post(
                self._update_url,
                body=b"",
                content_type="application/x-www-form-urlencoded",
                form={"update": sparql},
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
