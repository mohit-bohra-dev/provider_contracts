"""Langfuse telemetry provider implementation."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import AbstractTelemetryProvider, SpanContext

class LangfuseTelemetryProvider(AbstractTelemetryProvider):
    """Telemetry provider that exports spans + metrics to Langfuse."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from langfuse import Langfuse  # type: ignore[import-not-found]
            self._client = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
            )
        return self._client

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[SpanContext]:
        client = self._get_client()
        
        span_obj = client.start_observation(name=name, as_type="span", metadata=attributes)
        
        try:
            yield SpanContext(
                trace_id=span_obj.trace_id,
                span_id=span_obj.id,
                name=name
            )
        finally:
            span_obj.end()

    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        client = self._get_client()
        
        span_obj = client.start_observation(name="metric_record", as_type="span")
        
        client.create_score(
            trace_id=span_obj.trace_id,
            name=name,
            value=value,
            comment=f"Unit: {unit}",
        )
        span_obj.end()

    async def flush(self) -> None:
        """Flush buffered events to Langfuse."""
        if self._client:
            self._client.flush()
