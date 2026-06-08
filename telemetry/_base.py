"""Abstract base class for telemetry / observability providers."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel


class SpanContext(BaseModel):
    """Lightweight handle to an active tracing span."""

    trace_id: str
    span_id: str
    name: str


class AbstractTelemetryProvider(ABC):
    """
    Contract that every telemetry adapter must satisfy.

    Concrete implementations may emit to the console, OpenTelemetry collectors,
    Azure Application Insights, Datadog, etc.
    """

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[SpanContext]:
        """
        Create and activate a tracing span as an async context manager.

        Override this in concrete implementations.

        Args:
            name: Human-readable span name.
            attributes: Optional key-value attributes attached to the span.

        Yields:
            SpanContext with trace/span identifiers.
        """
        import uuid

        ctx = SpanContext(trace_id=str(uuid.uuid4()), span_id=str(uuid.uuid4()), name=name)
        yield ctx

    @abstractmethod
    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a single numeric metric observation.

        Args:
            name: Metric name (e.g. 'llm.latency_ms').
            value: Numeric value.
            unit: Unit string (e.g. 'ms', 'tokens', '1').
            attributes: Optional dimension labels.
        """
        ...
