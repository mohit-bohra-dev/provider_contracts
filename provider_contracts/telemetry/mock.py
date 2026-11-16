"""In-memory mock telemetry provider — no-op spans, records metrics for assertions."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import AbstractTelemetryProvider, SpanContext, SpanKind, trace_span


class MockTelemetryProvider(AbstractTelemetryProvider):
    """Buffered spans + records metrics into ``recorded_metrics`` for assertions."""

    def __init__(self) -> None:
        self.recorded_metrics: list[dict[str, Any]] = []
        self.recorded_spans: list[dict[str, Any]] = []

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind | None = None,
    ) -> AsyncIterator[SpanContext]:
        attrs = attributes or {}
        async with trace_span(name, attributes=attrs, kind=kind) as ctx:
            self.recorded_spans.append(
                {
                    "name": name,
                    "attributes": dict(attrs),
                    "kind": ctx.kind,
                    "trace_id": ctx.trace_id,
                    "span_id": ctx.span_id,
                    "parent_span_id": ctx.parent_span_id,
                }
            )
            yield ctx

    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.recorded_metrics.append(
            {"name": name, "value": value, "unit": unit, "attributes": attributes or {}}
        )
