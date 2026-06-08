"""In-memory mock telemetry provider — no-op spans, records metrics for assertions."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import AbstractTelemetryProvider, SpanContext


class MockTelemetryProvider(AbstractTelemetryProvider):
    """No-op spans + records metrics into ``recorded_metrics`` for assertions."""

    def __init__(self) -> None:
        self.recorded_metrics: list[dict[str, Any]] = []
        self._span_counter = 0

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> AsyncIterator[SpanContext]:
        self._span_counter += 1
        ctx = SpanContext(
            trace_id=f"mock-trace-{self._span_counter}",
            span_id=f"mock-span-{self._span_counter}",
            name=name,
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
