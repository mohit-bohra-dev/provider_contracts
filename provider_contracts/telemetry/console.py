"""Console telemetry provider — logs spans and metrics to stderr."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import AbstractTelemetryProvider, SpanContext, SpanKind, trace_span

_logger = logging.getLogger("telemetry.console")


def _get_logger() -> Any:
    """Try structlog; fall back to stdlib logging."""
    try:
        import structlog  # type: ignore[import-untyped]

        return structlog.get_logger("telemetry.console")
    except ImportError:
        return _logger


class ConsoleTelemetryProvider(AbstractTelemetryProvider):
    """Logs spans and metrics via structlog (preferred) or stdlib logging.

    Install ``structlog`` for pretty-printed structured output::

        pip install provider-contracts[structlog]
    """

    def __init__(self) -> None:
        self._log: Any = _get_logger()

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind | None = None,
    ) -> AsyncIterator[SpanContext]:
        async with trace_span(name, attributes=attributes, kind=kind) as ctx:
            self._log.info(
                "span.start",
                span_name=name,
                kind=ctx.kind,
                parent_span_id=ctx.parent_span_id,
                **(attributes or {}),
            )
            try:
                yield ctx
            finally:
                self._log.info(
                    "span.end",
                    span_name=name,
                    elapsed_ms=ctx.latency_ms,
                    is_error=ctx.is_error,
                )

    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._log.info("metric", metric_name=name, value=value, unit=unit, **(attributes or {}))
