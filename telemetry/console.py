"""Console telemetry provider — logs spans and metrics to stderr."""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import AbstractTelemetryProvider, SpanContext

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
            trace_id=f"console-{self._span_counter}",
            span_id=f"span-{self._span_counter}",
            name=name,
        )
        start = time.perf_counter()
        self._log.info("span.start", span_name=name, **(attributes or {}))
        try:
            yield ctx
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._log.info("span.end", span_name=name, elapsed_ms=round(elapsed_ms, 2))

    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._log.info("metric", metric_name=name, value=value, unit=unit, **(attributes or {}))
