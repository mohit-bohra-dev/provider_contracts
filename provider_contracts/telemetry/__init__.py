"""Telemetry provider interfaces."""

from ._base import (
    AbstractTelemetryProvider,
    BufferedSpan,
    SpanContext,
    SpanKind,
    begin_trace_collection,
    current_span_context,
    drain_trace_spans,
    record_completed_span,
    trace_span,
)

__all__ = [
    "AbstractTelemetryProvider",
    "BufferedSpan",
    "SpanContext",
    "SpanKind",
    "begin_trace_collection",
    "current_span_context",
    "drain_trace_spans",
    "record_completed_span",
    "trace_span",
]
