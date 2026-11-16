"""Abstract base class for telemetry / observability providers."""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, Field

SpanKind = Literal["turn", "llm", "tool", "subtool", "safety", "validator"]


class SpanContext(BaseModel):
    """Lightweight handle to an active tracing span.

    Set ``input`` / ``output`` inside the span. Langfuse shows those on the
    trace log; other providers ignore them.
    """

    trace_id: str
    span_id: str
    name: str
    parent_span_id: str | None = None
    kind: SpanKind = "subtool"
    start_offset_ms: float = 0.0
    latency_ms: float = 0.0
    is_error: bool = False
    error_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    input: Any = None
    output: Any = None


class BufferedSpan(BaseModel):
    """Immutable span snapshot for AgentTrace / UI."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: SpanKind = "subtool"
    start_offset_ms: float = 0.0
    latency_ms: float = 0.0
    is_error: bool = False
    error_message: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    input: Any = None
    output: Any = None
    tool: str | None = None


_active_span: ContextVar[SpanContext | None] = ContextVar("telemetry_active_span", default=None)
_trace_start: ContextVar[float | None] = ContextVar("telemetry_trace_start", default=None)
_trace_buffer: ContextVar[list[BufferedSpan] | None] = ContextVar(
    "telemetry_trace_buffer", default=None
)


def current_span_context() -> SpanContext | None:
    """Return the currently active span, if any."""
    return _active_span.get()


def _ensure_buffer() -> list[BufferedSpan]:
    buf = _trace_buffer.get()
    if buf is None:
        buf = []
        _trace_buffer.set(buf)
    return buf


def begin_trace_collection(trace_id: str | None = None) -> str:
    """Start (or reset) in-memory span collection for a turn. Returns trace_id."""
    tid = trace_id or str(uuid.uuid4())
    _trace_buffer.set([])
    _trace_start.set(time.perf_counter())
    # Clear any leftover active span from a previous cancelled turn.
    _active_span.set(None)
    return tid


def drain_trace_spans(trace_id: str | None = None) -> list[BufferedSpan]:
    """Drain and return buffered spans for the active (or matching) turn."""
    buf = _trace_buffer.get() or []
    if trace_id is not None:
        out = [s for s in buf if s.trace_id == trace_id]
    else:
        out = list(buf)
    _trace_buffer.set([])
    _trace_start.set(None)
    return out


def record_completed_span(
    *,
    name: str,
    kind: SpanKind = "subtool",
    latency_ms: float = 0.0,
    start_offset_ms: float | None = None,
    attributes: dict[str, Any] | None = None,
    usage: dict[str, int] | None = None,
    model: str | None = None,
    is_error: bool = False,
    error_message: str | None = None,
    input: Any = None,
    output: Any = None,
    parent_span_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> BufferedSpan:
    """Append a completed span snapshot (e.g. LLM/validator records from runtime)."""
    parent = _active_span.get()
    start_clock = _trace_start.get()
    offset = start_offset_ms
    if offset is None and start_clock is not None:
        offset = round((time.perf_counter() - start_clock) * 1000, 2) - latency_ms
        if offset < 0:
            offset = 0.0
    tid = (
        trace_id
        or (parent.trace_id if parent is not None else None)
        or str(uuid.uuid4())
    )
    attrs = dict(attributes or {})
    tool = attrs.get("tool")
    tool_name = tool.strip() if isinstance(tool, str) and tool.strip() else None
    snap = BufferedSpan(
        trace_id=str(tid),
        span_id=span_id or str(uuid.uuid4()),
        parent_span_id=parent_span_id
        if parent_span_id is not None
        else (parent.span_id if parent is not None else None),
        name=name,
        kind=kind,
        start_offset_ms=float(offset or 0.0),
        latency_ms=latency_ms,
        is_error=is_error,
        error_message=error_message,
        attributes=attrs,
        usage=dict(usage or {}),
        model=model,
        input=input,
        output=output,
        tool=tool_name,
    )
    _ensure_buffer().append(snap)
    return snap

def _snapshot(ctx: SpanContext) -> BufferedSpan:
    tool = ctx.attributes.get("tool")
    tool_name = tool.strip() if isinstance(tool, str) and tool.strip() else None
    return BufferedSpan(
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        parent_span_id=ctx.parent_span_id,
        name=ctx.name,
        kind=ctx.kind,
        start_offset_ms=ctx.start_offset_ms,
        latency_ms=ctx.latency_ms,
        is_error=ctx.is_error,
        error_message=ctx.error_message,
        attributes=dict(ctx.attributes),
        usage=dict(ctx.usage),
        model=ctx.model,
        input=ctx.input,
        output=ctx.output,
        tool=tool_name,
    )


def _resolve_kind(name: str, attributes: dict[str, Any] | None) -> SpanKind:
    attrs = attributes or {}
    raw = attrs.get("span.kind") or attrs.get("kind")
    if isinstance(raw, str) and raw in {
        "turn",
        "llm",
        "tool",
        "subtool",
        "safety",
        "validator",
    }:
        return raw  # type: ignore[return-value]
    if name == "agent.turn":
        return "turn"
    if name.startswith("llm.") or name == "llm.chat":
        return "llm"
    if name.startswith("mcp.tool") or name.startswith("tool."):
        return "tool"
    if name.startswith("safety.") or name.startswith("pii."):
        return "safety"
    if name.startswith("agent.validate") or name.startswith("validate"):
        return "validator"
    return "subtool"


@asynccontextmanager
async def trace_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
    kind: SpanKind | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
) -> AsyncIterator[SpanContext]:
    """Open a buffered span that inherits parent/trace from contextvars."""
    parent = _active_span.get()
    start_clock = _trace_start.get()
    if start_clock is None:
        start_clock = time.perf_counter()
        _trace_start.set(start_clock)
    attrs = dict(attributes or {})
    resolved_kind = kind or _resolve_kind(name, attrs)
    tid = (
        trace_id
        or (parent.trace_id if parent is not None else None)
        or attrs.get("agent.trace_id")
        or str(uuid.uuid4())
    )
    if not isinstance(tid, str) or not tid:
        tid = str(uuid.uuid4())
    sid = span_id or str(uuid.uuid4())
    ctx = SpanContext(
        trace_id=str(tid),
        span_id=sid,
        name=name,
        parent_span_id=parent.span_id if parent is not None else None,
        kind=resolved_kind,
        start_offset_ms=round((time.perf_counter() - start_clock) * 1000, 2),
        attributes=attrs,
    )
    token = _active_span.set(ctx)
    t0 = time.perf_counter()
    exc_info: BaseException | None = None
    try:
        yield ctx
    except BaseException as exc:
        exc_info = exc
        ctx.is_error = True
        ctx.error_message = str(exc)
        raise
    finally:
        ctx.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _ensure_buffer().append(_snapshot(ctx))
        _active_span.reset(token)
        if exc_info is not None:
            pass


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
        kind: SpanKind | None = None,
    ) -> AsyncIterator[SpanContext]:
        """
        Create and activate a tracing span as an async context manager.

        Override this in concrete implementations. Default buffers locally via
        ``trace_span``.
        """
        async with trace_span(name, attributes=attributes, kind=kind) as ctx:
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
