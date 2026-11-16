"""Contract tests for TelemetryProvider."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from provider_contracts.telemetry._base import (
    SpanContext,
    begin_trace_collection,
    drain_trace_spans,
)
from provider_contracts.telemetry.langfuse import LangfuseTelemetryProvider
from provider_contracts.telemetry.mock import MockTelemetryProvider


class _CurrentSpan:
    def __init__(self, span: MagicMock) -> None:
        self._span = span

    def __enter__(self) -> MagicMock:
        return self._span

    def __exit__(self, *_exc: object) -> bool:
        self._span.end()
        return False


@pytest.fixture
def mock_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    mock_span = MagicMock()
    mock_span.trace_id = "trace-123"
    mock_span.id = "span-456"
    mock.start_observation.return_value = mock_span
    mock.start_as_current_observation.side_effect = lambda **_kw: _CurrentSpan(mock_span)
    monkeypatch.setattr(
        LangfuseTelemetryProvider,
        "_get_client",
        lambda self: mock,
    )
    return mock


@pytest.mark.asyncio
async def test_langfuse_span(mock_langfuse_client: MagicMock) -> None:
    provider = LangfuseTelemetryProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
        environment="eval",
    )

    async with provider.span("test-span", attributes={"attr1": "val1"}) as ctx:
        assert isinstance(ctx, SpanContext)
        assert ctx.name == "test-span"
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"

    mock_langfuse_client.start_as_current_observation.assert_called()
    kwargs = mock_langfuse_client.start_as_current_observation.call_args.kwargs
    assert kwargs["name"] == "test-span"
    mock_span = mock_langfuse_client.start_observation.return_value
    mock_span.end.assert_called_once()
    mock_span.update.assert_not_called()


@pytest.mark.asyncio
async def test_langfuse_tool_span_shows_name_and_io(mock_langfuse_client: MagicMock) -> None:
    provider = LangfuseTelemetryProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
        environment="eval",
    )

    async with provider.span(
        "mcp.tool.call",
        attributes={"tool": "search_sse_apis"},
    ) as ctx:
        ctx.input = {"query": "escrow"}
        ctx.output = "call call_sse_api"

    kwargs = mock_langfuse_client.start_as_current_observation.call_args.kwargs
    assert kwargs["name"] == "search_sse_apis"
    mock_span = mock_langfuse_client.start_observation.return_value
    mock_span.update.assert_called_once_with(
        input={"query": "escrow"},
        output="call call_sse_api",
    )


@pytest.mark.asyncio
async def test_langfuse_record_metric(mock_langfuse_client: MagicMock) -> None:
    provider = LangfuseTelemetryProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
        environment="eval",
    )

    await provider.record_metric(
        "test.metric",
        0.95,
        unit="ratio",
        attributes={"env": "test"},
    )

    mock_langfuse_client.create_score.assert_called()
    kwargs = mock_langfuse_client.create_score.call_args.kwargs
    assert kwargs["name"] == "test.metric"
    assert kwargs["value"] == 0.95


@pytest.mark.asyncio
async def test_langfuse_record_metric_attaches_to_active_trace(
    mock_langfuse_client: MagicMock,
) -> None:
    provider = LangfuseTelemetryProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
        environment="eval",
    )
    async with provider.span("agent.turn", attributes={"agent.trace_id": "trace-123"}):
        await provider.record_metric(
            "agent.turn.latency_ms",
            42.0,
            unit="ms",
            attributes={"agent.trace_id": "trace-123"},
        )

    kwargs = mock_langfuse_client.create_score.call_args.kwargs
    assert kwargs["trace_id"] == "trace-123"
    # Should not open a detached metric_record observation when a trace is active.
    names = [
        c.kwargs.get("name")
        for c in mock_langfuse_client.start_as_current_observation.call_args_list
    ]
    assert "metric_record" not in names


@pytest.mark.asyncio
async def test_nested_spans_buffer_parent_and_latency() -> None:
    begin_trace_collection("trace-abc")
    provider = MockTelemetryProvider()
    async with provider.span("agent.turn", kind="turn", attributes={"agent.trace_id": "trace-abc"}):
        async with provider.span("mcp.tool.call", kind="tool", attributes={"tool": "search_docs"}):
            async with provider.span("docs.embed", kind="subtool"):
                pass
    spans = drain_trace_spans("trace-abc")
    assert len(spans) == 3
    by_name = {s.name: s for s in spans}
    assert by_name["agent.turn"].parent_span_id is None
    assert by_name["mcp.tool.call"].parent_span_id == by_name["agent.turn"].span_id
    assert by_name["docs.embed"].parent_span_id == by_name["mcp.tool.call"].span_id
    assert by_name["docs.embed"].latency_ms >= 0.0
    assert by_name["mcp.tool.call"].tool == "search_docs"


@pytest.mark.asyncio
async def test_langfuse_llm_span_uses_generation(mock_langfuse_client: MagicMock) -> None:
    provider = LangfuseTelemetryProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="http://localhost:3000",
        environment="eval",
    )
    async with provider.span("llm.chat", kind="llm") as ctx:
        ctx.model = "llama3.1"
        ctx.usage = {"prompt_tokens": 10, "completion_tokens": 5}
        ctx.input = "hi"
        ctx.output = "yo"

    kwargs = mock_langfuse_client.start_as_current_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    mock_span = mock_langfuse_client.start_observation.return_value
    update = mock_span.update.call_args.kwargs
    assert update["model"] == "llama3.1"
    assert update["usage_details"] == {"input": 10, "output": 5}
