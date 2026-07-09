"""Contract tests for TelemetryProvider."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from provider_contracts.telemetry._base import AbstractTelemetryProvider, SpanContext
from provider_contracts.telemetry.langfuse import LangfuseTelemetryProvider

@pytest.fixture
def mock_langfuse_client(monkeypatch):
    mock = MagicMock()
    # Mock the trace and span chain
    mock_trace = MagicMock()
    mock_span = MagicMock()
    
    mock.trace.return_value = mock_trace
    mock_trace.span.return_value = mock_span
    mock_trace.id = "trace-123"
    mock_span.id = "span-456"
    
    monkeypatch.setattr("provider_contracts.telemetry.langfuse.LangfuseTelemetryProvider._get_client", lambda self: mock)
    return mock

@pytest.mark.asyncio
async def test_langfuse_span(mock_langfuse_client):
    provider = LangfuseTelemetryProvider(
        public_key="pk-test", 
        secret_key="sk-test", 
        host="http://localhost:3000"
    )
    
    async with provider.span("test-span", attributes={"attr1": "val1"}) as ctx:
        assert isinstance(ctx, SpanContext)
        assert ctx.name == "test-span"
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
    
    mock_langfuse_client.trace.assert_called_once_with(name="test-span", metadata={"attr1": "val1"})
    mock_langfuse_client.trace().span.assert_called_once_with(name="test-span")
    mock_langfuse_client.trace().span().end.assert_called_once()

@pytest.mark.asyncio
async def test_langfuse_record_metric(mock_langfuse_client):
    provider = LangfuseTelemetryProvider(
        public_key="pk-test", 
        secret_key="sk-test", 
        host="http://localhost:3000"
    )
    
    await provider.record_metric(
        "test.metric", 
        0.95, 
        unit="ratio", 
        attributes={"env": "test"}
    )
    
    mock_langfuse_client.score.assert_called_once_with(
        trace_id="trace-123",
        name="test.metric",
        value=0.95,
        comment="Unit: ratio",
    )
