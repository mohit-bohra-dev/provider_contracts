"""Tests for the AWS Bedrock LLM provider."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    import aioboto3
except ImportError:
    pytest.skip("aioboto3 not installed, skipping Bedrock tests", allow_module_level=True)

from provider_contracts.llm import LLMMessage, LLMResponse, ToolCallRequest, ToolDefinition
from provider_contracts.llm.bedrock import BedrockProvider


@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    client = AsyncMock()
    
    # Setup context manager for async with
    client_cm = AsyncMock()
    client_cm.__aenter__.return_value = client
    session.client.return_value = client_cm
    
    return session


@pytest.fixture
def provider(mock_session: MagicMock) -> BedrockProvider:
    with patch("provider_contracts.llm.bedrock.aioboto3.Session", return_value=mock_session):
        return BedrockProvider(region="us-east-1")


@pytest.mark.asyncio
async def test_complete_returns_llm_response(
    provider: BedrockProvider, mock_session: MagicMock
) -> None:
    client = await mock_session.client.return_value.__aenter__()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello world!"}]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5}
    }

    result = await provider.complete("Hi")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello world!"
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 5
    client.converse.assert_called_once()
    
    # Check arguments passed to converse
    kwargs = client.converse.call_args[1]
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert "inferenceConfig" in kwargs


@pytest.mark.asyncio
async def test_chat_returns_llm_response(
    provider: BedrockProvider, mock_session: MagicMock
) -> None:
    client = await mock_session.client.return_value.__aenter__()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello!"}]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5}
    }

    messages = [
        LLMMessage(role="system", content="You are helpful."),
        LLMMessage(role="user", content="Hi")
    ]
    result = await provider.chat(messages)

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello!"
    
    kwargs = client.converse.call_args[1]
    assert kwargs["system"] == [{"text": "You are helpful."}]
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]


@pytest.mark.asyncio
async def test_chat_tool_calls(
    provider: BedrockProvider, mock_session: MagicMock
) -> None:
    client = await mock_session.client.return_value.__aenter__()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Let me check the weather."},
                    {"toolUse": {"name": "get_weather", "input": {"location": "Seattle"}}}
                ]
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 20}
    }

    tools = [
        ToolDefinition(name="get_weather", description="Get weather", parameters={"type": "object"})
    ]
    messages = [LLMMessage(role="user", content="Weather in Seattle?")]
    result = await provider.chat(messages, tools=tools)

    assert result.content == "Let me check the weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"location": "Seattle"}

    kwargs = client.converse.call_args[1]
    assert "toolConfig" in kwargs
    assert kwargs["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_stream_yields_strings(
    provider: BedrockProvider, mock_session: MagicMock
) -> None:
    client = await mock_session.client.return_value.__aenter__()
    
    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"contentBlockDelta": {"delta": {"text": "Hello "}}}
        yield {"contentBlockDelta": {"delta": {"text": "world!"}}}
        yield {"metadata": {"usage": {"inputTokens": 10}}}

    client.converse_stream.return_value = {"stream": mock_stream()}

    messages = [LLMMessage(role="user", content="Hi")]
    chunks: list[str] = []
    async for chunk in provider.stream(messages):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world!"]
    client.converse_stream.assert_called_once()


@pytest.mark.asyncio
async def test_embed_raises_not_implemented(provider: BedrockProvider) -> None:
    with pytest.raises(NotImplementedError):
        await provider.embed("test")
