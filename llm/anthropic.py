"""Anthropic Claude LLM adapter."""
import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_DEFAULT_MODEL = "claude-3-5-haiku-latest"


class AnthropicProvider(AbstractLLMProvider):
    """Adapter for Anthropic Claude API."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    def _build_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[dict[str, str]]]:
        system = None
        chat_msgs: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                chat_msgs.append({"role": msg.role, "content": msg.content})
        return system, chat_msgs

    def _to_anthropic_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    def _extract_text(self, blocks: list[Any]) -> str:
        text_chunks: list[str] = []
        for block in blocks:
            if getattr(block, "type", "") == "text":
                value = getattr(block, "text", "")
                if isinstance(value, str):
                    text_chunks.append(value)
        return "\n".join(chunk for chunk in text_chunks if chunk)

    def _parse_tool_calls(self, blocks: list[Any]) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        for block in blocks:
            if getattr(block, "type", "") != "tool_use":
                continue
            name = getattr(block, "name", "")
            if not name:
                continue
            input_data = getattr(block, "input", {})
            arguments: dict[str, Any]
            if isinstance(input_data, dict):
                arguments = input_data
            elif isinstance(input_data, str):
                try:
                    loaded = json.loads(input_data)
                    arguments = loaded if isinstance(loaded, dict) else {}
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = {}
            tool_calls.append(ToolCallRequest(name=name, arguments=arguments))
        return tool_calls

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return LLMResponse(
            content=self._extract_text(response.content),
            model=self._model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            tool_calls=self._parse_tool_calls(response.content),
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        system, chat_msgs = self._build_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_msgs,
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        response = await self._client.messages.create(**kwargs)
        return LLMResponse(
            content=self._extract_text(response.content),
            model=self._model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            tool_calls=self._parse_tool_calls(response.content),
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        return self._stream_impl(messages, temperature=temperature, max_tokens=max_tokens)

    async def _stream_impl(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        system, chat_msgs = self._build_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_msgs,
        )
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Anthropic does not provide an embedding API.")
