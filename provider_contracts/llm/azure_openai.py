"""Azure OpenAI LLM adapter."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncAzureOpenAI

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_API_VERSION = "2024-08-01-preview"


class AzureOpenAIProvider(AbstractLLMProvider):
    """Adapter for Azure OpenAI Service.

    Uses ``openai.AsyncAzureOpenAI`` which handles Azure-specific auth
    and endpoint routing.

    Requires the ``openai`` optional extra::

        pip install provider-contracts[openai]
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str = _DEFAULT_MODEL,
        api_version: str = _DEFAULT_API_VERSION,
        embed_deployment: str = "text-embedding-3-large",
    ) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment
        self._embed_deployment = embed_deployment

    def _to_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _to_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _parse_tool_calls(self, message: Any) -> list[ToolCallRequest]:
        parsed: list[ToolCallRequest] = []
        for tool_call in getattr(message, "tool_calls", []) or []:
            function = getattr(tool_call, "function", None)
            if function is None:
                continue

            name = getattr(function, "name", "")
            if not name:
                continue

            raw_arguments = getattr(function, "arguments", "{}") or "{}"
            arguments: dict[str, Any] = {}
            if isinstance(raw_arguments, str):
                try:
                    loaded = json.loads(raw_arguments)
                    if isinstance(loaded, dict):
                        arguments = loaded
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments

            parsed.append(ToolCallRequest(name=name, arguments=arguments))

        return parsed

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        kwargs: dict[str, object] = dict(
            model=self._deployment,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        message = response.choices[0].message
        return LLMResponse(
            content=message.content or "",
            model=self._deployment,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            tool_calls=self._parse_tool_calls(message),
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
        kwargs: dict[str, object] = dict(
            model=self._deployment,
            messages=self._to_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = self._to_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        message = response.choices[0].message
        return LLMResponse(
            content=message.content or "",
            model=self._deployment,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            tool_calls=self._parse_tool_calls(message),
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
        stream = await self._client.chat.completions.create(
            model=self._deployment,
            messages=self._to_messages(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._embed_deployment,
            input=text,
        )
        result: list[float] = response.data[0].embedding
        return result
