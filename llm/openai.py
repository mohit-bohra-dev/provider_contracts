"""OpenAI GPT LLM adapter."""
import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIProvider(AbstractLLMProvider):
    """Adapter for OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._embed_model = embed_model

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _to_openai_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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
            model=self._model,
            messages=msgs,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        message = response.choices[0].message
        return LLMResponse(
            content=message.content or "",
            model=self._model,
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
            model=self._model,
            messages=self._to_openai_messages(messages),
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        response = await self._client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
        message = response.choices[0].message
        return LLMResponse(
            content=message.content or "",
            model=self._model,
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
            model=self._model,
            messages=self._to_openai_messages(messages),
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._embed_model,
            input=text,
        )
        result: list[float] = response.data[0].embedding
        return result
