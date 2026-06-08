"""Ollama local LLM adapter (zero-cost dev mode)."""
import json as _json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(AbstractLLMProvider):
    """Adapter for Ollama local inference server."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _to_ollama_messages(self, messages: list[LLMMessage]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _to_ollama_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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

    def _parse_tool_calls(self, message: dict[str, Any]) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            name = function.get("name", "")
            if not name:
                continue
            arguments = function.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(ToolCallRequest(name=name, arguments=arguments))
        return calls

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        msgs: list[LLMMessage] = []
        if system:
            msgs.append(LLMMessage(role="system", content=system))
        msgs.append(LLMMessage(role="user", content=prompt))
        return await self.chat(msgs, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": self._to_ollama_messages(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        if tools:
            payload["tools"] = self._to_ollama_tools(tools)

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message = data.get("message", {})
        return LLMResponse(
            content=message.get("content", ""),
            model=self._model,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            raw=data,
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
        async with (
            httpx.AsyncClient(timeout=120) as client,
            client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": self._to_ollama_messages(messages),
                    "stream": True,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    chunk = _json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]  # type: ignore[no-any-return]
