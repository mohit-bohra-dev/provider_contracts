"""Google Gemini LLM adapter (google-genai SDK)."""
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)


class GeminiProvider(AbstractLLMProvider):
    """Adapter for Google Gemini API using the google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._embed_model = "text-embedding-004"

    def _build_history(
        self, messages: list[LLMMessage]
    ) -> tuple[str | None, list[types.Content]]:
        """Extract system instruction and convert messages to types.Content list."""
        system: str | None = None
        history: list[types.Content] = []
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            elif msg.role == "tool":
                history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Tool result: {msg.content}")],
                    )
                )
            else:
                role = "user" if msg.role == "user" else "model"
                history.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
        return system, history

    def _to_gemini_tools(self, tools: list[ToolDefinition]) -> list[types.Tool]:
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                ]
            )
            for tool in tools
        ]

    def _parse_tool_calls(self, response: Any) -> list[ToolCallRequest]:
        calls: list[ToolCallRequest] = []
        function_calls = getattr(response, "function_calls", None)
        if not function_calls:
            return calls

        for function_call in function_calls:
            name = getattr(function_call, "name", "")
            if not name:
                continue
            args = getattr(function_call, "args", {})
            arguments = args if isinstance(args, dict) else {}
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
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            # Gemini rejects json mode when tools are active
            response_mime_type="application/json" if json_mode else None,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=cfg,
        )
        return LLMResponse(
            content=response.text or "",
            model=self._model_name,
            usage={
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                or 0,
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )
                or 0,
            },
            tool_calls=self._parse_tool_calls(response),
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
        system, history = self._build_history(messages)
        gemini_tools = self._to_gemini_tools(tools) if tools else None
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            # Gemini API rejects response_mime_type when function-calling
            # tools are present — suppress json_mode in that case.
            response_mime_type="application/json" if (json_mode and not gemini_tools) else None,
            tools=gemini_tools,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=history,
            config=cfg,
        )
        return LLMResponse(
            content=response.text or "",
            model=self._model_name,
            usage={
                "prompt_tokens": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                or 0,
                "completion_tokens": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )
                or 0,
            },
            tool_calls=self._parse_tool_calls(response),
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
        system, history = self._build_history(messages)
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=history,
            config=cfg,
        ):
            if chunk.text:
                yield chunk.text

    async def embed(self, text: str) -> list[float]:
        response = await self._client.aio.models.embed_content(
            model=self._embed_model,
            contents=text,
        )
        return list(response.embeddings[0].values)
