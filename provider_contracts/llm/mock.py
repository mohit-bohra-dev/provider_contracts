"""In-memory mock LLM provider — zero API calls, for dev/tests."""
import random
from collections.abc import AsyncIterator

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_MOCK_RESPONSE = (
    "This is a mock LLM response. "
    "Set LLM_PROVIDER to a real provider to get actual AI output."
)


class MockLLMProvider(AbstractLLMProvider):
    """Deterministic mock — returns canned responses instantly."""

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        return LLMResponse(
            content=_MOCK_RESPONSE,
            model="mock",
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": 20},
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
        if tools:
            first_tool = tools[0]
            return LLMResponse(
                content="",
                model="mock",
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                tool_calls=[ToolCallRequest(name=first_tool.name, arguments={})],
            )

        last = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"[Mock] Echo: {last[:80]}",
            model="mock",
            usage={"prompt_tokens": len(last.split()), "completion_tokens": 10},
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        return self._stream_impl()

    async def _stream_impl(self) -> AsyncIterator[str]:
        words = _MOCK_RESPONSE.split()
        for word in words:
            yield word + " "

    async def embed(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding (1536-dim)
        rng = random.Random(len(text))
        return [rng.uniform(-1, 1) for _ in range(1536)]
