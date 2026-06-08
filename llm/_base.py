"""Abstract base class for all LLM providers."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """Provider-agnostic tool definition exposed to the model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class LLMResponse(BaseModel):
    """Structured response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


class AbstractLLMProvider(ABC):
    """Contract that every LLM adapter must satisfy."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Single-turn completion.

        Args:
            json_mode: When True, constrain the model to produce valid JSON.
                       Provider implementations should map this to their
                       native structured-output mechanism (e.g. Ollama
                       ``format: "json"``, OpenAI ``response_format``).
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        """Multi-turn chat completion.

        Args:
            json_mode: When True, constrain the model to produce valid JSON.
            tools: Optional native function/tool definitions available to the model.
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        """Streaming chat — yields text chunks."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for the given text."""
        ...
