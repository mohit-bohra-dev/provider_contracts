"""Abstract base class for tools client providers."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A request to invoke a tool endpoint."""

    tool_name: str
    parameters: dict[str, Any] = {}


class ToolResult(BaseModel):
    """The response from a tool endpoint invocation."""

    tool_name: str
    success: bool
    data: dict[str, Any] = {}
    error: str | None = None
    """Human-readable error message when *success* is False."""


class AbstractToolsClientProvider(ABC):
    """
    Contract that every tools client adapter must satisfy.

    Concrete implementations communicate with tool endpoints via HTTP, gRPC,
    mTLS-secured channels, etc.
    """

    @abstractmethod
    async def call(self, tool: ToolCall) -> ToolResult:
        """
        Invoke a tool endpoint and return its result.

        Args:
            tool: Structured tool call with name and parameters.

        Returns:
            ToolResult with success flag, response data, and optional error.
        """
        ...

    @abstractmethod
    async def list_tools(self) -> list[str]:
        """Return the names of all available tools from the endpoint."""
        ...
