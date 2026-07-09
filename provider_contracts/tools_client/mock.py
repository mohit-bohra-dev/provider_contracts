"""In-memory mock tools client — returns canned responses, for dev/tests."""
from __future__ import annotations

from ._base import AbstractToolsClientProvider, ToolCall, ToolResult


class MockToolsClientProvider(AbstractToolsClientProvider):
    """Canned-response mock — configure via ``responses`` dict.

    Missing tool names return a failure ``ToolResult``.
    """

    def __init__(self, responses: dict[str, ToolResult] | None = None) -> None:
        self._responses: dict[str, ToolResult] = dict(responses) if responses else {}
        self._available_tools: list[str] = list(self._responses.keys())

    def register(self, tool_name: str, result: ToolResult) -> None:
        """Helper for test setup — not part of the ABC."""
        self._responses[tool_name] = result
        if tool_name not in self._available_tools:
            self._available_tools.append(tool_name)

    async def call(self, tool: ToolCall) -> ToolResult:
        if tool.tool_name in self._responses:
            return self._responses[tool.tool_name]
        return ToolResult(
            tool_name=tool.tool_name,
            success=False,
            error=f"Unknown tool: {tool.tool_name}",
        )

    async def list_tools(self) -> list[str]:
        return list(self._available_tools)
