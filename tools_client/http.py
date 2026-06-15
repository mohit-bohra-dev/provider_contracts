"""HTTP tools client adapter — calls tool endpoints via httpx."""
from __future__ import annotations

from typing import Any

import httpx

from ._base import AbstractToolsClientProvider, ToolCall, ToolResult

_DEFAULT_BASE_URL = "http://localhost:8001"
_DEFAULT_TIMEOUT = 30


class HttpToolsClientProvider(AbstractToolsClientProvider):
    """Calls tool endpoints over HTTP with Bearer-token auth.

    Requires ``httpx`` (already a core dependency of provider-contracts[ollama]).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        token: str = "",
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout

    async def call(self, tool: ToolCall) -> ToolResult:
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            ) as client:
                response = await client.post(
                    f"/tools/{tool.tool_name}",
                    json=tool.parameters,
                )
                response.raise_for_status()
                raw: Any = response.json()
                # ToolResult.data expects dict; wrap list responses
                data: dict[str, Any] = (
                    raw if isinstance(raw, dict) else {"results": raw}
                )
                return ToolResult(
                    tool_name=tool.tool_name,
                    success=True,
                    data=data,
                )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                tool_name=tool.tool_name,
                success=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}",
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                tool_name=tool.tool_name,
                success=False,
                error=f"Request failed: {exc}",
            )

    async def list_tools(self) -> list[str]:
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        ) as client:
            response = await client.get("/tools")
            response.raise_for_status()
            result: list[str] = response.json()
            return result
