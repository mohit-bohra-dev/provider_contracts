"""Scripted agent runtime for unit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

from provider_contracts.llm import AbstractLLMProvider
from provider_contracts.tools_client import AbstractToolsClientProvider

from ._base import (
    AbstractAgentRuntimeProvider,
    AgentStreamEvent,
    AgentTurnRequest,
    AgentTurnResult,
    AgentValidatorFn,
    RuntimeCapabilities,
    ToolCallRecord,
    ToolExecutorFn,
)


class MockAgentRuntimeProvider(AbstractAgentRuntimeProvider):
    """Returns a canned turn result (optional scripted tool calls)."""

    def __init__(
        self,
        *,
        raw_text: str = '{"answer":"ok","confidence":0.9}',
        tool_calls: list[ToolCallRecord] | None = None,
        parsed_output: object | None = None,
    ) -> None:
        self._raw_text = raw_text
        self._tool_calls = list(tool_calls or [])
        self._parsed = parsed_output

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_checkpointing=False,
            supports_parallel_tools=True,
            supports_event_stream=True,
            supports_subgraphs=False,
            framework="mock",
        )

    async def run(
        self,
        request: AgentTurnRequest,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AgentTurnResult:
        _ = (request, llm, tools, validator, tool_executor)
        return AgentTurnResult(
            raw_text=self._raw_text,
            parsed_output=self._parsed,
            tool_calls=list(self._tool_calls),
            rounds_used=1,
            validation_retries_used=0,
            thread_id=request.thread_id,
        )

    async def stream_events(  # type: ignore[override]
        self,
        request: AgentTurnRequest,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        _ = (llm, tools, validator, tool_executor)
        yield AgentStreamEvent(type="node_entered", node="call_model")
        yield AgentStreamEvent(type="final_output", node="final_output", detail=self._raw_text)
