"""Zero-dep native ReAct loop (fallback + contract parity)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from provider_contracts.llm import AbstractLLMProvider, LLMMessage
from provider_contracts.tools_client import AbstractToolsClientProvider

from ._base import (
    AbstractAgentRuntimeProvider,
    AgentCheckpointState,
    AgentStreamEvent,
    AgentTurnRequest,
    AgentTurnResult,
    AgentValidatorFn,
    RuntimeCapabilities,
    ToolExecutorFn,
)
from ._loop import run_react_loop, stream_react_loop


class NativeLoopAgentRuntimeProvider(AbstractAgentRuntimeProvider):
    """In-process ReAct loop with MemorySaver-style dict checkpoints."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, AgentCheckpointState] = {}

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_checkpointing=True,
            supports_parallel_tools=True,
            supports_event_stream=True,
            supports_subgraphs=False,
            framework="native",
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
        return await run_react_loop(
            request,
            llm=llm,
            tools=tools,
            validator=validator,
            tool_executor=tool_executor,
            checkpoints=self._checkpoints,
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
        async for ev in stream_react_loop(
            request,
            llm=llm,
            tools=tools,
            validator=validator,
            tool_executor=tool_executor,
            checkpoints=self._checkpoints,
        ):
            yield ev

    async def resume(
        self,
        thread_id: str,
        human_input: str,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AgentTurnResult:
        ckpt = self._checkpoints.get(thread_id)
        if ckpt is None:
            raise KeyError(f"No checkpoint for thread_id={thread_id!r}")
        messages = list(ckpt.messages)
        messages.append(LLMMessage(role="user", content=human_input))
        req = AgentTurnRequest(
            messages=messages,
            tools=[],
            max_tool_rounds=1,
            thread_id=thread_id,
            interrupt_before=[],
        )
        # Preserve prior tool records via a custom path: seed checkpoint then run
        result = await run_react_loop(
            req,
            llm=llm,
            tools=tools,
            validator=validator,
            tool_executor=tool_executor,
            checkpoints=self._checkpoints,
        )
        # Prepend prior tool calls
        result.tool_calls = list(ckpt.tool_calls) + list(result.tool_calls)
        return result

    async def get_checkpoint(self, thread_id: str) -> AgentCheckpointState | None:
        return self._checkpoints.get(thread_id)
