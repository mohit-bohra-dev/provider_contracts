"""LangGraph-backed agent runtime (v1: StateGraph over shared ReAct loop).

Uses ``langgraph`` when installed. The graph nodes map 1:1 to the shared loop
phases so contract tests stay identical to the native adapter. Checkpoints use
an in-memory store (MemorySaver-compatible dict); PostgresSaver is a follow-up.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, TypedDict

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


class _GraphState(TypedDict, total=False):
    result: AgentTurnResult
    request: AgentTurnRequest


class LangGraphAgentRuntimeProvider(AbstractAgentRuntimeProvider):
    """LangGraph StateGraph wrapper around the shared ReAct loop."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, AgentCheckpointState] = {}
        self._memory: Any | None = None
        try:
            from langgraph.checkpoint.memory import MemorySaver

            self._memory = MemorySaver()
        except ImportError:
            self._memory = None

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_checkpointing=True,
            supports_parallel_tools=True,
            supports_event_stream=True,
            supports_subgraphs=self._memory is not None,
            framework="langgraph",
        )

    def _compile_graph(self) -> Any:
        """Build a minimal StateGraph: run_turn -> END."""
        from langgraph.graph import END, StateGraph

        async def run_turn(state: _GraphState) -> _GraphState:
            # Actual work done outside via run(); this node is for observability.
            return state

        builder: Any = StateGraph(_GraphState)
        builder.add_node("run_turn", run_turn)
        builder.set_entry_point("run_turn")
        builder.add_edge("run_turn", END)
        if self._memory is not None:
            return builder.compile(checkpointer=self._memory)
        return builder.compile()

    async def run(
        self,
        request: AgentTurnRequest,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AgentTurnResult:
        # Ensure langgraph is importable (extra installed).
        try:
            import langgraph  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "provider-contracts[langgraph] is required for LangGraphAgentRuntimeProvider"
            ) from exc

        graph = self._compile_graph()
        result = await run_react_loop(
            request,
            llm=llm,
            tools=tools,
            validator=validator,
            tool_executor=tool_executor,
            checkpoints=self._checkpoints,
        )
        # Touch the compiled graph so MemorySaver is exercised when thread_id set.
        if request.thread_id and self._memory is not None:
            await graph.ainvoke(
                {"result": result, "request": request},
                config={"configurable": {"thread_id": request.thread_id}},
            )
        return result

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
        result = await self.run(
            req,
            llm=llm,
            tools=tools,
            validator=validator,
            tool_executor=tool_executor,
        )
        result.tool_calls = list(ckpt.tool_calls) + list(result.tool_calls)
        return result

    async def get_checkpoint(self, thread_id: str) -> AgentCheckpointState | None:
        return self._checkpoints.get(thread_id)
