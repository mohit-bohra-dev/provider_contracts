"""Contract tests for agent_runtime adapters."""

from __future__ import annotations

import json

import pytest

from provider_contracts.agent_runtime import (
    AgentTurnRequest,
    AgentValidatorResult,
    MockAgentRuntimeProvider,
    NativeLoopAgentRuntimeProvider,
    ToolCallRecord,
)
from provider_contracts.llm import LLMMessage, ToolCallRequest, ToolDefinition
from provider_contracts.llm.mock import MockLLMProvider
from provider_contracts.tools_client import ToolCall, ToolResult
from provider_contracts.tools_client.mock import MockToolsClientProvider


class _ScriptedLLM(MockLLMProvider):
    def __init__(self) -> None:
        self._n = 0

    async def chat(self, messages, *, temperature=0.7, max_tokens=2048, json_mode=False, tools=None):  # type: ignore[no-untyped-def]
        from provider_contracts.llm import LLMResponse

        self._n += 1
        if tools and self._n == 1:
            return LLMResponse(
                content="",
                model="scripted",
                tool_calls=[
                    ToolCallRequest(name="search_docs", arguments={"query": "x"}),
                    ToolCallRequest(name="search_sse_apis", arguments={"query": "y"}),
                ],
            )
        return LLMResponse(
            content=json.dumps({"answer": "done", "confidence": 0.9}),
            model="scripted",
        )


@pytest.mark.asyncio
async def test_mock_runtime() -> None:
    rt = MockAgentRuntimeProvider(raw_text='{"ok":true}')
    result = await rt.run(
        AgentTurnRequest(messages=[LLMMessage(role="user", content="hi")]),
        llm=MockLLMProvider(),
        tools=MockToolsClientProvider(),
    )
    assert result.raw_text == '{"ok":true}'


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_cls", [NativeLoopAgentRuntimeProvider])
async def test_parallel_tools_and_validate(runtime_cls: type) -> None:
    rt = runtime_cls()
    tools = MockToolsClientProvider(
        {
            "search_docs": ToolResult(tool_name="search_docs", success=True, data={"n": 1}),
            "search_sse_apis": ToolResult(
                tool_name="search_sse_apis", success=True, data={"n": 2}
            ),
        }
    )

    async def validator(raw: str, records: list[ToolCallRecord]) -> AgentValidatorResult:
        assert len(records) == 2
        data = json.loads(raw)
        return AgentValidatorResult(is_valid=True, parsed_output=data)

    result = await rt.run(
        AgentTurnRequest(
            messages=[LLMMessage(role="user", content="find stuff")],
            tools=[
                ToolDefinition(name="search_docs", description="d", parameters={}),
                ToolDefinition(name="search_sse_apis", description="s", parameters={}),
            ],
            max_tool_rounds=3,
            max_validation_retries=1,
            parallel_tool_calls=True,
        ),
        llm=_ScriptedLLM(),
        tools=tools,
        validator=validator,
    )
    assert result.parsed_output is not None
    assert len(result.tool_calls) == 2
    assert result.rounds_used >= 1


@pytest.mark.asyncio
async def test_stream_events_native() -> None:
    rt = NativeLoopAgentRuntimeProvider()
    events = [
        ev
        async for ev in rt.stream_events(
            AgentTurnRequest(messages=[LLMMessage(role="user", content="hi")], tools=[]),
            llm=MockLLMProvider(),
            tools=MockToolsClientProvider(),
        )
    ]
    types = [e.type for e in events]
    assert "final_output" in types


@pytest.mark.asyncio
async def test_interrupt_and_resume() -> None:
    rt = NativeLoopAgentRuntimeProvider()
    interrupted = await rt.run(
        AgentTurnRequest(
            messages=[LLMMessage(role="user", content="start")],
            thread_id="t1",
            interrupt_before=["call_model"],
        ),
        llm=MockLLMProvider(),
        tools=MockToolsClientProvider(),
    )
    assert interrupted.interrupted is True
    ckpt = await rt.get_checkpoint("t1")
    assert ckpt is not None
    resumed = await rt.resume(
        "t1",
        "approved",
        llm=MockLLMProvider(),
        tools=MockToolsClientProvider(),
    )
    assert resumed.interrupted is False
    assert resumed.raw_text


@pytest.mark.asyncio
async def test_langgraph_runtime_if_installed() -> None:
    pytest.importorskip("langgraph")
    from provider_contracts.agent_runtime.langgraph import LangGraphAgentRuntimeProvider

    rt = LangGraphAgentRuntimeProvider()
    assert rt.capabilities.framework == "langgraph"
    result = await rt.run(
        AgentTurnRequest(messages=[LLMMessage(role="user", content="hi")], tools=[]),
        llm=MockLLMProvider(),
        tools=MockToolsClientProvider(),
    )
    assert result.raw_text
