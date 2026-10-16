"""Shared ReAct + validator loop used by native and LangGraph adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Sequence
from typing import Any, cast

from provider_contracts.llm import (
    AbstractLLMProvider,
    LLMMessage,
    ToolCallRequest,
)
from provider_contracts.tools_client import AbstractToolsClientProvider, ToolCall

from ._base import (
    AgentCheckpointState,
    AgentStreamEvent,
    AgentTurnRequest,
    AgentTurnResult,
    AgentValidatorFn,
    AgentValidatorResult,
    ToolCallRecord,
    ToolExecutorFn,
)


async def default_tool_executor(
    tool_calls: Sequence[ToolCallRequest],
    tools: AbstractToolsClientProvider,
) -> list[ToolCallRecord]:
    async def _one(tc: ToolCallRequest) -> ToolCallRecord:
        result = await tools.call(
            ToolCall(tool_name=tc.name, parameters=dict(tc.arguments))
        )
        summary = (
            json.dumps(result.data)[:500]
            if result.success
            else (result.error or "error")
        )
        return ToolCallRecord(
            name=tc.name,
            arguments=dict(tc.arguments),
            result_summary=summary,
            success=result.success,
        )

    return list(await asyncio.gather(*[_one(tc) for tc in tool_calls]))


async def _maybe_validate(
    validator: AgentValidatorFn | None,
    raw: str,
    records: list[ToolCallRecord],
) -> AgentValidatorResult:
    if validator is None:
        return AgentValidatorResult(is_valid=True, parsed_output=None, retry_feedback=None)
    out = validator(raw, records)
    if isinstance(out, Awaitable):
        return await cast(Awaitable[AgentValidatorResult], out)
    return out


async def run_react_loop(
    request: AgentTurnRequest,
    *,
    llm: AbstractLLMProvider,
    tools: AbstractToolsClientProvider,
    validator: AgentValidatorFn | None = None,
    tool_executor: ToolExecutorFn | None = None,
    checkpoints: dict[str, AgentCheckpointState] | None = None,
    emit: list[AgentStreamEvent] | None = None,
) -> AgentTurnResult:
    """Execute tool rounds then validate (with retries)."""
    messages = list(request.messages)
    recorded: list[ToolCallRecord] = []
    rounds_used = 0
    validation_retries = 0
    response_content = ""

    def _emit(ev: AgentStreamEvent) -> None:
        if emit is not None:
            emit.append(ev)

    async def _exec(tcs: Sequence[ToolCallRequest]) -> list[ToolCallRecord]:
        if tool_executor is not None:
            return await tool_executor(tcs)
        if request.parallel_tool_calls:
            return await default_tool_executor(tcs, tools)
        out: list[ToolCallRecord] = []
        for tc in tcs:
            out.extend(await default_tool_executor([tc], tools))
        return out

    if request.thread_id and "call_model" in request.interrupt_before:
        if checkpoints is not None:
            checkpoints[request.thread_id] = AgentCheckpointState(
                thread_id=request.thread_id,
                messages=messages,
                tool_calls=recorded,
                pending_node="call_model",
            )
            _emit(
                AgentStreamEvent(
                    type="checkpoint_saved",
                    node="call_model",
                    detail="interrupt_before call_model",
                )
            )
        return AgentTurnResult(
            raw_text="",
            tool_calls=recorded,
            interrupted=True,
            thread_id=request.thread_id,
        )

    for round_idx in range(request.max_tool_rounds):
        rounds_used = round_idx + 1
        _emit(AgentStreamEvent(type="node_entered", node="call_model", detail=f"round={round_idx}"))
        allow_tools = round_idx < request.max_tool_rounds - 1 and bool(request.tools)
        response = await llm.chat(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            json_mode=request.json_mode,
            tools=request.tools if allow_tools else None,
        )
        response_content = response.content or ""

        if not response.tool_calls:
            break

        _emit(
            AgentStreamEvent(
                type="tool_batch_started",
                node="execute_tools",
                detail=str(len(response.tool_calls)),
            )
        )
        new_tools = await _exec(response.tool_calls)
        recorded.extend(new_tools)
        for rec in new_tools:
            _emit(
                AgentStreamEvent(
                    type="tool_completed",
                    node="execute_tools",
                    detail=rec.name,
                    data={"success": rec.success},
                )
            )

        tool_results_text = "Here are the results from the tools you requested:\n"
        for item in new_tools:
            tool_results_text += f"[{item.name}]: {item.result_summary}\n"
        tool_results_text += (
            "\nIf you still need data, call another tool. "
            "Otherwise respond with the final answer."
        )
        messages.append(
            LLMMessage(
                role="assistant",
                content=response_content or "Let me check that using my tools.",
            )
        )
        messages.append(LLMMessage(role="user", content=tool_results_text))

    parsed: Any | None = None
    for attempt in range(request.max_validation_retries + 1):
        if attempt > 0:
            validation_retries = attempt
            response = await llm.chat(
                messages,
                temperature=0.1,
                max_tokens=request.max_tokens,
                json_mode=request.json_mode,
            )
            response_content = response.content or ""

        _emit(AgentStreamEvent(type="node_entered", node="validate_output"))
        result = await _maybe_validate(validator, response_content, recorded)
        if result.is_valid:
            parsed = result.parsed_output
            break

        _emit(
            AgentStreamEvent(
                type="validation_failed",
                node="validate_output",
                detail=result.retry_feedback or "invalid",
            )
        )
        if attempt >= request.max_validation_retries:
            break
        messages.append(LLMMessage(role="assistant", content=response_content))
        messages.append(
            LLMMessage(
                role="user",
                content=(
                    "Your previous response did not pass validation. "
                    f"Please fix and respond again. Error: {result.retry_feedback}"
                ),
            )
        )

    if request.thread_id and checkpoints is not None:
        checkpoints[request.thread_id] = AgentCheckpointState(
            thread_id=request.thread_id,
            messages=messages,
            tool_calls=recorded,
            pending_node=None,
        )
        _emit(AgentStreamEvent(type="checkpoint_saved", node="final_output"))

    _emit(
        AgentStreamEvent(
            type="final_output",
            node="final_output",
            detail=response_content[:200] if response_content else None,
        )
    )
    return AgentTurnResult(
        raw_text=response_content,
        parsed_output=parsed,
        tool_calls=recorded,
        rounds_used=rounds_used,
        validation_retries_used=validation_retries,
        thread_id=request.thread_id,
    )


async def stream_react_loop(
    request: AgentTurnRequest,
    *,
    llm: AbstractLLMProvider,
    tools: AbstractToolsClientProvider,
    validator: AgentValidatorFn | None = None,
    tool_executor: ToolExecutorFn | None = None,
    checkpoints: dict[str, AgentCheckpointState] | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []
    await run_react_loop(
        request,
        llm=llm,
        tools=tools,
        validator=validator,
        tool_executor=tool_executor,
        checkpoints=checkpoints,
        emit=events,
    )
    for ev in events:
        yield ev
