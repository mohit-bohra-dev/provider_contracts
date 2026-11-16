"""Abstract agent runtime — framework-neutral orchestration contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from provider_contracts.llm import AbstractLLMProvider, LLMMessage, ToolDefinition
from provider_contracts.tools_client import AbstractToolsClientProvider

AgentStreamEventType = Literal[
    "node_entered",
    "tool_batch_started",
    "tool_completed",
    "validation_failed",
    "checkpoint_saved",
    "final_output",
]


class ToolCallRecord(BaseModel):
    """One executed tool call (framework-neutral)."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None


class LLMCallRecord(BaseModel):
    """One LLM chat invocation during a turn."""

    round_idx: int
    phase: Literal["tool_loop", "validation_retry"] = "tool_loop"
    model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls_requested: int = 0
    finish_reason: str | None = None


class ValidationAttemptRecord(BaseModel):
    """One output-validation attempt."""

    attempt: int
    is_valid: bool
    retry_feedback: str | None = None
    latency_ms: float = 0.0


class AgentValidatorResult(BaseModel):
    """Outcome of schema / grounding validation on model text."""

    is_valid: bool
    parsed_output: Any | None = None
    retry_feedback: str | None = None


AgentValidatorFn = Callable[
    [str, list[ToolCallRecord]],
    Awaitable[AgentValidatorResult] | AgentValidatorResult,
]

ToolExecutorFn = Callable[
    [Sequence[Any]],  # list[ToolCallRequest]-like
    Awaitable[list[ToolCallRecord]],
]


class AgentTurnRequest(BaseModel):
    """Inputs for one agent turn."""

    messages: list[LLMMessage]
    tools: list[ToolDefinition] = Field(default_factory=list)
    max_tool_rounds: int = 3
    max_validation_retries: int = 1
    temperature: float = 0.0
    max_tokens: int = 2048
    json_mode: bool = True
    parallel_tool_calls: bool = True
    thread_id: str | None = None
    interrupt_before: list[str] = Field(default_factory=list)


class AgentTurnResult(BaseModel):
    """Completed turn result."""

    raw_text: str
    parsed_output: Any | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    rounds_used: int = 0
    validation_retries_used: int = 0
    interrupted: bool = False
    thread_id: str | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    validation_attempts: list[ValidationAttemptRecord] = Field(default_factory=list)
    model: str | None = None


class AgentStreamEvent(BaseModel):
    """Typed lifecycle event for UI / SSE streaming."""

    type: AgentStreamEventType
    node: str | None = None
    detail: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AgentCheckpointState(BaseModel):
    """Serializable checkpoint snapshot."""

    thread_id: str
    messages: list[LLMMessage] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_node: str | None = None


class RuntimeCapabilities(BaseModel):
    """What the active runtime adapter supports."""

    supports_checkpointing: bool = False
    supports_parallel_tools: bool = True
    supports_event_stream: bool = True
    supports_subgraphs: bool = False
    framework: str = "unknown"


class AbstractAgentRuntimeProvider(ABC):
    """Orchestration engine: ReAct tool loop + validator retry + streaming."""

    @property
    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities: ...

    @abstractmethod
    async def run(
        self,
        request: AgentTurnRequest,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AgentTurnResult: ...

    @abstractmethod
    def stream_events(
        self,
        request: AgentTurnRequest,
        *,
        llm: AbstractLLMProvider,
        tools: AbstractToolsClientProvider,
        validator: AgentValidatorFn | None = None,
        tool_executor: ToolExecutorFn | None = None,
    ) -> AsyncIterator[AgentStreamEvent]: ...

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
        """Resume after an interrupt. Default: not supported."""
        raise NotImplementedError(f"{type(self).__name__} does not support resume()")

    async def get_checkpoint(self, thread_id: str) -> AgentCheckpointState | None:
        """Return checkpoint if present."""
        return None
