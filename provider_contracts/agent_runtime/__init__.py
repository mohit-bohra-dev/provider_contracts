"""Agent runtime providers (LangGraph / native / mock)."""

from ._base import (
    AbstractAgentRuntimeProvider,
    AgentCheckpointState,
    AgentStreamEvent,
    AgentTurnRequest,
    AgentTurnResult,
    AgentValidatorFn,
    AgentValidatorResult,
    LLMCallRecord,
    RuntimeCapabilities,
    ToolCallRecord,
    ToolExecutorFn,
    ValidationAttemptRecord,
)
from .mock import MockAgentRuntimeProvider
from .native import NativeLoopAgentRuntimeProvider

__all__ = [
    "AbstractAgentRuntimeProvider",
    "AgentCheckpointState",
    "AgentStreamEvent",
    "AgentTurnRequest",
    "AgentTurnResult",
    "AgentValidatorFn",
    "AgentValidatorResult",
    "LLMCallRecord",
    "MockAgentRuntimeProvider",
    "NativeLoopAgentRuntimeProvider",
    "RuntimeCapabilities",
    "ToolCallRecord",
    "ToolExecutorFn",
    "ValidationAttemptRecord",
]


def __getattr__(name: str) -> object:
    if name == "LangGraphAgentRuntimeProvider":
        from .langgraph import LangGraphAgentRuntimeProvider

        return LangGraphAgentRuntimeProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
