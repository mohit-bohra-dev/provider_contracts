"""Abstract base class for audit sink providers."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    """A structured audit log entry."""

    event_id: str
    event_type: str
    """e.g. 'chat.request', 'tool.call', 'pii.redacted'"""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = {}
    """Arbitrary structured data — must contain no raw PII."""


class AbstractAuditSinkProvider(ABC):
    """Contract that every audit sink adapter must satisfy."""

    @abstractmethod
    async def emit(self, event: AuditEvent) -> None:
        """
        Emit a single audit event to the sink.

        Implementations must be non-blocking (fire-and-forget acceptable).
        Errors should be logged but not propagated to callers.

        Args:
            event: The audit event to record.
        """
        ...

    @abstractmethod
    async def flush(self) -> None:
        """
        Flush any buffered events to the backing store.

        Called on graceful shutdown; may be a no-op for unbuffered sinks.
        """
        ...
