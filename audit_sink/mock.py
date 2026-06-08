"""In-memory mock audit sink — stores events in a list, for dev/tests."""
from ._base import AbstractAuditSinkProvider, AuditEvent


class MockAuditSinkProvider(AbstractAuditSinkProvider):
    """Appends events to an in-memory list — inspect ``events`` in assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def emit(self, event: AuditEvent) -> None:
        self.events.append(event)

    async def flush(self) -> None:
        pass  # No buffering in mock.
