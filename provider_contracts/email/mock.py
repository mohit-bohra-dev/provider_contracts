"""In-memory mock email provider — logs to stdout, for dev/tests."""
import logging
import uuid

from ._base import AbstractEmailProvider, EmailMessage, EmailResult

logger = logging.getLogger(__name__)


class MockEmailProvider(AbstractEmailProvider):
    """Pretends to send emails; logs the content instead of delivering it."""

    def __init__(self) -> None:
        self.sent: list[EmailMessage] = []
        """All messages 'sent' — available for test assertions."""

    async def send(self, message: EmailMessage) -> EmailResult:
        self.sent.append(message)
        logger.info(
            "[MockEmail] to=%s | subject=%r | html_length=%d",
            message.to,
            message.subject,
            len(message.html),
        )
        return EmailResult(
            message_id=str(uuid.uuid4()),
            provider="mock",
            accepted=list(message.to),
        )
