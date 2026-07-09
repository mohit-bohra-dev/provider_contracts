"""Abstract base class for email providers."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class EmailMessage(BaseModel):
    """A structured email to be sent."""

    to: list[str]
    subject: str
    html: str
    text: str | None = None
    """Plain-text fallback (auto-generated from html if None)."""

    from_name: str | None = None
    from_email: str | None = None
    reply_to: str | None = None


class EmailResult(BaseModel):
    """Result returned after attempting to send an email."""

    message_id: str
    provider: str
    accepted: list[str]
    rejected: list[str] = []


class AbstractEmailProvider(ABC):
    """Contract that every email adapter must satisfy."""

    @abstractmethod
    async def send(self, message: EmailMessage) -> EmailResult:
        """
        Send an email.

        Args:
            message: The email to deliver.

        Returns:
            EmailResult with delivery metadata.
        """
        ...
