"""Abstract base class for speech-to-text providers."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class STTResult(BaseModel):
    """Structured transcript from an STT provider."""

    text: str
    model: str
    language: str | None = None
    duration_seconds: float | None = None
    raw: dict[str, Any] = {}


class AbstractSTTProvider(ABC):
    """Contract that every STT adapter must satisfy."""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        mime_type: str | None = None,
        language: str | None = None,
    ) -> STTResult:
        """
        Transcribe raw audio bytes to text.

        Args:
            audio: Raw audio file bytes.
            filename: Suggested filename (extension hints format for APIs).
            mime_type: Optional MIME type of the audio.
            language: Optional BCP-47 / ISO-639-1 hint (provider-specific).
        """
        ...
