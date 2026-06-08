"""Abstract base class for text-to-speech providers."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class TTSResult(BaseModel):
    """Structured audio output from a TTS provider."""

    audio: bytes
    """Raw audio bytes (typically MP3 or PCM depending on provider)."""

    model: str
    voice: str
    mime_type: str = "audio/mpeg"
    duration_seconds: float | None = None
    raw: dict[str, Any] = {}


class AbstractTTSProvider(ABC):
    """Contract that every TTS adapter must satisfy."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: The text to convert to speech.
            voice: Provider-specific voice identifier (uses provider default if None).
            speed: Playback speed multiplier (1.0 = normal).
            language: Optional BCP-47 language hint.
        """
        ...
