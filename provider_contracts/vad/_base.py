"""Abstract voice activity detection contract."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class VADResult(BaseModel):
    """Speech likelihood for a single analysed chunk."""

    is_speech: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Normalised speech confidence.")


class AbstractVADProvider(ABC):
    """Contract implemented by concrete VAD adapters."""

    @abstractmethod
    async def detect_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> VADResult:
        """
        Analyse *audio_chunk* for voiced content.

        Args:
            audio_chunk: PCM16 mono little-endian bytes (any length; adapters may window).
            sample_rate: Sample rate in Hz (typically 16000).

        Returns:
            Structured detection result.
        """
        ...
