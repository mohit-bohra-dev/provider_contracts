"""Mock VAD — treats any non-empty PCM as speech (dev / tests)."""
from __future__ import annotations

from ._base import AbstractVADProvider, VADResult


class MockVADProvider(AbstractVADProvider):
    """Always-on speech flag for fast local iteration without DSP."""

    async def detect_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> VADResult:
        if not audio_chunk:
            return VADResult(is_speech=False, confidence=0.0)
        return VADResult(is_speech=True, confidence=1.0)
