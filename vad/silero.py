"""Silero VAD — optional heavy dependency (torch). Stub until wired."""
from __future__ import annotations

from ._base import AbstractVADProvider, VADResult


class SileroVADProvider(AbstractVADProvider):
    """
    Placeholder adapter: full Silero integration requires ``torch`` + model weights.

    When implemented, classify short PCM windows and return calibrated confidences.
    """

    async def detect_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> VADResult:
        raise NotImplementedError(
            "Silero VAD is not wired yet. Set VAD_PROVIDER=webrtc or mock, "
            "or implement SileroVADProvider with torch + silero-vad assets."
        )
