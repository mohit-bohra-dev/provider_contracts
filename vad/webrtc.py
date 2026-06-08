"""WebRTC VAD adapter — lightweight frame classifier."""
from __future__ import annotations

import webrtcvad

from ._base import AbstractVADProvider, VADResult


class WebRtcVADProvider(AbstractVADProvider):
    """
    Frame-based VAD using ``webrtcvad`` (10/20/30 ms frames at 8/16/32/48 kHz).

    The final complete frame in *audio_chunk* is classified; shorter tails return
    ``is_speech=False`` with low confidence so callers can accumulate buffers.
    """

    def __init__(self, aggressiveness: int = 2, frame_ms: int = 30) -> None:
        if frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be 10, 20, or 30 for webrtcvad.")
        if not 0 <= aggressiveness <= 3:
            raise ValueError("aggressiveness must be 0..3.")
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_ms = frame_ms

    def _frame_bytes(self, sample_rate: int) -> int:
        samples = int(sample_rate * (self._frame_ms / 1000.0))
        return samples * 2  # 16-bit mono

    async def detect_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> VADResult:
        frame_bytes = self._frame_bytes(sample_rate)
        if sample_rate not in (8000, 16000, 32000, 48000):
            return VADResult(is_speech=False, confidence=0.0)
        if len(audio_chunk) < frame_bytes:
            return VADResult(is_speech=False, confidence=0.0)
        frame = audio_chunk[-frame_bytes:]
        try:
            speech = self._vad.is_speech(frame, sample_rate)
        except Exception:
            return VADResult(is_speech=True, confidence=0.55)
        return VADResult(is_speech=speech, confidence=1.0 if speech else 0.0)
