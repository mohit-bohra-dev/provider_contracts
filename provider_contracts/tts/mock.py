"""In-memory mock TTS provider — zero API calls, for dev/tests."""
import struct
import wave
from io import BytesIO

from ._base import AbstractTTSProvider, TTSResult

_SILENT_DURATION_S = 0.5
_SAMPLE_RATE = 22_050
_NUM_SAMPLES = int(_SAMPLE_RATE * _SILENT_DURATION_S)


def _make_silent_wav() -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(struct.pack(f"<{_NUM_SAMPLES}h", *([0] * _NUM_SAMPLES)))
    return buffer.getvalue()


_MOCK_AUDIO = _make_silent_wav()


class MockTTSProvider(AbstractTTSProvider):
    """Returns a short silent WAV without calling any external service."""

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
    ) -> TTSResult:
        return TTSResult(
            audio=_MOCK_AUDIO,
            model="mock",
            voice=voice or "mock-voice",
            mime_type="audio/wav",
            duration_seconds=_SILENT_DURATION_S,
            raw={"text_length": len(text)},
        )
