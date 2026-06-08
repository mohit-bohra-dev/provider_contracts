"""In-memory mock STT provider — zero API calls, for dev/tests."""

from ._base import AbstractSTTProvider, STTResult

_MOCK_TRANSCRIPT = (
    "This is a mock transcription. "
    "Set STT_PROVIDER to a real provider for actual speech-to-text."
)


class MockSTTProvider(AbstractSTTProvider):
    """Returns a fixed transcript without inspecting audio."""

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        mime_type: str | None = None,
        language: str | None = None,
    ) -> STTResult:
        return STTResult(
            text=_MOCK_TRANSCRIPT,
            model="mock",
            language=language or "en",
            duration_seconds=None,
            raw={"byte_length": len(audio), "filename": filename},
        )
