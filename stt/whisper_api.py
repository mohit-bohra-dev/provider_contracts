"""OpenAI Whisper API STT adapter (via official OpenAI SDK)."""
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI

from ._base import AbstractSTTProvider, STTResult

_DEFAULT_MODEL = "whisper-1"


class WhisperApiProvider(AbstractSTTProvider):
    """Transcribe audio using OpenAI's hosted Whisper API."""

    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    def _transcription_to_raw(self, transcription: Any) -> dict[str, Any]:
        if hasattr(transcription, "model_dump"):
            return transcription.model_dump()  # type: ignore[no-any-return]
        if isinstance(transcription, dict):
            return transcription
        return {"repr": repr(transcription)}

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        mime_type: str | None = None,
        language: str | None = None,
    ) -> STTResult:
        buffer = BytesIO(audio)
        buffer.name = filename

        kwargs: dict[str, Any] = {
            "model": self._model,
            "file": buffer,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        transcription = await self._client.audio.transcriptions.create(**kwargs)

        text = getattr(transcription, "text", None)
        if text is None and isinstance(transcription, str):
            text = transcription
        if text is None:
            text = ""

        duration = getattr(transcription, "duration", None)
        detected_language = getattr(transcription, "language", None)

        return STTResult(
            text=text,
            model=self._model,
            language=detected_language or language,
            duration_seconds=float(duration) if duration is not None else None,
            raw=self._transcription_to_raw(transcription),
        )
