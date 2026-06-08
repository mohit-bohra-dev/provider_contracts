"""ElevenLabs TTS adapter."""
from typing import Any

import httpx

from ._base import AbstractTTSProvider, TTSResult

_API_BASE = "https://api.elevenlabs.io/v1"
_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel
_DEFAULT_MODEL = "eleven_turbo_v2_5"


class ElevenLabsProvider(AbstractTTSProvider):
    """Synthesise speech via the ElevenLabs REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = _DEFAULT_VOICE,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        self._api_key = api_key
        self._default_voice = voice
        self._model = model

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        language: str | None = None,
    ) -> TTSResult:
        selected_voice = voice or self._default_voice
        url = f"{_API_BASE}/text-to-speech/{selected_voice}"

        payload: dict[str, Any] = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            audio = response.content

        return TTSResult(
            audio=audio,
            model=self._model,
            voice=selected_voice,
            mime_type="audio/mpeg",
            raw={"voice_id": selected_voice, "model": self._model, "chars": len(text)},
        )
