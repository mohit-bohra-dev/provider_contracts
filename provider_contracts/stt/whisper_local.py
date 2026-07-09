"""Local STT adapter backed by faster-whisper (CTranslate2 int8 quantised Whisper).

faster-whisper runs entirely on-device — no API key required.  CPU inference is
~4× faster than openai-whisper, and GPU (CUDA) inference is even faster.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
import time
from typing import Any

from ._base import AbstractSTTProvider, STTResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "base.en"
_DEFAULT_DEVICE = "cpu"
_DEFAULT_COMPUTE = "int8"


def _get_model(
    model_size: str,
    device: str,
    compute_type: str,
    cpu_threads: int,
) -> Any:
    """Load and cache the faster-whisper model (heavy — only done once)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. "
            "Run: uv pip install 'provider-contracts[whisper-local]'"
        ) from exc

    logger.info(
        "Loading faster-whisper model=%s device=%s compute=%s cpu_threads=%s",
        model_size,
        device,
        compute_type,
        cpu_threads,
    )
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


class WhisperLocalProvider(AbstractSTTProvider):
    """Transcribe audio using a locally loaded faster-whisper model.

    Inference is CPU/GPU-bound, so we offload it to a thread-pool executor
    to keep the event-loop unblocked.
    """

    def __init__(
        self,
        *,
        model_size: str = _DEFAULT_MODEL,
        device: str = _DEFAULT_DEVICE,
        compute_type: str = _DEFAULT_COMPUTE,
        cpu_threads: int = 1,
        beam_size: int = 5,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._beam_size = beam_size
        # Lazy-load model on first transcribe call so startup is instant.
        self._model: Any | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = _get_model(
                self._model_size,
                self._device,
                self._compute_type,
                self._cpu_threads,
            )
        return self._model

    def _run_transcription(
        self,
        audio: bytes,
        filename: str,
        language: str | None,
    ) -> STTResult:
        """Blocking transcription — executed inside a thread-pool executor."""
        model = self._ensure_model()

        suffix = os.path.splitext(filename)[-1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        try:
            t0 = time.perf_counter()
            segments, info = model.transcribe(
                tmp_path,
                beam_size=self._beam_size,
                language=language,
                vad_filter=True,
            )
            seg_list = list(segments)
            elapsed = time.perf_counter() - t0
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        text = " ".join(s.text.strip() for s in seg_list).strip()
        detected_lang = getattr(info, "language", None) or language

        raw: dict[str, Any] = {
            "model": self._model_size,
            "device": self._device,
            "compute_type": self._compute_type,
            "inference_seconds": round(elapsed, 3),
            "detected_language": detected_lang,
            "detected_language_probability": getattr(
                info, "language_probability", None
            ),
            "duration": getattr(info, "duration", None),
            "segments": [
                {
                    "id": s.id,
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                }
                for s in seg_list
            ],
        }

        return STTResult(
            text=text,
            model=self._model_size,
            language=detected_lang,
            duration_seconds=getattr(info, "duration", None),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # AbstractSTTProvider interface
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        mime_type: str | None = None,
        language: str | None = None,
    ) -> STTResult:
        """Transcribe *audio* bytes using the local faster-whisper model."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._run_transcription,
            audio,
            filename,
            language,
        )
