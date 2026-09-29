"""AWS Bedrock Titan embedding adapter."""
from __future__ import annotations

import json
import logging
from typing import Any

try:
    import aioboto3
except ImportError:
    aioboto3 = None  # type: ignore[assignment]

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_log = logging.getLogger(__name__)

_DEFAULT_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_REGION = "us-west-2"
_DEFAULT_DIM = 1024
_MAX_EMBED_CHARS = 24000


class BedrockEmbeddingProvider(AbstractEmbeddingProvider):
    """Amazon Titan Text Embeddings V2 via Bedrock ``InvokeModel``.

    Requires ``provider-contracts[bedrock]`` (aioboto3).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        region: str = _DEFAULT_REGION,
        *,
        profile: str | None = None,
        dimensions: int = _DEFAULT_DIM,
        api_key: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        if aioboto3 is None:
            raise ImportError(
                "aioboto3 is not installed. Please install provider-contracts[bedrock]."
            )
        self._model_id = model_id
        self._region = region
        self._dimensions = dimensions
        if api_key:
            import os

            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
            _log.debug("bedrock embedding: set AWS_BEARER_TOKEN_BEDROCK")
        session_kwargs: dict[str, Any] = {}
        if profile:
            session_kwargs["profile_name"] = profile
        self._session = aioboto3.Session(**session_kwargs)
        self._client_kwargs: dict[str, Any] = {
            "region_name": region,
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "aws_session_token": session_token,
        }

    async def get_dimensions(self) -> int:
        return self._dimensions

    async def _embed_one(self, text: str) -> list[float]:
        clipped = text[:_MAX_EMBED_CHARS] if len(text) > _MAX_EMBED_CHARS else text
        body = json.dumps(
            {
                "inputText": clipped,
                "dimensions": self._dimensions,
                "normalize": True,
            }
        )
        async with self._session.client("bedrock-runtime", **self._client_kwargs) as client:
            response = await client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            raw = await response["body"].read()
        parsed = json.loads(raw)
        embedding = parsed.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Bedrock Titan response missing embedding list")
        return [float(x) for x in embedding]

    async def embed(self, text: str) -> EmbeddingResult:
        vector = await self._embed_one(text)
        return EmbeddingResult(
            vector=vector,
            model=self._model_id,
            dimensions=len(vector),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        results: list[EmbeddingResult] = []
        for text in texts:
            results.append(await self.embed(text))
        return results
