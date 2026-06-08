"""OpenAI / Azure OpenAI embedding adapter."""
from __future__ import annotations

from openai import AsyncOpenAI

from ._base import AbstractEmbeddingProvider, EmbeddingResult

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbeddingProvider(AbstractEmbeddingProvider):
    """Wraps the OpenAI embeddings API.

    Works with Azure OpenAI by passing the Azure-specific ``base_url``
    and ``api_key`` (or using ``openai.AsyncAzureOpenAI`` externally).

    Requires the ``openai`` optional extra::

        pip install provider-contracts[openai]
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        *,
        base_url: str | None = None,
    ) -> None:
        kwargs: dict[str, str] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)  # type: ignore[arg-type]
        self._model = model

    async def get_dimensions(self) -> int:
        res = await self.embed("")
        return res.dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        vector: list[float] = response.data[0].embedding
        return EmbeddingResult(
            vector=vector,
            model=self._model,
            dimensions=len(vector),
        )

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [
            EmbeddingResult(
                vector=item.embedding,
                model=self._model,
                dimensions=len(item.embedding),
            )
            for item in response.data
        ]
