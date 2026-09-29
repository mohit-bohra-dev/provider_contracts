"""Bedrock embedding provider contract smoke (no live AWS)."""

from __future__ import annotations

import pytest

from provider_contracts.embedding.bedrock import BedrockEmbeddingProvider


def test_bedrock_embedding_requires_aioboto3_or_constructs() -> None:
    try:
        provider = BedrockEmbeddingProvider(region="us-west-2", dimensions=1024)
    except ImportError:
        pytest.skip("aioboto3 not installed")
    assert provider is not None
