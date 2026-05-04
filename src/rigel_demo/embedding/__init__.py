"""Rigel Embedding 召回能力。"""

from __future__ import annotations

from rigel_demo.config import (
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingFormat,
    EmbeddingInputMode,
)
from rigel_demo.embedding.client import (
    EmbeddingRequestError,
    EmbeddingResponseError,
    RigelEmbedding,
    build_openai_embedding_client,
    build_rigel_embedding,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingConfigurationError",
    "EmbeddingFormat",
    "EmbeddingInputMode",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "RigelEmbedding",
    "build_openai_embedding_client",
    "build_rigel_embedding",
]
