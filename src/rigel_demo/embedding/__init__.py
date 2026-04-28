"""Rigel Embedding 召回能力。"""

from __future__ import annotations

from rigel_demo.embedding.client import EmbeddingRequestError, EmbeddingResponseError, RigelEmbedding
from rigel_demo.embedding.config import (
    DEFAULT_OPENAI_EMBEDDINGS_BASE_URL,
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingFormat,
)

__all__ = [
    "DEFAULT_OPENAI_EMBEDDINGS_BASE_URL",
    "EmbeddingConfig",
    "EmbeddingConfigurationError",
    "EmbeddingFormat",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "RigelEmbedding",
]
