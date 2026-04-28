"""Rigel Embedding 召回能力。"""

from __future__ import annotations

from rigel_demo.embedding.client import EmbeddingRequestError, EmbeddingResponseError, RigelEmbedding
from rigel_demo.embedding.config import (
    EmbeddingConfig,
    EmbeddingConfigurationError,
    EmbeddingFormat,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingConfigurationError",
    "EmbeddingFormat",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "RigelEmbedding",
]
