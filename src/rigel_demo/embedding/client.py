"""OpenAI-compatible Embedding 客户端。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openai import DefaultHttpxClient, OpenAI, OpenAIError

from rigel_demo.embedding.config import EmbeddingConfig, EmbeddingConfigurationError, EmbeddingFormat


class EmbeddingRequestError(RuntimeError):
    """Embedding 远程请求失败。"""


class EmbeddingResponseError(RuntimeError):
    """Embedding 返回内容无法解析。"""


class RigelEmbedding:
    """统一封装 Embedding 模型调用。"""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        openai_client: Any | None = None,
    ) -> None:
        self._config = config
        if openai_client is not None:
            self._client = openai_client
            return
        if config.format is not EmbeddingFormat.OPENAI_EMBEDDINGS:
            raise EmbeddingConfigurationError(f"不支持的 Embedding 请求格式：{config.format.value}")

        client_options: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout_seconds,
            "http_client": DefaultHttpxClient(trust_env=False),
        }
        if config.base_url:
            client_options["base_url"] = config.base_url

        try:
            self._client = OpenAI(**client_options)
        except Exception as error:
            raise EmbeddingConfigurationError(f"Embedding 客户端初始化失败：{error}") from error

    @property
    def config(self) -> EmbeddingConfig:
        """返回当前 Embedding 配置。"""

        return self._config

    def embed_query(self, text: str) -> list[float]:
        """为单条用户查询生成向量。"""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成文本向量。"""

        normalized_texts = _normalize_texts(texts)
        if not normalized_texts:
            raise EmbeddingResponseError("Embedding 输入不能为空")

        embeddings: list[list[float]] = []
        for start_index in range(0, len(normalized_texts), self._config.batch_size):
            batch = normalized_texts[start_index : start_index + self._config.batch_size]
            embeddings.extend(self._create_embedding_batch(batch))
        return embeddings

    def _create_embedding_batch(self, texts: Sequence[str]) -> list[list[float]]:
        request_body: dict[str, object] = {
            "model": self._config.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self._config.dimensions is not None:
            request_body["dimensions"] = self._config.dimensions

        try:
            response = self._client.embeddings.create(**request_body)
        except OpenAIError as error:
            raise EmbeddingRequestError(f"Embedding 调用失败：{error}") from error

        response_data = list(getattr(response, "data", []))
        if len(response_data) != len(texts):
            raise EmbeddingResponseError("Embedding 返回数量与输入数量不一致")

        response_data.sort(key=lambda item: int(getattr(item, "index", 0)))
        embeddings: list[list[float]] = []
        for item in response_data:
            raw_embedding = getattr(item, "embedding", None)
            if not isinstance(raw_embedding, list):
                raise EmbeddingResponseError("Embedding 返回向量格式不正确")
            embedding = _normalize_embedding(raw_embedding)
            if not embedding:
                raise EmbeddingResponseError("Embedding 返回空向量")
            embeddings.append(embedding)
        return embeddings


def _normalize_texts(texts: Sequence[str]) -> list[str]:
    normalized_texts = [text.strip() for text in texts if text.strip()]
    return normalized_texts


def _normalize_embedding(raw_embedding: list[object]) -> list[float]:
    embedding: list[float] = []
    for value in raw_embedding:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EmbeddingResponseError("Embedding 向量必须全部是数字")
        embedding.append(float(value))
    return embedding
