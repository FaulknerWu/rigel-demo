"""从仓库 `.rigel/config.json` 加载 Embedding 运行配置。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rigel_demo.config_document import ConfigDocumentErrorMessages, ConfigFieldReader, read_config_document

EMBEDDING_CONFIG_SECTION_NAME = "embedding"
MAX_EMBEDDING_BATCH_SIZE = 2048


class EmbeddingFormat(StrEnum):
    """Embedding 请求格式。"""

    OPENAI_EMBEDDINGS = "openai_embeddings"


class EmbeddingConfigurationError(ValueError):
    """Embedding 配置不可用。"""


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Embedding 客户端所需配置。"""

    provider: str
    format: EmbeddingFormat
    model: str
    api_key: str
    base_url: str | None
    dimensions: int | None
    timeout_seconds: float
    batch_size: int

    @classmethod
    def from_repository(cls, repository_path: Path) -> "EmbeddingConfig":
        """从目标仓库 `.rigel/config.json` 读取 Embedding 配置。"""

        config_data = _read_embedding_config(repository_path)
        reader = ConfigFieldReader(
            data=config_data,
            section=EMBEDDING_CONFIG_SECTION_NAME,
            error_type=EmbeddingConfigurationError,
        )
        provider = reader.required_string("provider").lower()
        embedding_format = _read_format(reader)
        model = reader.required_string("model")
        api_key = reader.required_string("api_key")
        base_url = _read_base_url(reader, provider)
        dimensions = reader.nullable_positive_int("dimensions")
        timeout_seconds = reader.positive_float("timeout_seconds")
        batch_size = _read_batch_size(reader)

        return cls(
            provider=provider,
            format=embedding_format,
            model=model,
            api_key=api_key,
            base_url=base_url,
            dimensions=dimensions,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
        )


def _read_embedding_config(repository_path: Path) -> dict[str, Any]:
    config_document = read_config_document(
        repository_path,
        messages=ConfigDocumentErrorMessages(
            missing="缺少 Embedding 配置文件：{config_path}",
            read="读取 Embedding 配置文件失败：{config_path}",
            invalid_json="Embedding 配置文件不是合法 JSON：{config_path}",
            root="Embedding 配置文件根节点必须是 JSON 对象",
        ),
        error_type=EmbeddingConfigurationError,
    )
    embedding_config = config_document.get(EMBEDDING_CONFIG_SECTION_NAME)
    if not isinstance(embedding_config, dict):
        raise EmbeddingConfigurationError("Embedding 配置文件必须包含对象字段：embedding")
    return embedding_config


def _read_format(reader: ConfigFieldReader) -> EmbeddingFormat:
    format_value = reader.required_string("format").lower()
    try:
        return EmbeddingFormat(format_value)
    except ValueError as error:
        supported_values = ", ".join(embedding_format.value for embedding_format in EmbeddingFormat)
        raise EmbeddingConfigurationError(f"embedding.format 仅支持：{supported_values}") from error


def _read_base_url(reader: ConfigFieldReader, provider: str) -> str | None:
    custom_base_url = reader.nullable_string("base_url")
    if custom_base_url:
        return custom_base_url
    if provider == "openai":
        return None
    raise EmbeddingConfigurationError("自定义 Embedding 提供商必须配置 embedding.base_url")


def _read_batch_size(reader: ConfigFieldReader) -> int:
    value = reader.positive_int("batch_size")
    if value > MAX_EMBEDDING_BATCH_SIZE:
        raise EmbeddingConfigurationError(f"embedding.batch_size 不能大于 {MAX_EMBEDDING_BATCH_SIZE}")
    return value
