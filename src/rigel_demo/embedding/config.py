"""从仓库 `.rigel/config.json` 加载 Embedding 运行配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Any


RIGEL_CONFIG_DIRECTORY_NAME = ".rigel"
RIGEL_CONFIG_FILE_NAME = "config.json"
EMBEDDING_CONFIG_SECTION_NAME = "embedding"
EMBEDDING_CONFIG_RELATIVE_PATH = Path(RIGEL_CONFIG_DIRECTORY_NAME) / RIGEL_CONFIG_FILE_NAME
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
        provider = _require_string(config_data, "provider").lower()
        embedding_format = _read_format(config_data)
        model = _require_string(config_data, "model")
        api_key = _require_string(config_data, "api_key")
        base_url = _read_base_url(config_data, provider)
        dimensions = _read_nullable_positive_int(config_data, "dimensions")
        timeout_seconds = _read_positive_float(config_data, "timeout_seconds")
        batch_size = _read_batch_size(config_data)

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
    config_path = repository_path / EMBEDDING_CONFIG_RELATIVE_PATH
    if not config_path.exists():
        raise EmbeddingConfigurationError(f"缺少 Embedding 配置文件：{config_path}")

    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EmbeddingConfigurationError(f"读取 Embedding 配置文件失败：{config_path}") from error
    except JSONDecodeError as error:
        raise EmbeddingConfigurationError(f"Embedding 配置文件不是合法 JSON：{config_path}") from error

    if not isinstance(config_document, dict):
        raise EmbeddingConfigurationError("Embedding 配置文件根节点必须是 JSON 对象")

    embedding_config = config_document.get(EMBEDDING_CONFIG_SECTION_NAME)
    if not isinstance(embedding_config, dict):
        raise EmbeddingConfigurationError("Embedding 配置文件必须包含对象字段：embedding")
    return embedding_config


def _read_format(config_data: dict[str, Any]) -> EmbeddingFormat:
    format_value = _require_string(config_data, "format").lower()
    try:
        return EmbeddingFormat(format_value)
    except ValueError as error:
        supported_values = ", ".join(embedding_format.value for embedding_format in EmbeddingFormat)
        raise EmbeddingConfigurationError(f"embedding.format 仅支持：{supported_values}") from error


def _read_base_url(config_data: dict[str, Any], provider: str) -> str | None:
    value = _require_field(config_data, "base_url")
    if value is None:
        if provider == "openai":
            return None
        raise EmbeddingConfigurationError("自定义 Embedding 提供商必须配置 embedding.base_url")
    if not isinstance(value, str):
        raise EmbeddingConfigurationError("embedding.base_url 必须是字符串或 null")

    custom_base_url = value.strip()
    if custom_base_url:
        return custom_base_url
    if provider == "openai":
        return None
    raise EmbeddingConfigurationError("自定义 Embedding 提供商必须配置 embedding.base_url")


def _require_string(config_data: dict[str, Any], name: str) -> str:
    value = _require_field(config_data, name)
    if not isinstance(value, str):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是字符串")
    stripped_value = value.strip()
    if stripped_value:
        return stripped_value
    raise EmbeddingConfigurationError(f"缺少必要配置：embedding.{name}")


def _require_field(config_data: dict[str, Any], name: str) -> object:
    if name not in config_data:
        raise EmbeddingConfigurationError(f"缺少必要配置：embedding.{name}")
    return config_data[name]


def _read_positive_float(config_data: dict[str, Any], name: str) -> float:
    value = _require_field(config_data, name)
    if not _is_number(value):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是数字")
    parsed_value = float(value)
    if parsed_value <= 0:
        raise EmbeddingConfigurationError(f"embedding.{name} 必须大于 0")
    return parsed_value


def _read_nullable_positive_int(config_data: dict[str, Any], name: str) -> int | None:
    value = _require_field(config_data, name)
    if value is None:
        return None
    return _require_positive_int(value, name)


def _read_batch_size(config_data: dict[str, Any]) -> int:
    value = _require_positive_int(_require_field(config_data, "batch_size"), "batch_size")
    if value > MAX_EMBEDDING_BATCH_SIZE:
        raise EmbeddingConfigurationError(f"embedding.batch_size 不能大于 {MAX_EMBEDDING_BATCH_SIZE}")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是整数")
    if value <= 0:
        raise EmbeddingConfigurationError(f"embedding.{name} 必须大于 0")
    return value


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
