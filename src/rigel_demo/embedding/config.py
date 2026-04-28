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
DEFAULT_OPENAI_EMBEDDINGS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 60.0
DEFAULT_EMBEDDING_BATCH_SIZE = 64


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
        provider = _read_string(config_data, "provider", default="openai").lower()
        embedding_format = _read_format(config_data)
        model = _require_string(config_data, "model")
        api_key = _require_string(config_data, "api_key")
        base_url = _read_base_url(config_data, provider)
        dimensions = _read_optional_int(config_data, "dimensions")
        timeout_seconds = _read_float(config_data, "timeout_seconds", DEFAULT_EMBEDDING_TIMEOUT_SECONDS)
        batch_size = _read_int(config_data, "batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)

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
    format_value = _read_string(config_data, "format", default=EmbeddingFormat.OPENAI_EMBEDDINGS.value).lower()
    try:
        return EmbeddingFormat(format_value)
    except ValueError as error:
        supported_values = ", ".join(embedding_format.value for embedding_format in EmbeddingFormat)
        raise EmbeddingConfigurationError(f"embedding.format 仅支持：{supported_values}") from error


def _read_base_url(config_data: dict[str, Any], provider: str) -> str | None:
    custom_base_url = _read_string(config_data, "base_url")
    if custom_base_url:
        return custom_base_url
    if provider == "openai":
        return None
    raise EmbeddingConfigurationError("自定义 Embedding 提供商必须配置 embedding.base_url")


def _require_string(config_data: dict[str, Any], name: str) -> str:
    value = _read_string(config_data, name)
    if value:
        return value
    raise EmbeddingConfigurationError(f"缺少必要配置：embedding.{name}")


def _read_string(config_data: dict[str, Any], name: str, *, default: str | None = None) -> str | None:
    value = config_data.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是字符串")
    stripped_value = value.strip()
    return stripped_value or default


def _read_float(config_data: dict[str, Any], name: str, default: float) -> float:
    value = config_data.get(name)
    if value is None:
        return default
    if not _is_number(value):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是数字")
    parsed_value = float(value)
    if parsed_value <= 0:
        raise EmbeddingConfigurationError(f"embedding.{name} 必须大于 0")
    return parsed_value


def _read_int(config_data: dict[str, Any], name: str, default: int) -> int:
    value = config_data.get(name)
    if value is None:
        return default
    parsed_value = _require_positive_int(value, name)
    if parsed_value > 2048:
        raise EmbeddingConfigurationError(f"embedding.{name} 不能大于 2048")
    return parsed_value


def _read_optional_int(config_data: dict[str, Any], name: str) -> int | None:
    value = config_data.get(name)
    if value is None:
        return None
    return _require_positive_int(value, name)


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EmbeddingConfigurationError(f"embedding.{name} 必须是整数")
    if value <= 0:
        raise EmbeddingConfigurationError(f"embedding.{name} 必须大于 0")
    return value


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
