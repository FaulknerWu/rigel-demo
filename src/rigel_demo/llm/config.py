"""从仓库 `.rigel/config.json` 加载 LLM 运行配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Any


RIGEL_CONFIG_DIRECTORY_NAME = ".rigel"
RIGEL_CONFIG_FILE_NAME = "config.json"
LLM_CONFIG_SECTION_NAME = "llm"
LLM_CONFIG_RELATIVE_PATH = Path(RIGEL_CONFIG_DIRECTORY_NAME) / RIGEL_CONFIG_FILE_NAME
DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱分析助手。回答时优先基于用户给出的代码图谱、仓库上下文与当前问题，"
    "无法从上下文确认的内容要明确说明不确定。"
)


class LLMFormat(StrEnum):
    """LLM 请求格式。"""

    OPENAI_CHAT = "openai_chat"
    GOOGLE_GENERATE_CONTENT = "google_generate_content"
    OPENAI_RESPONSES = "openai_responses"


class LLMConfigurationError(ValueError):
    """LLM 配置不可用。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """LLM 客户端所需的最小配置。"""

    provider: str
    format: LLMFormat
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    system_prompt: str
    temperature: float | None = None
    max_output_tokens: int | None = None

    @classmethod
    def from_repository(cls, repository_path: Path) -> "LLMConfig":
        """从目标仓库 `.rigel/config.json` 读取 LLM 配置。"""

        config_data = _read_llm_config(repository_path)
        provider = _read_string(config_data, "provider", default="openai").lower()
        llm_format = _read_format(config_data, provider)
        model = _require_string(config_data, "model")
        api_key = _require_string(config_data, "api_key")
        base_url = _read_base_url(config_data, provider, llm_format)
        timeout_seconds = _read_float(config_data, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        temperature = _read_optional_float(config_data, "temperature")
        max_output_tokens = _read_optional_int(config_data, "max_output_tokens")
        system_prompt = _read_string(config_data, "system_prompt", default=DEFAULT_SYSTEM_PROMPT)

        return cls(
            provider=provider,
            format=llm_format,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )


def _read_llm_config(repository_path: Path) -> dict[str, Any]:
    config_path = repository_path / LLM_CONFIG_RELATIVE_PATH
    if not config_path.exists():
        raise LLMConfigurationError(f"缺少 LLM 配置文件：{config_path}")

    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise LLMConfigurationError(f"读取 LLM 配置文件失败：{config_path}") from error
    except JSONDecodeError as error:
        raise LLMConfigurationError(f"LLM 配置文件不是合法 JSON：{config_path}") from error

    if not isinstance(config_document, dict):
        raise LLMConfigurationError("LLM 配置文件根节点必须是 JSON 对象")

    llm_config = config_document.get(LLM_CONFIG_SECTION_NAME)
    if not isinstance(llm_config, dict):
        raise LLMConfigurationError("LLM 配置文件必须包含对象字段：llm")
    return llm_config


def _read_format(config_data: dict[str, Any], provider: str) -> LLMFormat:
    format_value = _read_string(config_data, "format", default=_default_format(provider).value).lower()
    try:
        return LLMFormat(format_value)
    except ValueError as error:
        supported_values = ", ".join(llm_format.value for llm_format in LLMFormat)
        raise LLMConfigurationError(f"llm.format 仅支持：{supported_values}") from error


def _default_format(provider: str) -> LLMFormat:
    if provider == "google":
        return LLMFormat.GOOGLE_GENERATE_CONTENT
    return LLMFormat.OPENAI_RESPONSES


def _read_base_url(config_data: dict[str, Any], provider: str, format: LLMFormat) -> str | None:
    custom_base_url = _read_string(config_data, "base_url")
    if custom_base_url:
        return custom_base_url

    if provider == "google" and format is LLMFormat.GOOGLE_GENERATE_CONTENT:
        return DEFAULT_GOOGLE_BASE_URL

    if provider != "openai":
        raise LLMConfigurationError("自定义提供商必须配置 llm.base_url")
    return None


def _require_string(config_data: dict[str, Any], name: str) -> str:
    value = _read_string(config_data, name)
    if value:
        return value
    raise LLMConfigurationError(f"缺少必要配置：llm.{name}")


def _read_string(config_data: dict[str, Any], name: str, *, default: str | None = None) -> str | None:
    value = config_data.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise LLMConfigurationError(f"llm.{name} 必须是字符串")
    stripped_value = value.strip()
    return stripped_value or default


def _read_float(config_data: dict[str, Any], name: str, default: float) -> float:
    value = config_data.get(name)
    if value is None:
        return default
    if not _is_number(value):
        raise LLMConfigurationError(f"llm.{name} 必须是数字")
    parsed_value = float(value)
    if parsed_value <= 0:
        raise LLMConfigurationError(f"llm.{name} 必须大于 0")
    return parsed_value


def _read_optional_float(config_data: dict[str, Any], name: str) -> float | None:
    value = config_data.get(name)
    if value is None:
        return None
    if not _is_number(value):
        raise LLMConfigurationError(f"llm.{name} 必须是数字")
    return float(value)


def _read_optional_int(config_data: dict[str, Any], name: str) -> int | None:
    value = config_data.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise LLMConfigurationError(f"llm.{name} 必须是整数")
    parsed_value = value
    if parsed_value <= 0:
        raise LLMConfigurationError(f"llm.{name} 必须大于 0")
    return parsed_value


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
