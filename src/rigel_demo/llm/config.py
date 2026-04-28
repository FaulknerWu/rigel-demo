"""从仓库 `.rigel/config.json` 加载功能级 Chat Completions 运行配置。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from json import JSONDecodeError
from pathlib import Path
from typing import Any


RIGEL_CONFIG_DIRECTORY_NAME = ".rigel"
RIGEL_CONFIG_FILE_NAME = "config.json"
LLM_CONFIG_RELATIVE_PATH = Path(RIGEL_CONFIG_DIRECTORY_NAME) / RIGEL_CONFIG_FILE_NAME
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱分析助手。回答时优先基于用户给出的代码图谱、仓库上下文与当前问题，"
    "无法从上下文确认的内容要明确说明不确定。"
)
DEFAULT_SUMMARY_SYSTEM_PROMPT = (
    "你是 Rigel 的代码图谱摘要生成器。请基于给出的节点结构信息生成一条简洁、准确、"
    "适合语义检索的中文摘要，只输出摘要正文。"
)


class LLMConfigSection(StrEnum):
    """`.rigel/config.json` 中的 LLM 功能配置段。"""

    CHAT = "chat"
    SUMMARY = "summary"


class LLMConfigurationError(ValueError):
    """LLM 配置不可用。"""


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Chat Completions 客户端所需配置。"""

    provider: str
    model: str
    api_key: str
    base_url: str | None
    timeout_seconds: float
    system_prompt: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    section: LLMConfigSection = LLMConfigSection.CHAT

    @classmethod
    def from_repository(
        cls,
        repository_path: Path,
        section: LLMConfigSection | str = LLMConfigSection.CHAT,
    ) -> "LLMConfig":
        """从目标仓库 `.rigel/config.json` 读取指定功能的 LLM 配置。"""

        normalized_section = _normalize_section(section)
        reader = _LLMConfigReader(
            data=_read_llm_config(repository_path, normalized_section),
            section=normalized_section,
        )
        provider = reader.string("provider", default="openai").lower()

        return cls(
            provider=provider,
            model=reader.required_string("model"),
            api_key=reader.required_string("api_key"),
            base_url=_read_base_url(reader, provider),
            timeout_seconds=reader.positive_float("timeout_seconds", default=DEFAULT_TIMEOUT_SECONDS),
            system_prompt=reader.string("system_prompt", default=_default_system_prompt(normalized_section)),
            temperature=reader.optional_float("temperature"),
            max_output_tokens=reader.optional_positive_int("max_output_tokens"),
            section=normalized_section,
        )


def _normalize_section(section: LLMConfigSection | str) -> LLMConfigSection:
    try:
        return LLMConfigSection(section)
    except ValueError as error:
        supported_values = ", ".join(config_section.value for config_section in LLMConfigSection)
        raise LLMConfigurationError(f"LLM 配置段仅支持：{supported_values}") from error


def _read_llm_config(repository_path: Path, section: LLMConfigSection) -> dict[str, Any]:
    config_path = repository_path / LLM_CONFIG_RELATIVE_PATH
    if not config_path.exists():
        raise LLMConfigurationError(f"缺少 {section.value} 配置文件：{config_path}")

    try:
        config_document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise LLMConfigurationError(f"读取 {section.value} 配置文件失败：{config_path}") from error
    except JSONDecodeError as error:
        raise LLMConfigurationError(f"{section.value} 配置文件不是合法 JSON：{config_path}") from error

    if not isinstance(config_document, dict):
        raise LLMConfigurationError(f"{section.value} 配置文件根节点必须是 JSON 对象")

    llm_config = config_document.get(section.value)
    if not isinstance(llm_config, dict):
        raise LLMConfigurationError(f"LLM 配置文件必须包含对象字段：{section.value}")
    return llm_config


@dataclass(frozen=True, slots=True)
class _LLMConfigReader:
    """集中处理字段类型校验。"""

    data: dict[str, Any]
    section: LLMConfigSection

    def string(self, name: str, *, default: str) -> str:
        return self.optional_string(name) or default

    def required_string(self, name: str) -> str:
        value = self.optional_string(name)
        if value:
            return value
        raise LLMConfigurationError(f"缺少必要配置：{self.field_path(name)}")

    def optional_string(self, name: str) -> str | None:
        value = self.data.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise LLMConfigurationError(f"{self.field_path(name)} 必须是字符串")
        stripped_value = value.strip()
        return stripped_value or None

    def positive_float(self, name: str, *, default: float) -> float:
        value = self.data.get(name)
        if value is None:
            return default
        if not _is_number(value):
            raise LLMConfigurationError(f"{self.field_path(name)} 必须是数字")
        parsed_value = float(value)
        if parsed_value <= 0:
            raise LLMConfigurationError(f"{self.field_path(name)} 必须大于 0")
        return parsed_value

    def optional_float(self, name: str) -> float | None:
        value = self.data.get(name)
        if value is None:
            return None
        if not _is_number(value):
            raise LLMConfigurationError(f"{self.field_path(name)} 必须是数字")
        return float(value)

    def optional_positive_int(self, name: str) -> int | None:
        value = self.data.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise LLMConfigurationError(f"{self.field_path(name)} 必须是整数")
        if value <= 0:
            raise LLMConfigurationError(f"{self.field_path(name)} 必须大于 0")
        return value

    def field_path(self, name: str) -> str:
        return f"{self.section.value}.{name}"


def _default_system_prompt(section: LLMConfigSection) -> str:
    if section is LLMConfigSection.SUMMARY:
        return DEFAULT_SUMMARY_SYSTEM_PROMPT
    return DEFAULT_CHAT_SYSTEM_PROMPT


def _read_base_url(reader: _LLMConfigReader, provider: str) -> str | None:
    custom_base_url = reader.optional_string("base_url")
    if custom_base_url:
        return custom_base_url

    if provider != "openai":
        raise LLMConfigurationError(f"自定义 Chat Completions 提供商必须配置 {reader.field_path('base_url')}")
    return None


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)
